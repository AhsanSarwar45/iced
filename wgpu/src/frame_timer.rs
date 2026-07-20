//! Whole-frame GPU timing for the wgpu renderer — a small, self-contained patch
//! (not part of upstream iced) that lets an embedder show *UI rendering* cost in
//! a profiler overlay.
//!
//! Iced records the entire frame — the UI widgets **and** any custom shader
//! primitives — into one command encoder and submits it (see [`crate::Renderer::draw`]).
//! An embedder's own per-pass profiler (e.g. corsetta's `render::Profiler`) can
//! only see its *own* passes; the time Iced spends on quads/text/images is
//! invisible to it. [`FrameTimer`] therefore records **two** nested timestamp
//! spans (each an empty render pass carrying only a begin/end timestamp):
//!
//! * the **whole encoder** → [`last_ui_gpu_ms`], and
//! * the **shader-primitive region** → [`last_primitive_gpu_ms`]: the part of
//!   [`crate::Renderer::render`] where custom primitives (i.e. the embedder's own
//!   renderer) record their passes.
//!
//! Those two are what let an embedder split a genuine **`ui`** number from a
//! catch-all "other". Iced's own drawing is `whole − primitive` — a *measured*
//! quantity, not a residual. Whatever is left inside the primitive region after
//! the embedder's own spans (`primitive − Σ spans`) is the embedder's own untimed
//! GPU work, which is a completely different thing and belongs in its own row.
//! Lumping them together (the old single-timer scheme) produced one `ui/other` bar
//! that could grow for either reason and told you nothing about which.
//!
//! Requires [`wgpu::Features::TIMESTAMP_QUERY`]; without it the timer is never
//! created and both accessors stay `None`.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

/// Sentinel bit pattern meaning "no GPU timing available yet". A real ms value is
/// a finite non-negative `f32`, so it never collides with this NaN.
const UNSET: u32 = u32::MAX;
static UI_GPU_MS_BITS: AtomicU32 = AtomicU32::new(UNSET);
static PRIM_GPU_MS_BITS: AtomicU32 = AtomicU32::new(UNSET);

/// Smoothed GPU time (in ms) Iced spent rendering the most recently *timed*
/// frame — the full submitted encoder, including any embedded shader primitives.
///
/// Returns `None` when the device lacks `TIMESTAMP_QUERY` or no frame has been
/// timed yet. Subtract [`last_primitive_gpu_ms`] to get Iced's OWN drawing cost
/// (quads/text/images/clear). See corsetta's stats HUD.
pub fn last_ui_gpu_ms() -> Option<f32> {
    match UI_GPU_MS_BITS.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f32::from_bits(bits)),
    }
}

/// Smoothed GPU time (in ms) spent inside the **custom shader primitives** of the
/// most recently timed frame — i.e. the embedder's own renderer, measured from the
/// first primitive's `render` to the last's.
///
/// `last_ui_gpu_ms() − last_primitive_gpu_ms()` is Iced's own widget drawing;
/// `last_primitive_gpu_ms() − (the embedder's summed spans)` is the embedder's
/// untimed GPU work. Returns `None` when timing is unavailable, and `Some(0.0)`
/// on a frame that drew no shader primitive at all.
pub fn last_primitive_gpu_ms() -> Option<f32> {
    match PRIM_GPU_MS_BITS.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f32::from_bits(bits)),
    }
}

/// Four timestamp slots: begin/end of the whole encoder (0, 1) and begin/end of
/// the shader-primitive region (2, 3).
const QUERY_COUNT: u32 = 4;
const FRAME_BEGIN: u32 = 0;
const FRAME_END: u32 = 1;
const PRIM_BEGIN: u32 = 2;
const PRIM_END: u32 = 3;

/// Owns the timestamp query set + async readback for whole-encoder GPU timing.
/// The renderer holds one; [`begin`](Self::begin)/[`end`](Self::end) bracket the
/// encoder, [`resolve`](Self::resolve) records the copy-back, and
/// [`drive`](Self::drive) advances the (frame-lagged) async map and publishes the
/// smoothed result to [`last_ui_gpu_ms`].
pub(crate) struct FrameTimer {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer, // QUERY_RESOLVE | COPY_SRC
    readback: wgpu::Buffer,    // MAP_READ  | COPY_DST
    period_ns: f32,            // nanoseconds per timestamp tick
    ready: Arc<AtomicBool>,    // set by the map callback when readback is mapped
    inflight: bool,            // a map cycle is outstanding
    resolved: bool,            // readback holds fresh resolved data awaiting a map
    ms: f32,                   // exponentially-smoothed whole-encoder GPU time
    prim_ms: f32,              // exponentially-smoothed shader-primitive region time
    last_raw: [u64; QUERY_COUNT as usize], // last frame's raw ticks, per slot
}

impl FrameTimer {
    /// `period_ns` = nanoseconds per timestamp tick (`queue.get_timestamp_period()`).
    pub(crate) fn new(device: &wgpu::Device, period_ns: f32) -> FrameTimer {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("iced frame timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT,
        });
        let bytes = QUERY_COUNT as u64 * 8;
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iced frame timestamp resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iced frame timestamp readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        FrameTimer {
            query_set,
            resolve_buf,
            readback,
            period_ns,
            ready: Arc::new(AtomicBool::new(false)),
            inflight: false,
            resolved: false,
            ms: 0.0,
            prim_ms: 0.0,
            last_raw: [0; QUERY_COUNT as usize],
        }
    }

    /// Empty render pass carrying only the begin timestamp — record it first in
    /// the encoder. `target` is loaded + stored unchanged (a no-op we only need
    /// because WebGPU timestamps must live inside a render pass).
    pub(crate) fn begin(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        self.marker(encoder, target, Some(FRAME_BEGIN), None);
    }

    /// Empty render pass carrying only the end timestamp — record it last, after
    /// all UI + primitive passes.
    pub(crate) fn end(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        self.marker(encoder, target, None, Some(FRAME_END));
    }

    /// Open the **shader-primitive** span — record immediately before the first
    /// custom primitive's `render` (the embedder's own passes).
    pub(crate) fn primitive_begin(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        self.marker(encoder, target, Some(PRIM_BEGIN), None);
    }

    /// Close the shader-primitive span — record immediately after the last custom
    /// primitive's `render`. With several primitive regions in a frame the begin
    /// is written once (at the first) and the end is overwritten at each, so the
    /// span covers first-begin → last-end.
    pub(crate) fn primitive_end(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        self.marker(encoder, target, None, Some(PRIM_END));
    }

    fn marker(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        begin: Option<u32>,
        end: Option<u32>,
    ) {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("iced frame timestamp"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: begin,
                end_of_pass_write_index: end,
            }),
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    /// Record the resolve + copy-to-readback for this frame's two timestamps.
    /// No-ops while a map cycle is in flight so it never touches a mapped buffer.
    pub(crate) fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.inflight {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..QUERY_COUNT, &self.resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&self.resolve_buf, 0, &self.readback, 0, QUERY_COUNT as u64 * 8);
        self.resolved = true;
    }

    /// Advance the async readback: poll, fold any ready result into the smoothed
    /// time + publish it, then re-arm the map. Call once per frame before
    /// `begin`.
    pub(crate) fn drive(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        self.collect();
        self.arm_map();
    }

    fn collect(&mut self) {
        if !self.ready.load(Ordering::Acquire) {
            return;
        }
        {
            let view = self.readback.slice(..).get_mapped_range();
            let mut raw = [0u64; QUERY_COUNT as usize];
            for (i, slot) in raw.iter_mut().enumerate() {
                *slot = u64::from_le_bytes(
                    view[i * 8..i * 8 + 8].try_into().unwrap(),
                );
            }

            // A span whose two raw ticks are byte-identical to last frame did NOT
            // run (a timestamp that actually fired always advances the GPU clock),
            // so it reads 0 rather than repeating a stale figure. That is what makes
            // "this frame drew no shader primitive" report as 0 ms instead of the
            // last frame that did.
            let span = |begin: u32, end: u32, last: &[u64; QUERY_COUNT as usize]| {
                let (b, e) = (begin as usize, end as usize);
                let ran = raw[b] != last[b] || raw[e] != last[e];
                if ran && raw[e] > raw[b] {
                    (raw[e] - raw[b]) as f32 * self.period_ns / 1.0e6
                } else {
                    0.0
                }
            };
            let frame = span(FRAME_BEGIN, FRAME_END, &self.last_raw);
            let prim = span(PRIM_BEGIN, PRIM_END, &self.last_raw);
            self.last_raw = raw;

            // Same exponential smoothing on both, so the derived `ui = frame − prim`
            // is a difference of two equally-lagged numbers rather than a beat
            // between a smoothed and an unsmoothed one.
            let smooth = |prev: f32, sample: f32| {
                if prev == 0.0 {
                    sample
                } else {
                    prev * 0.85 + sample * 0.15
                }
            };
            self.ms = smooth(self.ms, frame);
            self.prim_ms = smooth(self.prim_ms, prim);
            UI_GPU_MS_BITS.store(self.ms.to_bits(), Ordering::Relaxed);
            PRIM_GPU_MS_BITS.store(self.prim_ms.to_bits(), Ordering::Relaxed);
        }
        self.readback.unmap();
        self.ready.store(false, Ordering::Release);
        self.inflight = false;
    }

    fn arm_map(&mut self) {
        if self.inflight || !self.resolved {
            return;
        }
        self.inflight = true;
        self.resolved = false;
        let ready = self.ready.clone();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| {
                if res.is_ok() {
                    ready.store(true, Ordering::Release);
                }
            });
    }
}
