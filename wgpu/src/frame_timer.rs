//! Whole-frame GPU timing for the wgpu renderer — a small, self-contained patch
//! (not part of upstream iced) that lets an embedder show *UI rendering* cost in
//! a profiler overlay.
//!
//! Iced records the entire frame — the UI widgets **and** any custom shader
//! primitives — into one command encoder and submits it (see [`crate::Renderer::draw`]).
//! An embedder's own per-pass profiler (e.g. corsetta's `render::Profiler`) can
//! only see its *own* passes; the time Iced spends on quads/text/images is
//! invisible to it. [`FrameTimer`] brackets the whole encoder with a timestamp
//! pair (two empty render passes carrying only a begin/end timestamp), reads the
//! span back asynchronously, and publishes the smoothed millisecond value to a
//! process-global slot. The embedder reads [`last_ui_gpu_ms`] and subtracts its
//! own summed passes to attribute the remainder to UI rendering + overhead.
//!
//! Requires [`wgpu::Features::TIMESTAMP_QUERY`]; without it the timer is never
//! created and [`last_ui_gpu_ms`] stays `None`.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

/// Sentinel bit pattern meaning "no UI GPU timing available yet". A real ms
/// value is a finite non-negative `f32`, so it never collides with this NaN.
const UNSET: u32 = u32::MAX;
static UI_GPU_MS_BITS: AtomicU32 = AtomicU32::new(UNSET);

/// Smoothed GPU time (in ms) Iced spent rendering the most recently *timed*
/// frame — the full submitted encoder, including any embedded shader primitives.
///
/// Returns `None` when the device lacks `TIMESTAMP_QUERY` or no frame has been
/// timed yet. Embedders that also time their own passes can subtract that sum
/// from this total to isolate the UI/overhead portion. See corsetta's stats HUD.
pub fn last_ui_gpu_ms() -> Option<f32> {
    match UI_GPU_MS_BITS.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f32::from_bits(bits)),
    }
}

/// Two timestamp slots: begin (0) and end (1) of the frame's command encoder.
const QUERY_COUNT: u32 = 2;

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
    ms: f32,                   // exponentially-smoothed frame GPU time
    last_raw: [u64; 2],        // last frame's raw begin/end ticks
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
            last_raw: [0; 2],
        }
    }

    /// Empty render pass carrying only the begin timestamp — record it first in
    /// the encoder. `target` is loaded + stored unchanged (a no-op we only need
    /// because WebGPU timestamps must live inside a render pass).
    pub(crate) fn begin(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        self.marker(encoder, target, Some(0), None);
    }

    /// Empty render pass carrying only the end timestamp — record it last, after
    /// all UI + primitive passes.
    pub(crate) fn end(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        self.marker(encoder, target, None, Some(1));
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
            let t0 = u64::from_le_bytes(view[0..8].try_into().unwrap());
            let t1 = u64::from_le_bytes(view[8..16].try_into().unwrap());
            // Unchanged raw ticks ⇒ nothing new was recorded; treat as 0.
            let ran = t0 != self.last_raw[0] || t1 != self.last_raw[1];
            self.last_raw = [t0, t1];
            let sample = if ran && t1 > t0 {
                (t1 - t0) as f32 * self.period_ns / 1.0e6
            } else {
                0.0
            };
            self.ms = if self.ms == 0.0 {
                sample
            } else {
                self.ms * 0.85 + sample * 0.15
            };
            UI_GPU_MS_BITS.store(self.ms.to_bits(), Ordering::Relaxed);
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
