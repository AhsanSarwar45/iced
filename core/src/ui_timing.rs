//! Wall-clock timing of three CPU spans an embedder cannot otherwise see: the
//! `interface.update`/`interface.draw` calls in `iced_winit`'s redraw loop, and
//! `Renderer::present` inside `iced_wgpu`'s compositor. corsetta patch.
//!
//! This lives in `iced_core` because `iced_winit` (which owns the redraw loop and
//! times `update`/`draw`) and `iced_wgpu` (which owns the compositor and times
//! `present`) don't otherwise see each other — a global here is the one place both
//! can write to.
//!
//! Values are published **raw**, one sample per frame — no EMA, no smoothing.
//! Smoothing a subtrahend against a raw frame time is the exact bug this exists to
//! diagnose: a soak trace showed 34 ms/frame unaccounted for while every *measured*
//! span stayed flat, so the fix is to measure, not to infer-and-smooth.

use std::sync::atomic::{AtomicU32, Ordering};

const UNSET: u32 = u32::MAX;

static EVENTS_MS_BITS: AtomicU32 = AtomicU32::new(UNSET);
static DRAW_MS_BITS: AtomicU32 = AtomicU32::new(UNSET);
static SUBMIT_MS_BITS: AtomicU32 = AtomicU32::new(UNSET);
static PROGRAM_MS_BITS: AtomicU32 = AtomicU32::new(UNSET);
static ENCODE_MS_BITS: AtomicU32 = AtomicU32::new(UNSET);
static QSUBMIT_MS_BITS: AtomicU32 = AtomicU32::new(UNSET);

/// Record the summed wall-clock ms of this frame's `interface.update(...)` calls.
/// A redraw can loop more than once (layout invalidation), so the caller sums the
/// iterations before calling this rather than overwriting per-iteration.
pub fn record_events_ms(ms: f32) {
    EVENTS_MS_BITS.store(ms.to_bits(), Ordering::Relaxed);
}

/// Summed wall-clock ms the last frame's `interface.update(...)` call(s) took.
/// `None` until the first redraw.
pub fn last_events_ms() -> Option<f32> {
    match EVENTS_MS_BITS.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f32::from_bits(bits)),
    }
}

/// Record the wall-clock ms of this frame's `interface.draw(...)` call.
pub fn record_draw_ms(ms: f32) {
    DRAW_MS_BITS.store(ms.to_bits(), Ordering::Relaxed);
}

/// Wall-clock ms the last frame's `interface.draw(...)` call took. `None` until the
/// first redraw.
pub fn last_draw_ms() -> Option<f32> {
    match DRAW_MS_BITS.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f32::from_bits(bits)),
    }
}

/// Record the summed wall-clock ms of this frame's message rounds: caching the
/// widget trees, the program's `update`, and rebuilding + re-laying-out every tree.
/// Summed for the same reason as `record_events_ms`.
pub fn record_program_ms(ms: f32) {
    PROGRAM_MS_BITS.store(ms.to_bits(), Ordering::Relaxed);
}

/// Summed wall-clock ms the last frame spent in `update` + widget-tree rebuild.
/// `None` until the first frame that produced a message.
pub fn last_program_ms() -> Option<f32> {
    match PROGRAM_MS_BITS.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f32::from_bits(bits)),
    }
}

/// Record `Renderer::present`'s two halves: the encode (every widget's `prepare`
/// and `render`) and the `Queue::submit` that follows it. The submit blocks on GPU
/// back-pressure, so keeping it apart from the encode is the difference between
/// "our CPU is slow" and "we are ahead of the GPU".
pub fn record_encode_ms(encode: f32, submit: f32) {
    ENCODE_MS_BITS.store(encode.to_bits(), Ordering::Relaxed);
    QSUBMIT_MS_BITS.store(submit.to_bits(), Ordering::Relaxed);
}

/// Wall-clock ms the last frame spent encoding. `None` until the first frame.
pub fn last_encode_ms() -> Option<f32> {
    match ENCODE_MS_BITS.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f32::from_bits(bits)),
    }
}

/// Wall-clock ms the last frame spent inside `Queue::submit`. `None` until the
/// first frame.
pub fn last_queue_submit_ms() -> Option<f32> {
    match QSUBMIT_MS_BITS.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f32::from_bits(bits)),
    }
}

/// Record the wall-clock ms of this frame's `Renderer::present(...)` call.
pub fn record_submit_ms(ms: f32) {
    SUBMIT_MS_BITS.store(ms.to_bits(), Ordering::Relaxed);
}

/// Wall-clock ms the last frame's `Renderer::present(...)` call took. `None` until
/// the first frame. Timed on both of `compositor::present`'s paths (offscreen and
/// surface), so this is valid regardless of which one ran.
pub fn last_submit_ms() -> Option<f32> {
    match SUBMIT_MS_BITS.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f32::from_bits(bits)),
    }
}
