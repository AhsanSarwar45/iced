//! Wall-clock timing of the two *blocking* calls an embedder cannot see — a small,
//! self-contained patch (not part of upstream iced).
//!
//! [`crate::window::compositor::present`] is where a frame actually waits: first on
//! `Surface::get_current_texture` (no swapchain image is free — under `Fifo` this is
//! the vsync block, under `Mailbox` it is the GPU running behind), then on
//! `SurfaceTexture::present` (handing the image to the compositor). Both are CPU
//! wall time on the render thread, and both sit *outside* every span an embedder can
//! record: outside its own `prepare`/`render`, and outside any GPU timestamp.
//!
//! An embedder that reports "present/idle" as `frame − cpu − gpu` is therefore
//! guessing. It gets the number right only when nothing else is missing — and wrong,
//! silently and in the alarming direction, when something is. These two accessors
//! make the wait a **measured** quantity so the leftover can be labelled honestly as
//! unattributed rather than dressed up as a vsync wait.
//!
//! Values are published **raw**, one sample per presented frame, deliberately: a
//! consumer can always smooth, but it cannot un-smooth, and subtracting a smoothed
//! term from a raw frame time is what makes a residual oscillate. Unlike
//! [`crate::frame_timer`] this needs no device feature — it is `Instant`, not a
//! timestamp query — so it is always available.

use std::sync::atomic::{AtomicU32, Ordering};

/// Sentinel bit pattern meaning "nothing presented yet". A real ms value is a finite
/// non-negative `f32`, so it never collides with this NaN.
const UNSET: u32 = u32::MAX;
static ACQUIRE_MS_BITS: AtomicU32 = AtomicU32::new(UNSET);
static PRESENT_MS_BITS: AtomicU32 = AtomicU32::new(UNSET);

/// Raw wall-clock ms the last presented frame spent inside
/// `Surface::get_current_texture` — the block waiting for a free swapchain image.
///
/// This is the honest "we are refresh-bound" number. `None` until the first frame
/// is presented.
pub fn last_acquire_ms() -> Option<f32> {
    match ACQUIRE_MS_BITS.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f32::from_bits(bits)),
    }
}

/// Raw wall-clock ms the last presented frame spent inside `SurfaceTexture::present`
/// — queueing the finished image to the compositor. Small and boring when healthy;
/// a large value means the compositor is pushing back. `None` until the first frame
/// is presented.
pub fn last_present_ms() -> Option<f32> {
    match PRESENT_MS_BITS.load(Ordering::Relaxed) {
        UNSET => None,
        bits => Some(f32::from_bits(bits)),
    }
}

/// Publish one frame's spans. Called from `window::compositor::present` only.
pub(crate) fn record(acquire_ms: f32, present_ms: f32) {
    ACQUIRE_MS_BITS.store(acquire_ms.to_bits(), Ordering::Relaxed);
    PRESENT_MS_BITS.store(present_ms.to_bits(), Ordering::Relaxed);
}

/// A failed acquire presents nothing, so the previous frame's spans would otherwise
/// persist and be re-reported as if they were this frame's. Zero them instead.
pub(crate) fn record_no_frame(acquire_ms: f32) {
    record(acquire_ms, 0.0);
}
