//! Who paces the frame loop: the app, or the desktop compositor.
//!
//! corsetta patch. This lives in `iced_core` because it is read by **two** crates
//! that don't otherwise see each other — `iced_wgpu` (which owns the swapchain and
//! its `PresentMode`) and `iced_winit` (which owns the redraw loop). It is a global
//! because a request has to cross from application code into both of them, and this
//! is a single-window desktop app.
//!
//! # Why this exists
//!
//! On Wayland, `Window::pre_present_notify()` issues a `wl_surface.frame` request,
//! and the next `RedrawRequested` is gated on that callback firing — which the
//! compositor sends at its own cadence. Since iced's redraw loop is driven by
//! `RedrawRequested`, calling `pre_present_notify()` unconditionally pins the whole
//! render loop to compositor cadence, *above* the swapchain — so no `PresentMode`
//! can have any effect, because the app never even attempts to present faster.
//!
//! When the app asks for a non-vsync present mode it is asking to pace itself, so
//! `iced_winit` skips `pre_present_notify()` and the throttle goes away.
//!
//! Note the hard ceiling this does **not** lift: a normal windowed (composited)
//! surface is still *displayed* at most once per compositor repaint. What this
//! recovers is the app's own loop rate (and with it, the in-app frame limiter and
//! lower latency) plus the double-buffered "miss a deadline → snap to half rate"
//! cliff. Exceeding refresh for real additionally needs an `Immediate` swapchain,
//! fullscreen, and compositor tearing support.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};

/// A present mode requested by the application, as a backend-agnostic enum so
/// `iced_core` needs no `wgpu` dependency. Mirrors `wgpu::PresentMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentModeRequest {
    /// Vsync, chosen automatically.
    AutoVsync,
    /// No vsync, chosen automatically.
    AutoNoVsync,
    /// Blocking vsync.
    Fifo,
    /// Adaptive vsync (may tear after a missed deadline).
    FifoRelaxed,
    /// No vsync; may tear.
    Immediate,
    /// Triple-buffered; no tearing, non-blocking present.
    Mailbox,
}

impl PresentModeRequest {
    /// Whether this mode means "let the compositor pace us".
    ///
    /// Only the true vsync family does. The others are the app declaring that it
    /// paces itself, which is what suppresses the Wayland frame-callback throttle.
    pub fn is_compositor_paced(self) -> bool {
        matches!(
            self,
            PresentModeRequest::AutoVsync
                | PresentModeRequest::Fifo
                | PresentModeRequest::FifoRelaxed
        )
    }

    fn to_u8(self) -> u8 {
        match self {
            PresentModeRequest::AutoVsync => 0,
            PresentModeRequest::AutoNoVsync => 1,
            PresentModeRequest::Fifo => 2,
            PresentModeRequest::FifoRelaxed => 3,
            PresentModeRequest::Immediate => 4,
            PresentModeRequest::Mailbox => 5,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => PresentModeRequest::AutoVsync,
            1 => PresentModeRequest::AutoNoVsync,
            2 => PresentModeRequest::Fifo,
            3 => PresentModeRequest::FifoRelaxed,
            4 => PresentModeRequest::Immediate,
            5 => PresentModeRequest::Mailbox,
            _ => return None,
        })
    }
}

const UNSET: u8 = u8::MAX;
static REQUESTED: AtomicU8 = AtomicU8::new(UNSET);

/// Request a present mode. Picked up by the compositor on its next present, and
/// by the redraw loop when deciding whether to throttle to the compositor.
pub fn request_present_mode(mode: PresentModeRequest) {
    REQUESTED.store(mode.to_u8(), Ordering::Relaxed);
}

/// The present mode the application has requested, if any.
pub fn requested_present_mode() -> Option<PresentModeRequest> {
    PresentModeRequest::from_u8(REQUESTED.load(Ordering::Relaxed))
}

/// Whether the redraw loop should hand pacing to the compositor.
///
/// `true` (the default, when the app has requested nothing) preserves stock iced
/// behaviour — `pre_present_notify()` is called and Wayland throttles redraws to
/// the compositor's cadence.
pub fn throttle_to_compositor() -> bool {
    requested_present_mode().is_none_or(PresentModeRequest::is_compositor_paced)
}

// corsetta patch — the FPS cap, read by `iced_winit`'s redraw loop.
//
// The cap is enforced by the redraw loop itself (`RedrawRequest::At(deadline)`),
// NOT by an app-side sleep. That is the whole point: a sleep that ran once per
// `Message::Tick` was decoupled from the actual present (which happens per
// `RedrawRequested`), so under a self-pacing (non-vsync) present mode the loop
// could free-run and outrun the cap. Pacing the redraw request at the single
// synchronous per-present point makes it a single driver that cannot be bypassed.
const UNSET_CAP: u32 = u32::MAX;
static FRAME_CAP_HZ: AtomicU32 = AtomicU32::new(UNSET_CAP);

/// Request an FPS cap in Hz. `0` = uncapped. Picked up by the redraw loop each
/// frame. corsetta patch.
pub fn request_frame_cap(hz: u32) {
    FRAME_CAP_HZ.store(hz, Ordering::Relaxed);
}

/// The requested FPS cap in Hz, or `0` (uncapped) if the app has set none.
/// corsetta patch.
pub fn frame_cap_hz() -> u32 {
    match FRAME_CAP_HZ.load(Ordering::Relaxed) {
        UNSET_CAP => 0,
        hz => hz,
    }
}
