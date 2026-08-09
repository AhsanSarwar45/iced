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

// corsetta patch — how long the cap above actually held the thread.
//
// The sleep+spin that enforces the cap runs in the redraw loop, outside every span
// an embedder can record: it is not its `prepare`, not its command recording, not
// GPU time, and not the present block either. An embedder that reports leftover
// frame time as "present/idle" therefore mislabels this as a vsync wait — and,
// because the pause shrinks by exactly as much as the frame's work grows, it makes
// that row swing on every wobble in work. Publishing it lets the pause be shown as
// what it is: deliberate pacing.
const UNSET_PACED: u32 = u32::MAX;
static PACED_MS_BITS: AtomicU32 = AtomicU32::new(UNSET_PACED);

/// Record the wall-clock ms the redraw loop was held at the frame-cap deadline.
/// Raw, one sample per redraw — a consumer can smooth, but cannot un-smooth.
/// corsetta patch.
pub fn record_paced_ms(ms: f32) {
    PACED_MS_BITS.store(ms.to_bits(), Ordering::Relaxed);
}

/// Wall-clock ms the last redraw spent waiting for the frame-cap deadline. `None`
/// until the first paced frame; `Some(0.0)` when uncapped or already late.
/// corsetta patch.
pub fn last_paced_ms() -> Option<f32> {
    match PACED_MS_BITS.load(Ordering::Relaxed) {
        UNSET_PACED => None,
        bits => Some(f32::from_bits(bits)),
    }
}

// corsetta patch — the gap BETWEEN redraws.
//
// A frame's wall time is measured tick to tick, but the redraw loop only starts
// timing anything once `RedrawRequested` arrives. Everything before that — the
// event-loop wait, the OS event pump, the compositor's dispatch — is outside every
// span an embedder can record, so a loop that wakes late reports a long frame with
// nothing to blame. It is distinct from the frame-cap pause: that one is a sleep we
// asked for, this one is time we did not.
const UNSET_LOOP: u32 = u32::MAX;
static LOOP_WAIT_MS_BITS: AtomicU32 = AtomicU32::new(UNSET_LOOP);

/// Record the wall-clock ms between the end of the previous redraw and the start of
/// this one. Raw, one sample per redraw. corsetta patch.
pub fn record_loop_wait_ms(ms: f32) {
    LOOP_WAIT_MS_BITS.store(ms.to_bits(), Ordering::Relaxed);
}

/// Time the redraw loop spent outside its own work before this frame began. `None`
/// until the second redraw (the first has no predecessor). corsetta patch.
pub fn last_loop_wait_ms() -> Option<f32> {
    match LOOP_WAIT_MS_BITS.load(Ordering::Relaxed) {
        UNSET_LOOP => None,
        bits => Some(f32::from_bits(bits)),
    }
}

// corsetta patch — the redraw itself, first instruction to after the present.
//
// The closing half of the pair above: `redraw_ms + loop_wait_ms` tiles the whole
// period between two redraws with no gap, so an embedder can tell a frame that was
// slow inside the redraw from one that was slow before it — and a frame time larger
// than their sum from a clock that is not counting redraws.
const UNSET_REDRAW: u32 = u32::MAX;
static REDRAW_MS_BITS: AtomicU32 = AtomicU32::new(UNSET_REDRAW);

/// Record the wall-clock ms this redraw took, present included. corsetta patch.
pub fn record_redraw_ms(ms: f32) {
    REDRAW_MS_BITS.store(ms.to_bits(), Ordering::Relaxed);
}

/// How long the last redraw took, present included. corsetta patch.
pub fn last_redraw_ms() -> Option<f32> {
    match REDRAW_MS_BITS.load(Ordering::Relaxed) {
        UNSET_REDRAW => None,
        bits => Some(f32::from_bits(bits)),
    }
}
