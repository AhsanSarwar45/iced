//! Connect a window with a renderer.
use crate::core::Color;
use crate::core::backend;
use crate::core::renderer;
use crate::graphics::color;
use crate::graphics::compositor;
use crate::graphics::{self, Antialiasing, Shell, Viewport};
use crate::{Engine, Renderer};

/// A window graphics backend for iced powered by `wgpu`.
pub struct Compositor {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
    engine: Engine,
    settings: Settings,
}

/// A compositor error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    /// The surface creation failed.
    #[error("the surface creation failed: {0}")]
    SurfaceCreationFailed(#[from] wgpu::CreateSurfaceError),
    /// The surface is not compatible.
    #[error("the surface is not compatible")]
    IncompatibleSurface,
    /// No adapter was found for the options requested.
    #[error("no adapter was found for the options requested: {0:?}")]
    NoAdapterFound(String),
    /// No device request succeeded.
    #[error("no device request succeeded: {0:?}")]
    RequestDeviceFailed(Vec<(wgpu::Limits, wgpu::RequestDeviceError)>),
}

impl From<Error> for backend::Error {
    fn from(error: Error) -> Self {
        Self::GraphicsAdapterNotFound {
            backend: "wgpu",
            reason: backend::Reason::RequestFailed(error.to_string()),
        }
    }
}

impl Compositor {
    /// Requests a new [`Compositor`] with the given [`Settings`].
    ///
    /// Returns `None` if no compatible graphics adapter could be found.
    pub async fn request(
        settings: Settings,
        display: impl compositor::Display,
        compatible_window: impl compositor::Window,
        shell: Shell,
    ) -> Result<Self, Error> {
        let instance = wgpu::util::new_instance_with_webgpu_detection(wgpu::InstanceDescriptor {
            backends: settings.backends,
            flags: if cfg!(feature = "strict-assertions") {
                wgpu::InstanceFlags::debugging()
            } else {
                wgpu::InstanceFlags::empty()
            },
            ..wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display))
        })
        .await;

        log::info!("{settings:#?}");

        #[cfg(not(target_arch = "wasm32"))]
        if log::max_level() >= log::LevelFilter::Info {
            let available_adapters: Vec<_> = instance
                .enumerate_adapters(settings.backends)
                .await
                .iter()
                .map(wgpu::Adapter::get_info)
                .collect();
            log::info!("Available adapters: {available_adapters:#?}");
        }

        #[allow(unsafe_code)]
        let compatible_surface = instance
            .create_surface(wgpu::SurfaceTarget::Window(Box::new(compatible_window)))
            .ok();

        let power_preference = match settings.power_preference {
            backend::PowerPreference::None => wgpu::PowerPreference::None,
            backend::PowerPreference::LowPower => wgpu::PowerPreference::LowPower,
            backend::PowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
        };

        let adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::from_env().unwrap_or(power_preference),
            compatible_surface: compatible_surface.as_ref(),
            force_fallback_adapter: false,
        };

        let adapter = instance
            .request_adapter(&adapter_options)
            .await
            .map_err(|_error| Error::NoAdapterFound(format!("{adapter_options:?}")))?;

        log::info!("Selected: {:#?}", adapter.get_info());

        let (format, alpha_mode) = compatible_surface
            .as_ref()
            .and_then(|surface| {
                let capabilities = surface.get_capabilities(&adapter);

                let formats = capabilities.formats.iter().copied();

                log::info!("Available formats: {formats:#?}");

                const BLACKLIST: &[wgpu::TextureFormat] = &[
                    wgpu::TextureFormat::Rgb10a2Unorm,
                    wgpu::TextureFormat::Rgb10a2Uint,
                ];

                let mut formats = formats.filter(|format| {
                    format.required_features() == wgpu::Features::empty()
                        && !BLACKLIST.contains(format)
                });

                let format = if color::GAMMA_CORRECTION {
                    formats.find(wgpu::TextureFormat::is_srgb)
                } else {
                    formats.find(|format| !wgpu::TextureFormat::is_srgb(format))
                };

                let format = format.or_else(|| {
                    log::warn!("No format found!");

                    capabilities.formats.first().copied()
                });

                let alpha_modes = capabilities.alpha_modes;

                log::info!("Available alpha modes: {alpha_modes:#?}");

                let preferred_alpha =
                    if alpha_modes.contains(&wgpu::CompositeAlphaMode::PreMultiplied) {
                        wgpu::CompositeAlphaMode::PreMultiplied
                    } else {
                        wgpu::CompositeAlphaMode::Auto
                    };

                format.zip(Some(preferred_alpha))
            })
            .ok_or(Error::IncompatibleSurface)?;

        log::info!("Selected format: {format:?} with alpha mode: {alpha_mode:?}");

        #[cfg(target_arch = "wasm32")]
        let limits = [wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits())];

        #[cfg(not(target_arch = "wasm32"))]
        let limits = [
            wgpu::Limits::default().using_resolution(adapter.limits()),
            wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
        ];

        let limits = limits.into_iter().map(|limits| wgpu::Limits {
            max_bind_groups: 4,
            max_non_sampler_bindings: 2048,
            // corsetta patch: this app is native-only (no WebGL2 target), so the
            // embedded renderer is no longer bound by WebGL2's 16-sampled-texture
            // floor. Raise it so the car über-shader can bind extra maps (e.g. the
            // LTC area-light LUT) without the old binding-reuse contortions. Every
            // desktop backend (Vulkan/Metal/DX12) supports far more than this.
            max_sampled_textures_per_shader_stage: 32,
            // corsetta patch — BINDLESS materials. The binding array is a SEPARATE limit from
            // the plain sampled-texture count above, and it defaults to 0, so the array cannot
            // be declared at all without raising it. 1024 is the ceiling DX12 validation
            // enforces anyway; per-track distinct textures measure 139–430, and a race is
            // track + 20 cars, so this is sized with room and DX12 parity in mind.
            // Requested only if the adapter supports the feature (see required_features below);
            // a limit that exceeds the adapter's is clamped by the request loop.
            max_binding_array_elements_per_shader_stage: 1024,
            // corsetta patch — the interior-capture G-buffer writes 5 targets
            // (4 × Rgba16Float + 1 × R32Uint = 36 B/sample), over the 32 B default.
            // `using_resolution` does not touch this limit, so raise it to the
            // adapter's own maximum. Keep on rebase.
            max_color_attachment_bytes_per_sample: adapter
                .limits()
                .max_color_attachment_bytes_per_sample
                .max(limits.max_color_attachment_bytes_per_sample),
            // corsetta patch — THE PRECOMPUTED TRANSFER is one scene-wide storage buffer, and a
            // TRACK's is hundreds of MB (Brands Hatch: 187 MB; 36 floats per source vertex,
            // `docs/precomputed-transfer.md`). `max_storage_buffer_binding_size` defaults to
            // **128 MiB** — a WebGPU downlevel BASELINE, not hardware; this adapter reports **2 GiB**
            // — and `using_resolution` does NOT touch it (it only raises texture dimensions). Without
            // this the track's transfer fails to BIND at 128 MiB with a validation error, and the
            // whole app aborts. Same trap as `wgpu::Limits::default()` in the bake examples.
            // Keep on rebase.
            max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
            max_buffer_size: adapter.limits().max_buffer_size,
            // corsetta patch — the PROBE LADDER packs every resident probe's radiance cube into
            // one cube-array (6 layers/probe, static-light-transport.md §10 steps 9-10); the
            // 256-layer default caps that at 42 probes. Native adapters report 2048+. Keep on rebase.
            max_texture_array_layers: adapter.limits().max_texture_array_layers,
            // corsetta patch — the GPU WING-KERNEL SYNTHESIS (`render/psf_kernel_gpu.rs`) shares ONE
            // bind-group layout across every entry point in `psf_kernel_synth.wgsl`, declaring 10
            // storage buffers. wgpu counts the LAYOUT's bindings against this limit, not the subset
            // a given entry point reads, so the WebGPU baseline of 8 aborts every physical-camera
            // scene at `create_bind_group_layout`. `using_resolution` does not touch it. Mirrors
            // the headless path in lib.rs. Keep on rebase.
            max_storage_buffers_per_shader_stage: adapter
                .limits()
                .max_storage_buffers_per_shader_stage
                .max(limits.max_storage_buffers_per_shader_stage),
            ..limits
        });

        // Request adapter features only if supported (WebGL2 lacks these).
        // SHADER_F16: iced's own use. TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES:
        // lets an embedded renderer (shader widget) use 8x MSAA on
        // adapter-specific formats such as the Rgba16Float HDR target.
        // TEXTURE_COMPRESSION_BC: lets the embedded renderer upload native
        // BC1/BC3/BC7 DDS textures (AC liveries) without CPU decoding.
        // TIMESTAMP_QUERY: per-pass GPU profiling in the embedded renderer.
        // corsetta patch — keep when rebasing on upstream (the fork's patches are
        // listed in corsetta crates/corsetta-app/Cargo.toml).
        let mut required_features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
        {
            required_features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
        }
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        // PIPELINE_STATISTICS_QUERY: counter (not timer) breakdown INSIDE a pass —
        // fragment/vertex invocations and clipper in/out per draw group, which is how
        // the profiler attributes cost within the main car pass without splitting it
        // into more render passes. Vulkan/DX12 only; absent on Metal, where the
        // profiler simply reports no counters. corsetta patch — keep on rebase.
        if adapter
            .features()
            .contains(wgpu::Features::PIPELINE_STATISTICS_QUERY)
        {
            required_features |= wgpu::Features::PIPELINE_STATISTICS_QUERY;
        }
        // POLYGON_MODE_LINE: the debug wireframe views (64-66) rasterise the car/track
        // meshes as line polygons through a Line-mode SurfacePipelines variant. Requested
        // only when supported (Vulkan/DX12 have it); without it those pipelines fall back
        // to Fill (a solid render) rather than failing. corsetta patch — keep on rebase.
        if adapter.features().contains(wgpu::Features::POLYGON_MODE_LINE) {
            required_features |= wgpu::Features::POLYGON_MODE_LINE;
        }

        // corsetta patch — BINDLESS materials.
        //
        // The embedded renderer puts every texture in one binding array and indexes it with a
        // per-vertex material id, so a material stops being a draw boundary. This is NOT an
        // optimisation for its own sake: **GPU-driven rendering is impossible without it.** A
        // compute shader that generates an indirect draw list has nowhere to put a
        // `set_bind_group` — `multi_draw_indirect_count` cannot rebind resources per draw. So
        // the entire cull → LOD → compact → indirect-draw pipeline is gated on this.
        //
        // All three are requested only when the adapter has them, so a machine without them
        // still runs (the renderer keeps its per-material bind group). Vulkan and DX12 have
        // them; Metal lacks PARTIALLY_BOUND.
        // Keep when rebasing on upstream (the fork's patches are listed in
        // corsetta crates/corsetta-app/Cargo.toml).
        for f in [
            wgpu::Features::TEXTURE_BINDING_ARRAY,
            wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY,
            // TRANSMISSION — the transparent pass's dual-source blend (`color = src0 + Src1·dst`)
            // is what lets glass weight the background by a per-CHANNEL Beer-Lambert
            // transmittance instead of a scalar alpha. surface.wgsl::fs_main_blend declares
            // `@blend_src`, so the transparent pipeline does not validate without it.
            // The headless device (lib.rs) requests the same — both must, or the debug harness
            // cannot render what the app renders.
            wgpu::Features::DUAL_SOURCE_BLENDING,
            // TEXTURE_FORMAT_16BIT_NORM — the MSM16 moment shadow store (msm.wgsl,
            // docs/shadows.md) resolves into an Rgba16Unorm render target, which needs this.
            wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
            // DEPTH_CLIP_CONTROL — shadow-caster pancaking (docs/shadows.md §2.2).
            // The headless device (lib.rs) requests the same; without it the renderer
            // falls back to an extended near plane (coarser MSM16 quantisation).
            wgpu::Features::DEPTH_CLIP_CONTROL,
            // GPU-DRIVEN rendering (docs/geometry-lod.md §7a). The cull compute pass emits a
            // compacted draw list whose LENGTH the CPU never learns, so the draw needs
            // MULTI_DRAW_INDIRECT_COUNT (count read from a GPU buffer) and
            // INDIRECT_FIRST_INSTANCE (the compacted record is reached through
            // `first_instance`, which is otherwise required to be 0). Mirrored in
            // `render/gpu_cull.rs::optional_features()` — the app FEATURE-DETECTS and falls
            // back to the CPU submission loop, so a device without these still runs
            // (Metal has no MULTI_DRAW_INDIRECT_COUNT).
            wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
            wgpu::Features::INDIRECT_FIRST_INSTANCE,
        ] {
            if adapter.features().contains(f) {
                required_features |= f;
            }
        }

        let mut errors = Vec::new();

        for required_limits in limits {
            let result = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("iced_wgpu::window::compositor device descriptor"),
                    required_features,
                    required_limits: required_limits.clone(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                })
                .await;

            match result {
                Ok((device, queue)) => {
                    let engine = Engine::new(
                        &adapter,
                        device,
                        queue,
                        format,
                        settings.antialiasing,
                        shell,
                    );

                    return Ok(Compositor {
                        instance,
                        adapter,
                        format,
                        alpha_mode,
                        engine,
                        settings,
                    });
                }
                Err(error) => {
                    errors.push((required_limits, error));
                }
            }
        }

        Err(Error::RequestDeviceFailed(errors))
    }
}

/// Creates a [`Compositor`] with the given [`Settings`] and window.
pub async fn new(
    settings: Settings,
    display: impl compositor::Display,
    compatible_window: impl compositor::Window,
    shell: Shell,
) -> Result<Compositor, Error> {
    Compositor::request(settings, display, compatible_window, shell).await
}

/// Presents the given primitives with the given [`Compositor`].
pub fn present(
    renderer: &mut Renderer,
    surface: &mut wgpu::Surface<'static>,
    viewport: &Viewport,
    background_color: Color,
    on_pre_present: impl FnOnce(),
) -> Result<(), compositor::SurfaceError> {
    match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(frame) => {
            let view = &frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let _submission = renderer.present(
                Some(background_color),
                frame.texture.format(),
                view,
                viewport,
            );

            // Present the frame
            on_pre_present();
            frame.present();

            Ok(())
        }
        wgpu::CurrentSurfaceTexture::Suboptimal(_) | wgpu::CurrentSurfaceTexture::Outdated => {
            Err(compositor::SurfaceError::Outdated)
        }
        wgpu::CurrentSurfaceTexture::Timeout => Err(compositor::SurfaceError::Timeout),
        wgpu::CurrentSurfaceTexture::Occluded => Err(compositor::SurfaceError::Occluded),
        wgpu::CurrentSurfaceTexture::Lost => Err(compositor::SurfaceError::Lost),
        wgpu::CurrentSurfaceTexture::Validation => Err(compositor::SurfaceError::Other),
    }
}

impl graphics::Compositor for Compositor {
    type Renderer = Renderer;
    type Surface = wgpu::Surface<'static>;

    async fn new(
        settings: backend::Settings,
        display: impl compositor::Display,
        compatible_window: impl compositor::Window,
        shell: Shell,
    ) -> Result<Self, backend::Error> {
        if settings.backend.hardware().is_none() && !settings.backend.matches("wgpu") {
            return Err(backend::Error::GraphicsAdapterNotFound {
                backend: "wgpu",
                reason: backend::Reason::DidNotMatch {
                    preferred_backend: settings.backend,
                },
            });
        }

        let mut settings = Settings::from(settings);

        if let Some(backends) = wgpu::Backends::from_env() {
            settings.backends = backends;
        }

        if let Some(present_mode) = present_mode_from_env() {
            settings.present_mode = present_mode;
        }

        Ok(new(settings, display, compatible_window, shell).await?)
    }

    fn create_renderer(&self, settings: renderer::Settings) -> Self::Renderer {
        Renderer::new(self.engine.clone(), settings)
    }

    fn create_surface(
        &mut self,
        window: impl compositor::Window,
        width: u32,
        height: u32,
    ) -> Self::Surface {
        let mut surface = self
            .instance
            .create_surface(wgpu::SurfaceTarget::Window(Box::new(window)))
            .expect("Create surface");

        if width > 0 && height > 0 {
            self.configure_surface(&mut surface, width, height);
        }

        surface
    }

    fn configure_surface(&mut self, surface: &mut Self::Surface, width: u32, height: u32) {
        // corsetta patch — a live present-mode request (Immediate/Mailbox) may not be
        // supported by this surface; clamp to one that is so a runtime toggle can never
        // panic `configure`. Auto*/Fifo are always valid, so this only queries caps for
        // the explicit modes.
        let present_mode = supported_present_mode(surface, &self.adapter, self.settings.present_mode);
        surface.configure(
            &self.engine.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.format,
                present_mode,
                width,
                height,
                alpha_mode: self.alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 1,
            },
        );
    }

    fn information(&self) -> compositor::Information {
        let information = self.adapter.get_info();

        compositor::Information {
            adapter: information.name,
            backend: format!("{:?}", information.backend),
        }
    }

    fn present(
        &mut self,
        renderer: &mut Self::Renderer,
        surface: &mut Self::Surface,
        viewport: &Viewport,
        background_color: Color,
        on_pre_present: impl FnOnce(),
    ) -> Result<(), compositor::SurfaceError> {
        // corsetta patch — apply a LIVE present-mode request (the game's in-app vsync
        // switch) before acquiring this frame. The app can't reach the surface itself
        // (Iced owns it), so it sets a global that we honour here by reconfiguring the
        // swapchain at the viewport's current size.
        if let Some(requested) = crate::requested_present_mode() {
            if requested != self.settings.present_mode {
                self.settings.present_mode = requested;
                let (w, h) = (viewport.physical_width(), viewport.physical_height());
                if w > 0 && h > 0 {
                    self.configure_surface(surface, w, h);
                }
            }
        }
        present(
            renderer,
            surface,
            viewport,
            background_color,
            on_pre_present,
        )
    }

    fn screenshot(
        &mut self,
        renderer: &mut Self::Renderer,
        viewport: &Viewport,
        background_color: Color,
    ) -> Vec<u8> {
        renderer.screenshot(viewport, background_color)
    }
}

/// The settings of a [`Compositor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// The present mode of the [`Renderer`].
    ///
    /// [`Renderer`]: crate::Renderer
    pub present_mode: wgpu::PresentMode,

    /// The graphics backends to use.
    pub backends: wgpu::Backends,

    /// The power-usage preference for graphics adapters.
    ///
    /// By default, it is [`backend::PowerPreference::None`].
    pub power_preference: backend::PowerPreference,

    /// The antialiasing strategy that will be used for triangle primitives.
    ///
    /// By default, it is `None`.
    pub antialiasing: Option<Antialiasing>,
}

// corsetta patch: default to `PRIMARY` (Vulkan/Metal/DX12) instead of `all()`,
// which additionally offers GL. The embedded renderer requires capabilities the
// GL backend cannot provide (`textureLoad` on depth textures, sampling one mip
// while rendering another, >3-layer array render targets), and its GL
// workarounds have been removed. Selecting GL would render incorrectly, so it is
// not offered. Keep when rebasing on upstream.
impl Default for Settings {
    fn default() -> Settings {
        Settings {
            present_mode: wgpu::PresentMode::AutoVsync,
            backends: wgpu::Backends::PRIMARY,
            power_preference: backend::PowerPreference::None,
            antialiasing: None,
        }
    }
}

impl From<backend::Settings> for Settings {
    fn from(settings: backend::Settings) -> Self {
        let backends = settings
            .backend
            .hardware()
            .map(|api| match api {
                // corsetta patch: `Best` must not resolve to GL — see above.
                backend::Api::Best => wgpu::Backends::PRIMARY,
                backend::Api::Vulkan => wgpu::Backends::VULKAN,
                backend::Api::Metal => wgpu::Backends::METAL,
                backend::Api::DirectX12 => wgpu::Backends::DX12,
                backend::Api::OpenGL => wgpu::Backends::GL,
                backend::Api::WebGPU => wgpu::Backends::BROWSER_WEBGPU,
            })
            .unwrap_or(wgpu::Backends::PRIMARY);

        Self {
            present_mode: if settings.vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            antialiasing: settings.antialiasing.then_some(Antialiasing::MSAAx4),
            backends,
            power_preference: settings.power_preference,
        }
    }
}

/// Obtains a [`wgpu::PresentMode`] from the current environment
/// configuration, if set.
///
/// The value returned by this function can be changed by setting
/// the `ICED_PRESENT_MODE` env variable. The possible values are:
///
/// - `vsync` → [`wgpu::PresentMode::AutoVsync`]
/// - `no_vsync` → [`wgpu::PresentMode::AutoNoVsync`]
/// - `immediate` → [`wgpu::PresentMode::Immediate`]
/// - `fifo` → [`wgpu::PresentMode::Fifo`]
/// - `fifo_relaxed` → [`wgpu::PresentMode::FifoRelaxed`]
/// - `mailbox` → [`wgpu::PresentMode::Mailbox`]
/// Clamp a requested present mode to one the surface supports. corsetta patch.
///
/// `AutoVsync`/`AutoNoVsync`/`Fifo` are guaranteed valid by wgpu, so they pass
/// through without a capability query. Only the explicit modes (Immediate, Mailbox,
/// FifoRelaxed) are checked; an unsupported one falls back to `AutoNoVsync` (which
/// still escapes compositor vsync where possible), keeping a runtime toggle safe.
fn supported_present_mode(
    surface: &wgpu::Surface<'static>,
    adapter: &wgpu::Adapter,
    want: wgpu::PresentMode,
) -> wgpu::PresentMode {
    match want {
        wgpu::PresentMode::AutoVsync
        | wgpu::PresentMode::AutoNoVsync
        | wgpu::PresentMode::Fifo => want,
        explicit => {
            let caps = surface.get_capabilities(adapter);
            if caps.present_modes.contains(&explicit) {
                explicit
            } else {
                log::warn!(
                    "present mode {explicit:?} unsupported by surface (has {:?}); \
                     falling back to AutoNoVsync",
                    caps.present_modes
                );
                wgpu::PresentMode::AutoNoVsync
            }
        }
    }
}

pub fn present_mode_from_env() -> Option<wgpu::PresentMode> {
    let present_mode = std::env::var("ICED_PRESENT_MODE").ok()?;

    match present_mode.to_lowercase().as_str() {
        "vsync" => Some(wgpu::PresentMode::AutoVsync),
        "no_vsync" => Some(wgpu::PresentMode::AutoNoVsync),
        "immediate" => Some(wgpu::PresentMode::Immediate),
        "fifo" => Some(wgpu::PresentMode::Fifo),
        "fifo_relaxed" => Some(wgpu::PresentMode::FifoRelaxed),
        "mailbox" => Some(wgpu::PresentMode::Mailbox),
        _ => None,
    }
}
