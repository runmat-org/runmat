//! Implementation methods for the GUI plot window

#[cfg(feature = "gui")]
use super::plot_overlay::{OverlayConfig, OverlayMetrics};
#[cfg(feature = "gui")]
use super::{PlotWindow, WindowConfig};
#[cfg(feature = "gui")]
use crate::core::PipelineType;
#[cfg(feature = "gui")]
use egui_winit::State as EguiState;
#[cfg(feature = "gui")]
use glam::{Mat4, Vec2, Vec3, Vec4};
#[cfg(feature = "gui")]
use runmat_time::Instant;
#[cfg(feature = "gui")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "gui")]
use std::sync::Arc;
#[cfg(feature = "gui")]
use winit::{dpi::PhysicalSize, event::Event, event_loop::EventLoop, window::WindowBuilder};
#[cfg(feature = "gui")]
impl<'window> PlotWindow<'window> {
    /// Create a new interactive plot window
    pub async fn new(config: WindowConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Create a new EventLoop (assumes this is the only EventLoop creation)
        let event_loop =
            EventLoop::new().map_err(|e| format!("Failed to create EventLoop: {e}"))?;
        let window = WindowBuilder::new()
            .with_title(&config.title)
            .with_inner_size(PhysicalSize::new(config.width, config.height))
            .with_resizable(config.resizable)
            .with_maximized(config.maximized)
            .build(&event_loop)?;
        let window = Arc::new(window);

        // Reuse shared context when available; fall back to creating a dedicated device otherwise.
        let shared_ctx = crate::context::shared_wgpu_context();
        let (instance, surface, shared_ctx) = if let Some(ctx) = shared_ctx {
            let surface = ctx.instance.create_surface(window.clone())?;
            (ctx.instance.clone(), surface, Some(ctx))
        } else {
            let instance = Arc::new(wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            }));
            let surface = instance.create_surface(window.clone())?;
            (instance, surface, None)
        };

        let (adapter, device, queue) = if let Some(ctx) = shared_ctx {
            (ctx.adapter, ctx.device, ctx.queue)
        } else {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                })
                .await
                .ok_or("Failed to request adapter")?;

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("RunMat Plot Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await?;

            (Arc::new(adapter), Arc::new(device), Arc::new(queue))
        };

        // Configure surface
        let surface_caps = surface.get_capabilities(adapter.as_ref());
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: config.width,
            height: config.height,
            present_mode: if config.vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Create depth texture
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create unified plot renderer
        let plot_renderer =
            crate::core::PlotRenderer::new(device.clone(), queue.clone(), surface_config).await?;
        let plot_overlay = crate::gui::PlotOverlay::new();

        // Setup egui with modern dark theme
        let egui_ctx = egui::Context::default();

        // Apply our beautiful modern dark theme to egui
        let theme = crate::styling::ModernDarkTheme::default();
        theme.apply_to_egui(&egui_ctx);

        let egui_state = EguiState::new(
            egui_ctx.clone(),
            egui::viewport::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            None, // egui doesn't need depth buffer
            1,
        );

        Ok(Self {
            window,
            event_loop: Some(event_loop),
            plot_renderer,
            plot_overlay,
            surface,
            depth_texture,
            depth_view,
            egui_ctx,
            egui_state,
            egui_renderer,
            config,
            mouse_position: Vec2::ZERO,
            is_mouse_over_plot: true,
            needs_initial_redraw: true,
            pixels_per_point: 1.0,
            mouse_left_down: false,
            close_signal: None,
        })
    }

    /// Add a simple line plot to the scene for testing
    pub fn add_test_plot(&mut self) {
        use crate::core::vertex_utils;

        // Create some test data
        let x_data: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y_data: Vec<f64> = x_data.iter().map(|x| x.sin()).collect();

        // Create vertices for the line plot
        let vertices =
            vertex_utils::create_line_plot(&x_data, &y_data, Vec4::new(0.0, 0.5, 1.0, 1.0));

        // Create a scene node
        let mut render_data = crate::core::RenderData {
            pipeline_type: PipelineType::Lines,
            vertices,
            indices: None,
            gpu_vertices: None,
            material: crate::core::Material::default(),
            draw_calls: vec![crate::core::DrawCall {
                vertex_offset: 0,
                vertex_count: (x_data.len() - 1) * 2, // Each line segment has 2 vertices
                index_offset: None,
                index_count: None,
                instance_count: 1,
            }],
            image: None,
        };

        // Set material color
        render_data.material.albedo = Vec4::new(0.0, 0.5, 1.0, 1.0);

        let node = crate::core::SceneNode {
            id: 0, // Will be set by scene
            name: "Test Line Plot".to_string(),
            transform: Mat4::IDENTITY,
            visible: true,
            cast_shadows: false,
            receive_shadows: false,
            axes_index: 0,
            parent: None,
            children: Vec::new(),
            render_data: Some(render_data),
            bounds: crate::core::BoundingBox::from_points(
                &x_data
                    .iter()
                    .zip(y_data.iter())
                    .map(|(&x, &y)| Vec3::new(x as f32, y as f32, 0.0))
                    .collect::<Vec<_>>(),
            ),
            lod_levels: Vec::new(),
            current_lod: 0,
        };

        self.plot_renderer.scene.add_node(node);

        // Fit camera to show the plot
        let bounds_min = Vec3::new(-1.0, -1.5, -1.0);
        let bounds_max = Vec3::new(10.0, 1.5, 1.0);
        self.plot_renderer.camera.fit_bounds(bounds_min, bounds_max);
    }

    /// Set the figure to display in this window (clears existing content)
    pub fn set_figure(&mut self, figure: crate::plots::Figure) {
        // Use the unified plot renderer
        self.plot_renderer.set_figure(figure);
    }

    /// Attach a signal that lets external callers request the window to close.
    pub fn install_close_signal(&mut self, signal: Arc<AtomicBool>) {
        self.close_signal = Some(signal);
    }

    /// Run the interactive plot window event loop
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let event_loop = self
            .event_loop
            .take()
            .ok_or("Event loop already consumed")?;
        let window = self.window.clone();
        let mut last_render_time = Instant::now();
        let close_signal = self.close_signal.clone();

        event_loop.run(move |event, target| {
            if let Some(signal) = close_signal.as_ref() {
                if signal.load(Ordering::Relaxed) {
                    target.exit();
                    return;
                }
            }
            target.set_control_flow(winit::event_loop::ControlFlow::Poll);

            // Track current modifiers for Command/Ctrl shortcuts
            static mut MODIFIERS: Option<winit::keyboard::ModifiersState> = None;

            // Handle egui events and record consumption
            let mut repaint = false;
            let mut egui_consumed = false;
            if let Event::WindowEvent { ref event, .. } = event {
                let response = self.egui_state.on_window_event(&window, event);
                repaint = response.repaint;
                egui_consumed = response.consumed;
            }
            if repaint {
                window.request_redraw();
            }

            match event {
                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::ModifiersChanged(mods),
                } if window_id == window.id() => unsafe {
                    MODIFIERS = Some(mods.state());
                },
                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::CloseRequested,
                } if window_id == window.id() => {
                    target.exit();
                }

                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::Resized(new_size),
                } if window_id == window.id() => {
                    // Resize surface and depth texture
                    if new_size.width > 0 && new_size.height > 0 {
                        self.resize(new_size.width, new_size.height);
                    }
                }

                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::RedrawRequested,
                } if window_id == window.id() => {
                    let now = Instant::now();
                    let dt = now - last_render_time;
                    last_render_time = now;

                    match self.render(dt) {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            self.resize(self.config.width, self.config.height)
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                        Err(e) => eprintln!("Render error: {e:?}"),
                    }
                }

                // Exit on Escape key for quick UX
                winit::event::Event::WindowEvent { window_id, event }
                    if window_id == window.id() =>
                {
                    if let winit::event::WindowEvent::KeyboardInput {
                        event: key_event, ..
                    } = event
                    {
                        if key_event.state == winit::event::ElementState::Pressed {
                            if let winit::keyboard::PhysicalKey::Code(
                                winit::keyboard::KeyCode::Escape,
                            ) = key_event.physical_key
                            {
                                target.exit();
                            }
                            // macOS-like Command+Q (and Ctrl+Q on other platforms) to quit
                            if let Some(text) = key_event.text {
                                if text == "\u{11}" { /* ignore control chars */ }
                            }
                            // Handle Q with Command or Control modifier
                            if let winit::keyboard::PhysicalKey::Code(
                                winit::keyboard::KeyCode::KeyQ,
                            ) = key_event.physical_key
                            {
                                let mods = unsafe {
                                    MODIFIERS.unwrap_or_else(winit::keyboard::ModifiersState::empty)
                                };
                                if mods.super_key() || mods.control_key() {
                                    target.exit();
                                }
                            }
                        }
                    }
                }

                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::MouseInput { button, state, .. },
                } if window_id == window.id() => {
                    // Allow interactions inside plot even if egui reports consumed elsewhere
                    let mut route = !egui_consumed;
                    if let Some(plot_rect) = self.plot_overlay.plot_area() {
                        let ppp = self.pixels_per_point.max(0.5);
                        let mx = self.mouse_position.x;
                        let my = self.mouse_position.y;
                        let px_min_x = plot_rect.min.x * ppp;
                        let px_min_y = plot_rect.min.y * ppp;
                        let px_w = plot_rect.width() * ppp;
                        let px_h = plot_rect.height() * ppp;
                        if mx >= px_min_x
                            && mx <= px_min_x + px_w
                            && my >= px_min_y
                            && my <= px_min_y + px_h
                        {
                            route = true;
                            if let Some(tb) = self.plot_overlay.toolbar_rect() {
                                if my >= tb.min.y * ppp && my <= tb.max.y * ppp {
                                    route = false;
                                }
                            }
                            if let Some(sb) = self.plot_overlay.sidebar_rect() {
                                if mx >= sb.min.x * ppp
                                    && mx <= sb.max.x * ppp
                                    && my >= sb.min.y * ppp
                                    && my <= sb.max.y * ppp
                                {
                                    route = false;
                                }
                            }
                        }
                    }
                    if route {
                        // Track left button state to avoid stray pan starts
                        use winit::event::{ElementState, MouseButton};
                        if button == MouseButton::Left {
                            self.mouse_left_down = state == ElementState::Pressed;
                        }
                        self.handle_mouse_input(button, state);
                        window.request_redraw();
                    }
                }

                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::CursorMoved { position, .. },
                } if window_id == window.id() => {
                    let mut route = !egui_consumed;
                    if let Some(plot_rect) = self.plot_overlay.plot_area() {
                        let ppp = self.pixels_per_point.max(0.5);
                        let mx = position.x as f32;
                        let my = position.y as f32;
                        let px_min_x = plot_rect.min.x * ppp;
                        let px_min_y = plot_rect.min.y * ppp;
                        let px_w = plot_rect.width() * ppp;
                        let px_h = plot_rect.height() * ppp;
                        if mx >= px_min_x
                            && mx <= px_min_x + px_w
                            && my >= px_min_y
                            && my <= px_min_y + px_h
                        {
                            route = true;
                        }
                    }
                    if route {
                        self.handle_mouse_move(position);
                        window.request_redraw();
                    }
                }

                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::MouseWheel { delta, .. },
                } if window_id == window.id() => {
                    let mut route = !egui_consumed;
                    if let Some(plot_rect) = self.plot_overlay.plot_area() {
                        let ppp = self.pixels_per_point.max(0.5);
                        let mx = self.mouse_position.x;
                        let my = self.mouse_position.y;
                        let px_min_x = plot_rect.min.x * ppp;
                        let px_min_y = plot_rect.min.y * ppp;
                        let px_w = plot_rect.width() * ppp;
                        let px_h = plot_rect.height() * ppp;
                        if mx >= px_min_x
                            && mx <= px_min_x + px_w
                            && my >= px_min_y
                            && my <= px_min_y + px_h
                        {
                            route = true;
                        }
                    }
                    if route {
                        self.handle_mouse_scroll(delta);
                        window.request_redraw();
                    }
                }

                winit::event::Event::AboutToWait => {
                    // Always request the first redraw; afterwards, only redraw when needed
                    if self.needs_initial_redraw || repaint {
                        self.needs_initial_redraw = false;
                        window.request_redraw();
                    }
                }

                _ => {}
            }
        })?;

        Ok(())
    }

    /// Handle window resize
    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return; // Skip invalid sizes that could cause crashes
        }

        self.config.width = width;
        self.config.height = height;

        // Recreate surface configuration with error handling
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.plot_renderer.wgpu_renderer.surface_config.format,
            width,
            height,
            present_mode: if self.config.vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // Update renderer's surface config
        self.plot_renderer.wgpu_renderer.surface_config = surface_config.clone();
        self.surface
            .configure(&self.plot_renderer.wgpu_renderer.device, &surface_config);

        // Recreate depth texture
        self.depth_texture =
            self.plot_renderer
                .wgpu_renderer
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("Depth Texture"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });

        self.depth_view = self
            .depth_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Update camera aspect ratio
        self.plot_renderer
            .camera
            .update_aspect_ratio(width as f32 / height as f32);
    }

    /// Render a frame
    fn render(&mut self, _dt: std::time::Duration) -> Result<(), wgpu::SurfaceError> {
        // Get the next frame
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Camera updates will be handled by simple interaction code

        // Create command encoder
        let mut encoder = self
            .plot_renderer
            .wgpu_renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Render egui
        let raw_input = self.egui_state.take_egui_input(&self.window);

        // Get UI data before borrowing
        let scene_stats = self.plot_renderer.scene.statistics();
        let _camera_pos = self.plot_renderer.camera.position;

        // Track the plot area for WGPU rendering
        let mut plot_area: Option<egui::Rect> = None;

        // Ensure data bounds are current before drawing overlay (keeps axes in sync with render)
        let _ = self.plot_renderer.calculate_data_bounds();

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            // Use PlotOverlay for unified UI rendering - no more duplicate sidebar code!
            let overlay_config = OverlayConfig {
                // Grid drawn under data in WGPU; overlay handles axes/labels/titles only
                show_grid: self.plot_renderer.overlay_show_grid(),
                show_axes: true,
                show_title: true,
                title: self
                    .plot_renderer
                    .overlay_title()
                    .cloned()
                    .or(Some("Plot".to_string())),
                x_label: self
                    .plot_renderer
                    .overlay_x_label()
                    .cloned()
                    .or(Some("X".to_string())),
                y_label: self
                    .plot_renderer
                    .overlay_y_label()
                    .cloned()
                    .or(Some("Y".to_string())),
                ..Default::default()
            };
            let overlay_metrics = OverlayMetrics {
                vertex_count: scene_stats.total_vertices,
                triangle_count: scene_stats.total_triangles,
                render_time_ms: 0.0, // TODO: Add timing
                fps: 60.0,           // TODO: Calculate actual FPS
            };

            let frame_info = self.plot_overlay.render(
                ctx,
                &self.plot_renderer,
                &overlay_config,
                overlay_metrics,
            );
            plot_area = frame_info.plot_area;
        });

        // Update pixels-per-point for input mapping and calculate data bounds
        let ppp_now = full_output.pixels_per_point;
        if ppp_now > 0.0 {
            // store for later mapping
            // SAFETY: field exists in window struct
            self.pixels_per_point = ppp_now;
        }
        // Calculate data bounds (kept for potential overlay/tick use)
        let _data_bounds = self.plot_renderer.data_bounds();

        // Handle toolbar actions requested by overlay
        let (save_png, save_svg, reset_view, toggle_grid_opt, toggle_legend_opt) =
            self.plot_overlay.take_toolbar_actions();
        if let Some(show) = toggle_grid_opt {
            // mutate last_figure and overlay flag
            if let Some(mut fig) = self.plot_renderer.last_figure.clone() {
                fig.grid_enabled = show;
                self.plot_renderer.set_figure(fig);
            }
        }
        if let Some(show) = toggle_legend_opt {
            if let Some(mut fig) = self.plot_renderer.last_figure.clone() {
                fig.legend_enabled = show;
                self.plot_renderer.set_figure(fig);
            }
        }
        if reset_view {
            // Refit main camera to data
            self.plot_renderer.fit_camera_to_data();
        }
        if save_png || save_svg {
            // OS Save Dialog to select path
            #[cfg(any(target_os = "macos", target_os = "windows", target_os = "linux"))]
            {
                if save_png {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("PNG Image", &["png"])
                        .set_file_name("plot.png")
                        .save_file()
                    {
                        let mut fig_for_save = self.plot_renderer.export_figure_clone();
                        let _ = std::thread::spawn(move || {
                            let rt = tokio::runtime::Builder::new_current_thread()
                                .enable_all()
                                .build();
                            if let Ok(rt) = rt {
                                let _ = rt.block_on(async move {
                                    if let Ok(exporter) =
                                        crate::export::image::ImageExporter::new().await
                                    {
                                        let _ = exporter.export_png(&mut fig_for_save, &path).await;
                                    }
                                });
                            }
                        });
                    }
                }
                if save_svg {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("SVG", &["svg"])
                        .set_file_name("plot.svg")
                        .save_file()
                    {
                        let mut fig_for_save = self.plot_renderer.export_figure_clone();
                        let exporter = crate::export::vector::VectorExporter::new();
                        let _ = exporter.export_svg(&mut fig_for_save, &path);
                    }
                }
            }
            #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
            {
                // Fallback to temp directory
                if save_png {
                    let mut fig = self.plot_renderer.export_figure_clone();
                    let tmp = std::env::temp_dir().join("runmat_export.png");
                    let _ = std::thread::spawn(move || {
                        let rt = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build();
                        if let Ok(rt) = rt {
                            let _ = rt.block_on(async move {
                                if let Ok(exporter) =
                                    crate::export::image::ImageExporter::new().await
                                {
                                    let _ = exporter.export_png(&mut fig, &tmp).await;
                                }
                            });
                        }
                    });
                }
                if save_svg {
                    let mut fig = self.plot_renderer.export_figure_clone();
                    let tmp = std::env::temp_dir().join("runmat_export.svg");
                    let exporter = crate::export::vector::VectorExporter::new();
                    let _ = exporter.export_svg(&mut fig, &tmp);
                }
            }
        }

        // Now we have the plot area, update camera and WGPU rendering accordingly
        if let Some(plot_rect) = plot_area {
            // Update camera aspect ratio to match the plot area
            let plot_width = plot_rect.width();
            let plot_height = plot_rect.height();
            if plot_width > 0.0 && plot_height > 0.0 {
                self.plot_renderer
                    .camera
                    .update_aspect_ratio(plot_width / plot_height);
            }
        }

        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(
                &self.plot_renderer.wgpu_renderer.device,
                &self.plot_renderer.wgpu_renderer.queue,
                *id,
                image_delta,
            );
        }

        self.egui_renderer.update_buffers(
            &self.plot_renderer.wgpu_renderer.device,
            &self.plot_renderer.wgpu_renderer.queue,
            &mut encoder,
            &tris,
            &egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: full_output.pixels_per_point,
            },
        );

        // First render the plot data into the scissored viewport (MSAA-friendly)
        if let Some(plot_rect) = plot_area {
            // Use egui's pixels-per-point for exact device pixel mapping
            let ppp = self.pixels_per_point.max(0.5);
            let vx = (plot_rect.min.x * ppp).round();
            let vy = (plot_rect.min.y * ppp).round();
            let vw = (plot_rect.width() * ppp).round().max(1.0);
            let vh = (plot_rect.height() * ppp).round().max(1.0);

            // Clamp to surface dimensions
            let sw = self.config.width as f32;
            let sh = self.config.height as f32;
            let cvx = vx.max(0.0);
            let cvy = vy.max(0.0);
            let mut cvw = vw;
            let mut cvh = vh;
            if cvx + cvw > sw {
                cvw = (sw - cvx).max(1.0);
            }
            if cvy + cvh > sh {
                cvh = (sh - cvy).max(1.0);
            }

            // Scissor rectangle is specified in physical pixels as u32
            let scissor = (cvx as u32, cvy as u32, cvw as u32, cvh as u32);

            // If this figure has a subplot grid > 1, split into axes rectangles and render each
            let (rows, cols) = self.plot_renderer.figure_axes_grid();
            if rows * cols > 1 {
                // compute subplot rects in UI points first, then convert to pixels
                let rects = self
                    .plot_overlay
                    .compute_subplot_rects(plot_rect, rows, cols, 8.0, 8.0);
                let mut viewports: Vec<(u32, u32, u32, u32)> = Vec::new();
                let mut hovered_axes: Option<usize> = None;
                // Detect hovered subplot for camera interaction
                let mouse_pos = self.mouse_position;
                for (i, r) in rects.iter().enumerate() {
                    let rx = (r.min.x * ppp).round();
                    let ry = (r.min.y * ppp).round();
                    let rw = (r.width() * ppp).round().max(1.0);
                    let rh = (r.height() * ppp).round().max(1.0);
                    // clamp each to surface
                    let svx = rx.max(0.0);
                    let svy = ry.max(0.0);
                    let mut svw = rw;
                    let mut svh = rh;
                    if svx + svw > sw {
                        svw = (sw - svx).max(1.0);
                    }
                    if svy + svh > sh {
                        svh = (sh - svy).max(1.0);
                    }
                    viewports.push((svx as u32, svy as u32, svw as u32, svh as u32));

                    if hovered_axes.is_none() {
                        let px_min_x = rx;
                        let px_min_y = ry;
                        if mouse_pos.x >= px_min_x
                            && mouse_pos.x <= px_min_x + rw
                            && mouse_pos.y >= px_min_y
                            && mouse_pos.y <= px_min_y + rh
                        {
                            hovered_axes = Some(i);
                        }
                    }
                }
                // Do not overwrite per-axes cameras; keep their independent state for interaction
                let _ =
                    self.plot_renderer
                        .render_axes_to_viewports(&mut encoder, &view, &viewports, 4);
            } else {
                // Single axes fallback: Render into the scissored viewport using camera path
                let cfg = crate::core::plot_renderer::PlotRenderConfig {
                    width: scissor.2,
                    height: scissor.3,
                    msaa_samples: 4,
                    ..Default::default()
                };
                let _ = self.plot_renderer.render_camera_to_viewport(
                    &mut encoder,
                    &view,
                    scissor,
                    &cfg,
                );
            }
        }

        // Then render the UI overlay on top (legend, labels, etc.)
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.egui_renderer.render(
                &mut render_pass,
                &tris,
                &egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [self.config.width, self.config.height],
                    pixels_per_point: full_output.pixels_per_point,
                },
            );
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        // Submit commands
        self.plot_renderer
            .wgpu_renderer
            .queue
            .submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Handle mouse input
    fn handle_mouse_input(
        &mut self,
        button: winit::event::MouseButton,
        state: winit::event::ElementState,
    ) {
        use winit::event::{ElementState, MouseButton};

        match (button, state) {
            (MouseButton::Left, ElementState::Pressed) => {
                // Only start panning if press occurs inside the plot area (or a subplot rect)
                self.is_mouse_over_plot = false;
                if let Some(plot_rect) = self.plot_overlay.plot_area() {
                    let mx = self.mouse_position.x;
                    let my = self.mouse_position.y;
                    let (rows, cols) = self.plot_renderer.figure_axes_grid();
                    if rows * cols > 1 {
                        // Check sub-rects
                        let rects = self
                            .plot_overlay
                            .compute_subplot_rects(plot_rect, rows, cols, 8.0, 8.0);
                        for r in rects {
                            let rx = r.min.x * self.pixels_per_point;
                            let ry = r.min.y * self.pixels_per_point;
                            let rw = r.width() * self.pixels_per_point;
                            let rh = r.height() * self.pixels_per_point;
                            if mx >= rx && mx <= rx + rw && my >= ry && my <= ry + rh {
                                self.is_mouse_over_plot = true;
                                break;
                            }
                        }
                    } else {
                        let ppp = self.pixels_per_point.max(0.5);
                        let px_min_x = plot_rect.min.x * ppp;
                        let px_min_y = plot_rect.min.y * ppp;
                        let px_w = plot_rect.width() * ppp;
                        let px_h = plot_rect.height() * ppp;
                        self.is_mouse_over_plot = mx >= px_min_x
                            && mx <= px_min_x + px_w
                            && my >= px_min_y
                            && my <= px_min_y + px_h;
                    }
                }
            }
            (MouseButton::Left, ElementState::Released) => {
                self.is_mouse_over_plot = false;
            }
            _ => {}
        }
    }

    /// Handle mouse movement
    fn handle_mouse_move(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        let new_position = glam::Vec2::new(position.x as f32, position.y as f32);
        let delta = if self.mouse_left_down {
            new_position - self.mouse_position
        } else {
            glam::Vec2::ZERO
        };
        self.mouse_position = new_position;

        // Pan when left mouse button is held down: shift orthographic bounds in world units
        if self.is_mouse_over_plot && delta.length() > 0.0 {
            if let Some(plot_rect) = self.plot_overlay.plot_area() {
                let (rows, cols) = self.plot_renderer.figure_axes_grid();
                if rows * cols > 1 {
                    // Determine hovered subplot and pan its camera
                    let rects = self
                        .plot_overlay
                        .compute_subplot_rects(plot_rect, rows, cols, 8.0, 8.0);
                    for (i, r) in rects.iter().enumerate() {
                        let rx = r.min.x * self.pixels_per_point;
                        let ry = r.min.y * self.pixels_per_point;
                        let rw = r.width() * self.pixels_per_point;
                        let rh = r.height() * self.pixels_per_point;
                        if self.mouse_position.x >= rx
                            && self.mouse_position.x <= rx + rw
                            && self.mouse_position.y >= ry
                            && self.mouse_position.y <= ry + rh
                        {
                            if let Some(cam) = self.plot_renderer.axes_camera_mut(i) {
                                if let crate::core::camera::ProjectionType::Orthographic {
                                    left,
                                    right,
                                    bottom,
                                    top,
                                    ..
                                } = cam.projection
                                {
                                    let pw = rw.max(1.0);
                                    let ph = rh.max(1.0);
                                    let world_w = right - left;
                                    let world_h = top - bottom;
                                    let dx_world = (delta.x / pw) * world_w;
                                    let dy_world = (delta.y / ph) * world_h;
                                    cam.projection =
                                        crate::core::camera::ProjectionType::Orthographic {
                                            left: left - dx_world,
                                            right: right - dx_world,
                                            bottom: bottom + dy_world,
                                            top: top + dy_world,
                                            near: -1.0,
                                            far: 1.0,
                                        };
                                    cam.mark_dirty();
                                }
                            }
                            break;
                        }
                    }
                } else if let crate::core::camera::ProjectionType::Orthographic {
                    ref mut left,
                    ref mut right,
                    ref mut bottom,
                    ref mut top,
                    ..
                } = self.plot_renderer.camera.projection
                {
                    let pw = (plot_rect.width() * self.pixels_per_point).max(1.0);
                    let ph = (plot_rect.height() * self.pixels_per_point).max(1.0);
                    let world_w = *right - *left;
                    let world_h = *top - *bottom;
                    let dx_world = (delta.x / pw) * world_w;
                    let dy_world = (delta.y / ph) * world_h;
                    *left -= dx_world;
                    *right -= dx_world;
                    *bottom += dy_world;
                    *top += dy_world;
                    self.plot_renderer.camera.mark_dirty();
                }
            }
        }
    }

    /// Handle mouse scroll
    fn handle_mouse_scroll(&mut self, delta: winit::event::MouseScrollDelta) {
        let scroll_delta = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => y,
            winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
        };

        // Zoom in/out by scaling the orthographic projection. Anchor zoom at cursor when inside plot area.
        if let Some(plot_rect) = self.plot_overlay.plot_area() {
            let (rows, cols) = self.plot_renderer.figure_axes_grid();
            if rows * cols > 1 {
                let rects = self
                    .plot_overlay
                    .compute_subplot_rects(plot_rect, rows, cols, 8.0, 8.0);
                for (i, r) in rects.iter().enumerate() {
                    let rx = r.min.x * self.pixels_per_point;
                    let ry = r.min.y * self.pixels_per_point;
                    let rw = r.width() * self.pixels_per_point;
                    let rh = r.height() * self.pixels_per_point;
                    let mx = self.mouse_position.x;
                    let my = self.mouse_position.y;
                    if mx >= rx && mx <= rx + rw && my >= ry && my <= ry + rh {
                        if let Some(cam) = self.plot_renderer.axes_camera_mut(i) {
                            if let crate::core::camera::ProjectionType::Orthographic {
                                left,
                                right,
                                bottom,
                                top,
                                ..
                            } = cam.projection
                            {
                                let factor = (1.0 - scroll_delta * 0.1).clamp(0.2, 5.0);
                                let tx = (mx - rx) / rw;
                                let ty = (my - ry) / rh;
                                let w = right - left;
                                let h = top - bottom;
                                let pivot_x = left + tx * w;
                                let pivot_y = top - ty * h;
                                let new_left = pivot_x - (pivot_x - left) * factor;
                                let new_right = pivot_x + (right - pivot_x) * factor;
                                let new_bottom = pivot_y - (pivot_y - bottom) * factor;
                                let new_top = pivot_y + (top - pivot_y) * factor;
                                cam.projection =
                                    crate::core::camera::ProjectionType::Orthographic {
                                        left: new_left,
                                        right: new_right,
                                        bottom: new_bottom,
                                        top: new_top,
                                        near: -1.0,
                                        far: 1.0,
                                    };
                                cam.mark_dirty();
                            }
                        }
                        break;
                    }
                }
            } else if let crate::core::camera::ProjectionType::Orthographic {
                ref mut left,
                ref mut right,
                ref mut bottom,
                ref mut top,
                ..
            } = self.plot_renderer.camera.projection
            {
                let factor = (1.0 - scroll_delta * 0.1).clamp(0.2, 5.0);
                let px_min_x = plot_rect.min.x * self.pixels_per_point;
                let px_min_y = plot_rect.min.y * self.pixels_per_point;
                let px_w = plot_rect.width() * self.pixels_per_point;
                let px_h = plot_rect.height() * self.pixels_per_point;
                let mx = self.mouse_position.x;
                let my = self.mouse_position.y;
                let mut pivot_x = (*left + *right) * 0.5;
                let mut pivot_y = (*bottom + *top) * 0.5;
                if mx >= px_min_x
                    && mx <= px_min_x + px_w
                    && my >= px_min_y
                    && my <= px_min_y + px_h
                {
                    let tx = (mx - px_min_x) / px_w;
                    let ty = (my - px_min_y) / px_h;
                    let w = *right - *left;
                    let h = *top - *bottom;
                    pivot_x = *left + tx * w;
                    pivot_y = *top - ty * h;
                }
                let new_left = pivot_x - (pivot_x - *left) * factor;
                let new_right = pivot_x + (*right - pivot_x) * factor;
                let new_bottom = pivot_y - (pivot_y - *bottom) * factor;
                let new_top = pivot_y + (*top - pivot_y) * factor;
                *left = new_left;
                *right = new_right;
                *bottom = new_bottom;
                *top = new_top;
                self.plot_renderer.camera.mark_dirty();
            }
        }
    }
}
