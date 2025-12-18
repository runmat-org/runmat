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
use std::sync::Arc;
#[cfg(feature = "gui")]
#[allow(unused_imports)]  // Conditional compilation may not use all imports
use winit::{dpi::PhysicalSize, event::Event, event_loop::{EventLoop, EventLoopBuilder}, window::WindowBuilder};
#[cfg(feature = "gui")]
impl<'window> PlotWindow<'window> {
    /// Create a new interactive plot window
    pub async fn new(config: WindowConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Create EventLoop with platform-specific support for background threads
        #[cfg(target_os = "windows")]
        let event_loop = {
            use winit::platform::windows::EventLoopBuilderExtWindows;
            EventLoopBuilder::new()
                .with_any_thread(true)
                .build()
                .map_err(|e| format!("Failed to create EventLoop: {e}"))?
        };
        
        #[cfg(not(target_os = "windows"))]
        let event_loop = EventLoopBuilder::new()
            .build()
            .map_err(|e| format!("Failed to create EventLoop: {e}"))?;
        let window = WindowBuilder::new()
            .with_title(&config.title)
            .with_inner_size(PhysicalSize::new(config.width, config.height))
            .with_resizable(config.resizable)
            .with_maximized(config.maximized)
            .build(&event_loop)?;
        let window = Arc::new(window);

        // Create WGPU instance and surface
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        // Request adapter and device
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

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
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
            material: crate::core::Material::default(),
            draw_calls: vec![crate::core::DrawCall {
                vertex_offset: 0,
                vertex_count: (x_data.len() - 1) * 2, // Each line segment has 2 vertices
                index_offset: None,
                index_count: None,
                instance_count: 1,
            }],
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

    /// Run the interactive plot window event loop
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let event_loop = self
            .event_loop
            .take()
            .ok_or("Event loop already consumed")?;
        let window = self.window.clone();
        let mut last_render_time = std::time::Instant::now();

        event_loop.run(move |event, target| {
            target.set_control_flow(winit::event_loop::ControlFlow::Poll);

            // Handle egui events
            let mut repaint = false;
            if let Event::WindowEvent { ref event, .. } = event {
                let response = self.egui_state.on_window_event(&window, event);
                repaint = response.repaint;
            }
            if repaint {
                window.request_redraw();
            }

            match event {
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
                    let now = std::time::Instant::now();
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

                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::MouseInput { button, state, .. },
                } if window_id == window.id() => {
                    self.handle_mouse_input(button, state);
                }

                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::CursorMoved { position, .. },
                } if window_id == window.id() => {
                    self.handle_mouse_move(position);
                }

                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::MouseWheel { delta, .. },
                } if window_id == window.id() => {
                    self.handle_mouse_scroll(delta);
                }

                winit::event::Event::AboutToWait => {
                    // Request redraw only when interaction occurs - prevents infinite loop
                    if repaint {
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

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            // Use PlotOverlay for unified UI rendering - no more duplicate sidebar code!
            let overlay_config = OverlayConfig {
                show_grid: true,
                show_axes: true,
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

        // Calculate data bounds for viewport transformation
        let data_bounds = self.plot_renderer.calculate_data_bounds();

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
                pixels_per_point: self.window.scale_factor() as f32,
            },
        );

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
                    pixels_per_point: self.window.scale_factor() as f32,
                },
            );

            // End the egui render pass to avoid borrowing conflicts
            drop(render_pass);

            // Render WGPU plot data on top of egui content using the unified renderer
            if let Some(plot_rect) = plot_area {
                let scale_factor = self.window.scale_factor() as f32;

                let viewport = (
                    plot_rect.min.x * scale_factor,
                    plot_rect.min.y * scale_factor,
                    plot_rect.width() * scale_factor,
                    plot_rect.height() * scale_factor,
                );

                // Execute optimized direct viewport rendering
                if let Some(bounds) = data_bounds {
                    let _ = self.plot_renderer.render_direct_to_viewport(
                        &mut encoder,
                        &view,
                        viewport,
                        bounds,
                        false, // Don't clear background, preserve egui content
                        None,  // No custom background color
                    );
                }
            }
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
                self.is_mouse_over_plot = true; // For panning
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
        let delta = new_position - self.mouse_position;
        self.mouse_position = new_position;

        // Pan when left mouse button is held down
        if self.is_mouse_over_plot && delta.length() > 0.0 {
            self.plot_renderer.camera.pan(-delta * 0.01); // Negative for natural feel
        }
    }

    /// Handle mouse scroll
    fn handle_mouse_scroll(&mut self, delta: winit::event::MouseScrollDelta) {
        let scroll_delta = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => y,
            winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
        };

        // Zoom in/out by scaling the orthographic projection
        if let crate::core::camera::ProjectionType::Orthographic {
            ref mut left,
            ref mut right,
            ref mut bottom,
            ref mut top,
            ..
        } = self.plot_renderer.camera.projection
        {
            let zoom_factor = 1.0 + scroll_delta * 0.1;
            let center_x = (*left + *right) / 2.0;
            let center_y = (*bottom + *top) / 2.0;
            let width = (*right - *left) / zoom_factor;
            let height = (*top - *bottom) / zoom_factor;

            *left = center_x - width / 2.0;
            *right = center_x + width / 2.0;
            *bottom = center_y - height / 2.0;
            *top = center_y + height / 2.0;
        }
    }
}
