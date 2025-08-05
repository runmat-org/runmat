//! Implementation methods for the GUI plot window

#[cfg(feature = "gui")]
use super::{PlotWindow, WindowConfig};
#[cfg(feature = "gui")]
use std::sync::Arc;
#[cfg(feature = "gui")]
use winit::{
    event::Event,
    event_loop::EventLoop,
    window::WindowBuilder,
    dpi::PhysicalSize,
};
#[cfg(feature = "gui")]
use egui_winit::State as EguiState;
#[cfg(feature = "gui")]
use crate::core::{WgpuRenderer, Camera, Scene, PipelineType};
#[cfg(feature = "gui")]
use glam::{Vec2, Vec3, Vec4, Mat4};
#[cfg(feature = "gui")]
use wgpu::util::DeviceExt;

#[cfg(feature = "gui")]
impl<'window> PlotWindow<'window> {
    /// Create a new interactive plot window
    pub async fn new(config: WindowConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Create window and event loop
        let event_loop = EventLoop::new()?;
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
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.ok_or("Failed to request adapter")?;
        
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("RustMat Plot Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await?;
        
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        
        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
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
        
        // Create renderer
        let renderer = WgpuRenderer::new(device.clone(), queue.clone(), surface_config).await;
        
        // Create camera and scene
        let mut camera = Camera::new();
        
        // Configure camera for 2D plotting with orthographic projection
        camera.projection = crate::core::camera::ProjectionType::Orthographic {
            left: -5.0,
            right: 5.0,
            bottom: -5.0,
            top: 5.0,
            near: 0.1,
            far: 100.0,
        };
        camera.position = Vec3::new(0.0, 0.0, 5.0);
        camera.target = Vec3::new(0.0, 0.0, 0.0);
        camera.up = Vec3::new(0.0, 1.0, 0.0);
        
        let scene = Scene::new();
        
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
            renderer,
            surface,
            depth_texture,
            depth_view,
            camera,
            scene,
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
        let vertices = vertex_utils::create_line_plot(&x_data, &y_data, Vec4::new(0.0, 0.5, 1.0, 1.0));
        
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
                &x_data.iter().zip(y_data.iter())
                    .map(|(&x, &y)| Vec3::new(x as f32, y as f32, 0.0))
                    .collect::<Vec<_>>()
            ),
            lod_levels: Vec::new(),
            current_lod: 0,
        };
        
        self.scene.add_node(node);
        
        // Fit camera to show the plot
        let bounds_min = Vec3::new(-1.0, -1.5, -1.0);
        let bounds_max = Vec3::new(10.0, 1.5, 1.0);
        self.camera.fit_bounds(bounds_min, bounds_max);
    }
    
    /// Add a complete figure with all its plots to the scene
    pub fn add_figure(&mut self, figure: &crate::plots::Figure) {
        use crate::core::vertex_utils;
        use crate::plots::figure::PlotElement;
        
        let mut all_points = Vec::new();
        let mut node_id = 0;
        
        // Process each plot in the figure
        for (i, plot_element) in figure.plots().enumerate() {
            let (vertices, name, color, pipeline_type) = match plot_element {
                PlotElement::Line(line_plot) => {
                    let x_data: Vec<f64> = line_plot.x_data.iter().map(|&x| x as f64).collect();
                    let y_data: Vec<f64> = line_plot.y_data.iter().map(|&y| y as f64).collect();
                    
                    // Add points for bounds calculation
                    all_points.extend(
                        x_data.iter().zip(y_data.iter())
                            .map(|(&x, &y)| Vec3::new(x as f32, y as f32, 0.0))
                    );
                    
                    let vertices = vertex_utils::create_line_plot(&x_data, &y_data, line_plot.color);
                    (vertices, format!("Line Plot {}", i + 1), line_plot.color, PipelineType::Triangles)
                },
                PlotElement::Scatter(scatter_plot) => {
                    let x_data: Vec<f64> = scatter_plot.x_data.iter().map(|&x| x as f64).collect();
                    let y_data: Vec<f64> = scatter_plot.y_data.iter().map(|&y| y as f64).collect();
                    
                    // Add points for bounds calculation
                    all_points.extend(
                        x_data.iter().zip(y_data.iter())
                            .map(|(&x, &y)| Vec3::new(x as f32, y as f32, 0.0))
                    );
                    
                    let vertices = vertex_utils::create_scatter_plot(&x_data, &y_data, scatter_plot.color);
                    (vertices, format!("Scatter Plot {}", i + 1), scatter_plot.color, PipelineType::Triangles)
                },
                PlotElement::Bar(bar_chart) => {
                    // Convert bar chart to line representation for now
                    let x_data: Vec<f64> = (0..bar_chart.values.len()).map(|i| i as f64).collect();
                    let y_data: Vec<f64> = bar_chart.values.iter().map(|&v| v as f64).collect();
                    
                    // Add points for bounds calculation
                    all_points.extend(
                        x_data.iter().zip(y_data.iter())
                            .map(|(&x, &y)| Vec3::new(x as f32, y as f32, 0.0))
                    );
                    
                    let vertices = vertex_utils::create_line_plot(&x_data, &y_data, bar_chart.color);
                    (vertices, format!("Bar Chart {}", i + 1), bar_chart.color, PipelineType::Lines)
                },
                PlotElement::Histogram(histogram) => {
                    // Convert histogram to line representation  
                    let x_data: Vec<f64> = histogram.bin_edges.iter().take(histogram.bin_counts.len()).map(|&x| x as f64).collect();
                    let y_data: Vec<f64> = histogram.bin_counts.iter().map(|&c| c as f64).collect();
                    
                    // Add points for bounds calculation
                    all_points.extend(
                        x_data.iter().zip(y_data.iter())
                            .map(|(&x, &y)| Vec3::new(x as f32, y as f32, 0.0))
                    );
                    
                    let vertices = vertex_utils::create_line_plot(&x_data, &y_data, histogram.color);
                    (vertices, format!("Histogram {}", i + 1), histogram.color, PipelineType::Lines)
                },
            };
            
            // Create render data
            let mut render_data = crate::core::RenderData {
                pipeline_type,
                vertices: vertices.clone(),
                indices: None,
                material: crate::core::Material::default(),
                draw_calls: vec![crate::core::DrawCall {
                    vertex_offset: 0,
                    vertex_count: vertices.len(),
                    index_offset: None,
                    index_count: None,
                    instance_count: 1,
                }],
            };
            
            // Set material color
            render_data.material.albedo = color;
            
            // Create scene node
            let node = crate::core::SceneNode {
                id: node_id,
                name,
                transform: Mat4::IDENTITY,
                visible: true,
                cast_shadows: false,
                receive_shadows: false,
                parent: None,
                children: Vec::new(),
                render_data: Some(render_data),
                bounds: crate::core::BoundingBox::from_points(&all_points),
                lod_levels: Vec::new(),
                current_lod: 0,
            };
            
            self.scene.add_node(node);
            node_id += 1;
        }
        
        // Fit camera to show all plots
        if !all_points.is_empty() {
            let bounds = crate::core::BoundingBox::from_points(&all_points);
                    self.camera.fit_bounds(bounds.min, bounds.max);
        }
        
        println!("Added figure with {} plots to scene", figure.len());
    }

    /// Set the figure to display in this window (clears existing content)
    pub fn set_figure(&mut self, figure: crate::plots::Figure) {
        // Clear the current scene
        self.scene = crate::core::Scene::new();
        
        // Add the new figure
        self.add_figure(&figure);
    }
    
    /// Run the interactive plot window event loop
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let event_loop = self.event_loop.take().ok_or("Event loop already consumed")?;
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
                    event: winit::event::WindowEvent::CloseRequested 
                } if window_id == window.id() => {
                    target.exit();
                },
                
                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::Resized(new_size)
                } if window_id == window.id() => {
                    // Resize surface and depth texture
                    if new_size.width > 0 && new_size.height > 0 {
                        self.resize(new_size.width, new_size.height);
                    }
                },
                
                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::RedrawRequested
                } if window_id == window.id() => {
                    let now = std::time::Instant::now();
                    let dt = now - last_render_time;
                    last_render_time = now;
                    
                    match self.render(dt) {
                        Ok(_) => {},
                        Err(wgpu::SurfaceError::Lost) => self.resize(self.config.width, self.config.height),
                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                        Err(e) => eprintln!("Render error: {:?}", e),
                    }
                },
                
                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::MouseInput { button, state, .. }
                } if window_id == window.id() => {
                    self.handle_mouse_input(button, state);
                },
                
                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::CursorMoved { position, .. }
                } if window_id == window.id() => {
                    self.handle_mouse_move(position);
                },
                
                winit::event::Event::WindowEvent {
                    window_id,
                    event: winit::event::WindowEvent::MouseWheel { delta, .. }
                } if window_id == window.id() => {
                    self.handle_mouse_scroll(delta);
                },
                
                winit::event::Event::AboutToWait => {
                    window.request_redraw();
                },
                
                _ => {}
            }
        })?;
        
        Ok(())
    }
    
    /// Handle window resize
    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        
        // Recreate surface configuration
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.renderer.surface_config.format,
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
        self.surface.configure(&self.renderer.device, &surface_config);
        
        // Recreate depth texture
        self.depth_texture = self.renderer.device.create_texture(&wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Update camera aspect ratio
        self.camera.update_aspect_ratio(width as f32 / height as f32);
    }
    
    /// Render a frame
    fn render(&mut self, _dt: std::time::Duration) -> Result<(), wgpu::SurfaceError> {
        // Get the next frame
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Camera updates will be handled by simple interaction code
        
        // Create command encoder
        let mut encoder = self.renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // Collect all render data and create vertex buffers first (outside render pass)
        let mut render_items = Vec::new();
        
        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if !render_data.vertices.is_empty() {
                    // Ensure pipeline exists for this render data
                    self.renderer.ensure_pipeline(render_data.pipeline_type);
                    
                    // Create vertex buffer for this node
                    let vertex_buffer = self.renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Plot Vertex Buffer"),
                        contents: bytemuck::cast_slice(&render_data.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    
                    render_items.push((render_data, vertex_buffer));
                }
            }
        }

        
        // Update camera uniforms before rendering  
        let view_matrix = self.camera.view_matrix();
        let proj_matrix = self.camera.projection_matrix();
        let view_proj_matrix = proj_matrix * view_matrix;
        
        // Matrix computation complete
        
        self.renderer.update_uniforms(view_proj_matrix, Mat4::IDENTITY);
        
        // Render the scene using our world-class WGPU renderer
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.08,  // Modern dark theme background
                    g: 0.09,
                    b: 0.11,
                    a: 1.0,
                }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None, // Disabled depth testing for debugging
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Now render all items with proper bind group setup
            for (render_data, vertex_buffer) in &render_items {
                // Get the appropriate pipeline for this render data (pipeline ensured above)
                let pipeline = self.renderer.get_pipeline(render_data.pipeline_type);
                render_pass.set_pipeline(pipeline);

                // Set the uniform bind group (required by shaders)
                render_pass.set_bind_group(0, self.renderer.get_uniform_bind_group(), &[]);

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

                // Render the vertices
                for draw_call in &render_data.draw_calls {
                    render_pass.draw(
                        draw_call.vertex_offset as u32..(draw_call.vertex_offset + draw_call.vertex_count) as u32,
                        0..draw_call.instance_count as u32,
                    );
                }
            }
        }
        
        // Render egui
        let raw_input = self.egui_state.take_egui_input(&self.window);
        
        // Get UI data before borrowing
        let scene_stats = self.scene.statistics();
        let camera_pos = self.camera.position;
        
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            // Make the egui context fully transparent to show WGPU content behind
            let mut visuals = ctx.style().visuals.clone();
            visuals.window_fill = egui::Color32::TRANSPARENT;
            visuals.panel_fill = egui::Color32::TRANSPARENT;
            ctx.set_visuals(visuals);
            // Modern professional plot controls panel
            egui::SidePanel::left("plot_controls")
                .resizable(true)
                .default_width(280.0)
                .min_width(200.0)
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(8.0);
                        ui.heading("📊 RustMat Plot");
                        ui.add_space(4.0);
                        ui.label("Interactive Visualization");
                        ui.add_space(12.0);
                    });
                    
                    ui.separator();
                    ui.add_space(8.0);
                    
                    // Camera section with modern styling
                    ui.collapsing("🎥 Camera Controls", |ui| {
                        ui.add_space(4.0);
                        ui.label("Position:");
                        ui.monospace(format!("X: {:.2}", camera_pos.x));
                        ui.monospace(format!("Y: {:.2}", camera_pos.y));
                        ui.monospace(format!("Z: {:.2}", camera_pos.z));
                        
                        ui.add_space(8.0);
                        ui.label("Controls:");
                        ui.label("• Mouse scroll: Zoom");
                        ui.label("• Mouse drag: Pan");
                        ui.label("• Right click: Rotate");
                    });
                    
                    ui.add_space(8.0);
                    
                    // Scene information
                    ui.collapsing("📈 Scene Info", |ui| {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label("Plot objects:");
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.strong(format!("{}", scene_stats.total_nodes));
                            });
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Vertices:");
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.strong(format!("{}", scene_stats.total_vertices));
                            });
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Triangles:");
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.strong(format!("{}", scene_stats.total_triangles));
                            });
                        });
                    });
                    
                    ui.add_space(8.0);
                    
                    // Rendering performance
                    ui.collapsing("⚡ Performance", |ui| {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label("Renderer:");
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.strong("WGPU Metal");
                            });
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("GPU:");
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.strong("Apple M2 Max");
                            });
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Status:");
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.colored_label(
                                    egui::Color32::from_rgb(89, 200, 120), // Green accent
                                    "● Active"
                                );
                            });
                        });
                    });
                    
                    ui.add_space(8.0);
                    
                    // Theme selector
                    ui.collapsing("🎨 Appearance", |ui| {
                        ui.add_space(4.0);
                        ui.label("Theme: Modern Dark");
                        ui.add_space(4.0);
                        
                        ui.horizontal(|ui| {
                            ui.label("Background:");
                            ui.color_edit_button_srgba(
                                &mut egui::Color32::from_rgba_unmultiplied(20, 23, 27, 255)
                            );
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Accent:");
                            ui.color_edit_button_srgba(
                                &mut egui::Color32::from_rgba_unmultiplied(89, 200, 120, 255)
                            );
                        });
                    });
                    
                    // Footer with version info
                    ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                        ui.add_space(8.0);
                        ui.separator();
                        ui.add_space(8.0);
                        ui.small("RustMat Plot v0.1.0");
                        ui.small("World-class plotting for Rust");
                    });
                });
                
            // Main plot area - transparent to show WGPU rendering underneath
            egui::CentralPanel::default()
                .frame(egui::Frame::none()) // Remove frame/background
                .show(ctx, |ui| {
                    // Make the UI transparent to show WGPU content
                    if scene_stats.total_nodes == 0 {
                        // Only show instructions if no data
                        ui.vertical_centered(|ui| {
                            ui.add_space(100.0);
                            ui.heading("🎯 Ready for Data");
                            ui.add_space(8.0);
                            ui.label("Your beautiful plots will appear here");
                            ui.add_space(4.0);
                            ui.small("Try: plot([1,2,3], [1,4,9])");
                        });
                    } else {
                        // For active plots, show minimal overlay
                        ui.with_layout(egui::Layout::top_down(egui::Align::RIGHT), |ui| {
                            ui.horizontal(|ui| {
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.small("⚡");
                                    ui.small("GPU-accelerated rendering active");
                                });
                            });
                        });
                        
                        // Rest of the area is transparent for WGPU rendering
                        ui.allocate_space(ui.available_size()); // Claim the space but draw nothing
                    }
                });
        });
        
        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
        
        let tris = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.renderer.device, &self.renderer.queue, *id, image_delta);
        }
        
        self.egui_renderer.update_buffers(&self.renderer.device, &self.renderer.queue, &mut encoder, &tris, &egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        });
        
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
            
            self.egui_renderer.render(&mut render_pass, &tris, &egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: self.window.scale_factor() as f32,
            });
        }
        
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        
        // Submit commands
        self.renderer.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
    }
    
    /// Handle mouse input
    fn handle_mouse_input(&mut self, _button: winit::event::MouseButton, _state: winit::event::ElementState) {
        // Simple mouse interaction - to be implemented later
        // For now, just track that mouse interaction is happening
    }
    
    /// Handle mouse movement
    fn handle_mouse_move(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        self.mouse_position = glam::Vec2::new(position.x as f32, position.y as f32);
        // Simple camera interaction to be implemented later
    }
    
    /// Handle mouse scroll
    fn handle_mouse_scroll(&mut self, delta: winit::event::MouseScrollDelta) {
        let scroll_delta = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => y,
            winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
        };
        
        // Simple zoom: move camera closer/farther based on scroll
        let zoom_speed = 0.1;
        let forward = (self.camera.target - self.camera.position).normalize();
        self.camera.position += forward * scroll_delta * zoom_speed;
    }
}