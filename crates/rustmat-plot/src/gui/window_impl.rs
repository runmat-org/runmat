//! Implementation methods for the GUI plot window

#[cfg(feature = "gui")]
use super::{PlotWindow, WindowConfig};
#[cfg(feature = "gui")]
use std::sync::Arc;
#[cfg(feature = "gui")]
use winit::{
    event::{Event, WindowEvent, DeviceEvent, ElementState, MouseButton, MouseScrollDelta},
    event_loop::{EventLoop, ControlFlow},
    window::{Window, WindowBuilder},
    dpi::PhysicalSize,
};
#[cfg(feature = "gui")]
use egui_winit::State as EguiState;
#[cfg(feature = "gui")]
use crate::core::{WgpuRenderer, Camera, CameraController, Scene, PipelineType};
#[cfg(feature = "gui")]
use glam::{Vec2, Vec3, Vec4, Mat4};

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
        let camera = Camera::new();
        let camera_controller = CameraController::new();
        let scene = Scene::new();
        
        // Setup egui
        let egui_ctx = egui::Context::default();
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
            Some(wgpu::TextureFormat::Depth32Float),
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
            camera_controller,
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
}