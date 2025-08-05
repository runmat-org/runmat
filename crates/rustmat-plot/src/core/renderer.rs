//! WGPU-based rendering backend for high-performance plotting
//! 
//! This module provides GPU-accelerated rendering using WGPU, supporting
//! both desktop and web targets for maximum compatibility.

use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

/// Vertex data for rendering points, lines, and triangles
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    pub fn new(position: Vec3, color: Vec4) -> Self {
        Self {
            position: position.to_array(),
            color: color.to_array(),
            normal: [0.0, 0.0, 1.0], // Default normal
            tex_coords: [0.0, 0.0],  // Default UV
        }
    }
    
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 7]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Texture coordinates
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 10]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Uniform buffer for camera and transformation matrices
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 3], // Use 4x3 for proper alignment instead of 3x3
}

/// Optimized uniform buffer for direct coordinate transformation rendering
/// Enables precise viewport-constrained data visualization
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DirectUniforms {
    pub data_min: [f32; 2],      // (x_min, y_min) in data space
    pub data_max: [f32; 2],      // (x_max, y_max) in data space
    pub viewport_min: [f32; 2],  // NDC coordinates of viewport bottom-left
    pub viewport_max: [f32; 2],  // NDC coordinates of viewport top-right
}

impl Uniforms {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            model: Mat4::IDENTITY.to_cols_array_2d(),
            normal_matrix: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        }
    }
    
    pub fn update_view_proj(&mut self, view_proj: Mat4) {
        self.view_proj = view_proj.to_cols_array_2d();
    }
    
    pub fn update_model(&mut self, model: Mat4) {
        self.model = model.to_cols_array_2d();
        // Update normal matrix (upper 3x3 of inverse transpose) with proper alignment
        let normal_mat = model.inverse().transpose();
        self.normal_matrix = [
            [normal_mat.x_axis.x, normal_mat.x_axis.y, normal_mat.x_axis.z, 0.0],
            [normal_mat.y_axis.x, normal_mat.y_axis.y, normal_mat.y_axis.z, 0.0],
            [normal_mat.z_axis.x, normal_mat.z_axis.y, normal_mat.z_axis.z, 0.0],
        ];
    }
}

impl DirectUniforms {
    pub fn new(
        data_min: [f32; 2],
        data_max: [f32; 2], 
        viewport_min: [f32; 2],
        viewport_max: [f32; 2],
    ) -> Self {
        Self {
            data_min,
            data_max,
            viewport_min,
            viewport_max,
        }
    }
}

/// Rendering pipeline types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineType {
    Points,
    Lines,
    Triangles,
    PointCloud,
}

/// High-performance WGPU renderer for interactive plotting
pub struct WgpuRenderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface_config: wgpu::SurfaceConfiguration,
    
    // Rendering pipelines (traditional camera-based)
    point_pipeline: Option<wgpu::RenderPipeline>,
    line_pipeline: Option<wgpu::RenderPipeline>,
    triangle_pipeline: Option<wgpu::RenderPipeline>,
    
    // Direct rendering pipelines (optimized coordinate transformation)
    pub direct_line_pipeline: Option<wgpu::RenderPipeline>,
    
    // Uniform resources (traditional)
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    
    // Direct uniform resources (optimized coordinate transformation)
    direct_uniform_buffer: wgpu::Buffer,
    pub direct_uniform_bind_group: wgpu::BindGroup,
    direct_uniform_bind_group_layout: wgpu::BindGroupLayout,
    
    // Current uniforms
    uniforms: Uniforms,
    direct_uniforms: DirectUniforms,
}

impl WgpuRenderer {
    /// Create a new WGPU renderer
    pub async fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_config: wgpu::SurfaceConfiguration,
    ) -> Self {
        // Create uniform buffer
        let uniforms = Uniforms::new();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create bind group layout for uniforms
        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("uniform_bind_group_layout"),
        });
        
        // Create bind group
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });
        
        // Create direct rendering uniform buffer
        let direct_uniforms = DirectUniforms::new(
            [0.0, 0.0], // data_min
            [1.0, 1.0], // data_max  
            [-1.0, -1.0], // viewport_min (full NDC)
            [1.0, 1.0],   // viewport_max (full NDC)
        );
        let direct_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Direct Uniform Buffer"),
            contents: bytemuck::cast_slice(&[direct_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create direct bind group layout for uniforms
        let direct_uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("direct_uniform_bind_group_layout"),
        });
        
        // Create direct bind group
        let direct_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &direct_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: direct_uniform_buffer.as_entire_binding(),
            }],
            label: Some("direct_uniform_bind_group"),
        });
        
        Self {
            device,
            queue,
            surface_config,
            point_pipeline: None,
            line_pipeline: None,
            triangle_pipeline: None,
            direct_line_pipeline: None,
            uniform_buffer,
            uniform_bind_group,
            uniform_bind_group_layout,
            direct_uniform_buffer,
            direct_uniform_bind_group,
            direct_uniform_bind_group_layout,
            uniforms,
            direct_uniforms,
        }
    }
    
    /// Create a vertex buffer from vertex data
    pub fn create_vertex_buffer(&self, vertices: &[Vertex]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }
    
    /// Create an index buffer from index data
    pub fn create_index_buffer(&self, indices: &[u32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        })
    }
    
    /// Update uniform buffer with new matrices
    pub fn update_uniforms(&mut self, view_proj: Mat4, model: Mat4) {
        self.uniforms.update_view_proj(view_proj);
        self.uniforms.update_model(model);
        
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }
    
    /// Get the uniform bind group for rendering
    pub fn get_uniform_bind_group(&self) -> &wgpu::BindGroup {
        &self.uniform_bind_group
    }
    
    /// Ensure pipeline exists for the specified type
    pub fn ensure_pipeline(&mut self, pipeline_type: PipelineType) {
        match pipeline_type {
            PipelineType::Points => {
                if self.point_pipeline.is_none() {
                    self.point_pipeline = Some(self.create_point_pipeline());
                }
            }
            PipelineType::Lines => {
                if self.line_pipeline.is_none() {
                    self.line_pipeline = Some(self.create_line_pipeline());
                }
            }
            PipelineType::Triangles => {
                if self.triangle_pipeline.is_none() {
                    self.triangle_pipeline = Some(self.create_triangle_pipeline());
                }
            }
            PipelineType::PointCloud => {
                // For now, use points pipeline - will optimize later
                self.ensure_pipeline(PipelineType::Points);
            }
        }
    }
    
    /// Get a pipeline reference (pipeline must already exist)
    pub fn get_pipeline(&self, pipeline_type: PipelineType) -> &wgpu::RenderPipeline {
        match pipeline_type {
            PipelineType::Points => self.point_pipeline.as_ref().unwrap(),
            PipelineType::Lines => self.line_pipeline.as_ref().unwrap(),
            PipelineType::Triangles => self.triangle_pipeline.as_ref().unwrap(),
            PipelineType::PointCloud => self.get_pipeline(PipelineType::Points),
        }
    }
    
    /// Create point rendering pipeline
    fn create_point_pipeline(&self) -> wgpu::RenderPipeline {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Point Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/vertex/point.wgsl").into()),
        });
        
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Point Pipeline Layout"),
            bind_group_layouts: &[&self.uniform_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Point Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // Disable depth testing for 2D point plots
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        })
    }
    
    /// Create line rendering pipeline
    fn create_line_pipeline(&self) -> wgpu::RenderPipeline {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Line Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/vertex/line.wgsl").into()),
        });
        
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Line Pipeline Layout"),
            bind_group_layouts: &[&self.uniform_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // Disable depth testing for 2D line plots
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        })
    }
    
    /// Create optimized direct rendering pipeline for precise viewport mapping
    fn create_direct_line_pipeline(&self) -> wgpu::RenderPipeline {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Direct Line Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/vertex/line_direct.wgsl").into()),
        });
        
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Direct Line Pipeline Layout"),
            bind_group_layouts: &[&self.direct_uniform_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Direct Line Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // Disable depth testing for 2D line plots
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        })
    }
    
    /// Create triangle rendering pipeline
    fn create_triangle_pipeline(&self) -> wgpu::RenderPipeline {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Triangle Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/vertex/triangle.wgsl").into()),
        });
        
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Triangle Pipeline Layout"),
            bind_group_layouts: &[&self.uniform_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Triangle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        })
    }
    
    /// Begin a render pass
    pub fn begin_render_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        view: &'a wgpu::TextureView,
        depth_view: &'a wgpu::TextureView,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        })
    }
    
    /// Render vertices with the specified pipeline
    pub fn render_vertices<'a>(
        &'a mut self,
        render_pass: &mut wgpu::RenderPass<'a>,
        pipeline_type: PipelineType,
        vertex_buffer: &'a wgpu::Buffer,
        vertex_count: u32,
        index_buffer: Option<(&'a wgpu::Buffer, u32)>,
    ) {
        // Ensure the pipeline exists first
        self.ensure_pipeline(pipeline_type);
        
        // Now get the pipeline and render
        let pipeline = self.get_pipeline(pipeline_type);
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        
        match index_buffer {
            Some((indices, index_count)) => {
                render_pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..index_count, 0, 0..1);
            }
            None => {
                render_pass.draw(0..vertex_count, 0..1);
            }
        }
    }
    
    /// Ensure direct line pipeline exists
    pub fn ensure_direct_line_pipeline(&mut self) {
        if self.direct_line_pipeline.is_none() {
            self.direct_line_pipeline = Some(self.create_direct_line_pipeline());
        }
    }
    
    /// Update transformation uniforms for direct viewport rendering
    pub fn update_direct_uniforms(
        &mut self,
        data_min: [f32; 2],
        data_max: [f32; 2],
        viewport_min: [f32; 2],
        viewport_max: [f32; 2],
    ) {
        self.direct_uniforms = DirectUniforms::new(data_min, data_max, viewport_min, viewport_max);
        self.queue.write_buffer(
            &self.direct_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.direct_uniforms]),
        );
    }
}

/// Utility functions for creating common vertex patterns
pub mod vertex_utils {
    use super::*;
    
    /// Create vertices for a line from start to end point
    pub fn create_line(start: Vec3, end: Vec3, color: Vec4) -> Vec<Vertex> {
        vec![
            Vertex::new(start, color),
            Vertex::new(end, color),
        ]
    }
    
    /// Create vertices for a triangle
    pub fn create_triangle(p1: Vec3, p2: Vec3, p3: Vec3, color: Vec4) -> Vec<Vertex> {
        vec![
            Vertex::new(p1, color),
            Vertex::new(p2, color),
            Vertex::new(p3, color),
        ]
    }
    
    /// Create vertices for a point cloud
    pub fn create_point_cloud(points: &[Vec3], colors: &[Vec4]) -> Vec<Vertex> {
        points.iter()
            .zip(colors.iter())
            .map(|(&pos, &color)| Vertex::new(pos, color))
            .collect()
    }
    
    /// Create vertices for a parametric line plot
    pub fn create_line_plot(x_data: &[f64], y_data: &[f64], color: Vec4) -> Vec<Vertex> {
        let mut vertices = Vec::new();
        

        
        for i in 1..x_data.len() {
            let start = Vec3::new(x_data[i-1] as f32, y_data[i-1] as f32, 0.0);
            let end = Vec3::new(x_data[i] as f32, y_data[i] as f32, 0.0);
            vertices.extend(create_line(start, end, color));
        }
        
        vertices
    }
    
    /// Create vertices for a scatter plot
    pub fn create_scatter_plot(x_data: &[f64], y_data: &[f64], color: Vec4) -> Vec<Vertex> {
        x_data.iter()
            .zip(y_data.iter())
            .map(|(&x, &y)| Vertex::new(Vec3::new(x as f32, y as f32, 0.0), color))
            .collect()
    }
}