//! WGPU-based rendering backend for high-performance plotting
//!
//! This module provides GPU-accelerated rendering using WGPU, supporting
//! both desktop and web targets for maximum compatibility.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::{core::scene::GpuVertexBuffer, gpu::shaders};

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
        let stride = std::mem::size_of::<Vertex>() as wgpu::BufferAddress;
        log::trace!(
            target: "runmat_plot",
            "vertex layout: size={}, stride={}",
            std::mem::size_of::<Vertex>(),
            stride
        );
        wgpu::VertexBufferLayout {
            array_stride: stride,
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
    pub data_min: [f32; 2],     // (x_min, y_min) in data space
    pub data_max: [f32; 2],     // (x_max, y_max) in data space
    pub viewport_min: [f32; 2], // NDC coordinates of viewport bottom-left
    pub viewport_max: [f32; 2], // NDC coordinates of viewport top-right
    pub viewport_px: [f32; 2],  // viewport size in pixels (width, height)
}

/// Style uniforms for direct point rendering (scatter markers)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PointStyleUniforms {
    pub face_color: [f32; 4],
    pub edge_color: [f32; 4],
    pub edge_thickness_px: f32,
    pub marker_shape: u32,
    pub _pad: [f32; 2],
}

impl Default for Uniforms {
    fn default() -> Self {
        Self::new()
    }
}

impl Uniforms {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            model: Mat4::IDENTITY.to_cols_array_2d(),
            normal_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
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
            [
                normal_mat.x_axis.x,
                normal_mat.x_axis.y,
                normal_mat.x_axis.z,
                0.0,
            ],
            [
                normal_mat.y_axis.x,
                normal_mat.y_axis.y,
                normal_mat.y_axis.z,
                0.0,
            ],
            [
                normal_mat.z_axis.x,
                normal_mat.z_axis.y,
                normal_mat.z_axis.z,
                0.0,
            ],
        ];
    }
}

impl DirectUniforms {
    pub fn new(
        data_min: [f32; 2],
        data_max: [f32; 2],
        viewport_min: [f32; 2],
        viewport_max: [f32; 2],
        viewport_px: [f32; 2],
    ) -> Self {
        Self {
            data_min,
            data_max,
            viewport_min,
            viewport_max,
            viewport_px,
        }
    }
}

/// Rendering pipeline types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineType {
    Points,
    Lines,
    Triangles,
    Scatter3,
    Textured,
}

/// High-performance WGPU renderer for interactive plotting
pub struct WgpuRenderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface_config: wgpu::SurfaceConfiguration,

    // Global MSAA sample count for pipelines/attachments
    pub msaa_sample_count: u32,

    // Rendering pipelines (traditional camera-based)
    point_pipeline: Option<wgpu::RenderPipeline>,
    line_pipeline: Option<wgpu::RenderPipeline>,
    triangle_pipeline: Option<wgpu::RenderPipeline>,

    // Direct rendering pipelines (optimized coordinate transformation)
    pub direct_line_pipeline: Option<wgpu::RenderPipeline>,
    pub direct_triangle_pipeline: Option<wgpu::RenderPipeline>,
    pub direct_point_pipeline: Option<wgpu::RenderPipeline>,
    image_pipeline: Option<wgpu::RenderPipeline>,
    image_bind_group_layout: wgpu::BindGroupLayout,
    image_sampler: wgpu::Sampler,
    point_style_bind_group_layout: wgpu::BindGroupLayout,

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
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            [0.0, 0.0],   // data_min
            [1.0, 1.0],   // data_max
            [-1.0, -1.0], // viewport_min (full NDC)
            [1.0, 1.0],   // viewport_max (full NDC)
            [1.0, 1.0],   // viewport_px
        );
        let direct_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Direct Uniform Buffer"),
            contents: bytemuck::cast_slice(&[direct_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create direct bind group layout for uniforms
        let direct_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let image_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Image Bind Group Layout"),
                entries: &[
                    // sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // texture view
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                ],
            });

        let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Image Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Point style bind group layout (face/edge colors, thickness, shape)
        let point_style_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Point Style Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        Self {
            device,
            queue,
            surface_config,
            msaa_sample_count: 1,
            point_pipeline: None,
            line_pipeline: None,
            triangle_pipeline: None,
            direct_line_pipeline: None,
            direct_triangle_pipeline: None,
            direct_point_pipeline: None,
            image_pipeline: None,
            image_bind_group_layout,
            image_sampler,
            point_style_bind_group_layout,
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

    /// Ensure MSAA state matches requested count. Rebuild pipelines if changed.
    pub fn ensure_msaa(&mut self, requested_count: u32) {
        let clamped = match requested_count {
            0 => 1,
            1 => 1,
            2 => 2,
            4 => 4,
            8 => 8,
            16 => 8, // clamp to 8 for portability
            _ => 4,  // default reasonable MSAA
        };
        if self.msaa_sample_count != clamped {
            self.msaa_sample_count = clamped;
            // Drop pipelines so they are recreated with new MSAA count
            self.point_pipeline = None;
            self.line_pipeline = None;
            self.triangle_pipeline = None;
            self.direct_line_pipeline = None;
            self.direct_triangle_pipeline = None;
            self.direct_point_pipeline = None;
            self.image_pipeline = None;
        }
    }

    /// Create a GPU texture and bind group for an RGBA8 image
    pub fn create_image_texture_and_bind_group(
        &self,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Image Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Upload data
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Image Bind Group"),
            layout: &self.image_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&self.image_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
            ],
        });
        (texture, texture_view, bind_group)
    }

    /// Create a vertex buffer from vertex data
    pub fn create_vertex_buffer(&self, vertices: &[Vertex]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            })
    }

    /// Choose the most efficient vertex buffer source for the provided data.
    pub fn vertex_buffer_from_sources(
        &self,
        gpu: Option<&GpuVertexBuffer>,
        cpu_vertices: &[Vertex],
    ) -> Option<Arc<wgpu::Buffer>> {
        if let Some(buffer) = gpu {
            Some(buffer.buffer.clone())
        } else if !cpu_vertices.is_empty() {
            Some(Arc::new(self.create_vertex_buffer(cpu_vertices)))
        } else {
            None
        }
    }

    /// Create a vertex buffer for direct points by expanding each point to a quad.
    /// This reuses Vertex but encodes corner index via tex_coords and marker size in normal.z
    pub fn create_direct_point_vertices(&self, points: &[Vertex], size_px: f32) -> Vec<Vertex> {
        let corners: [[f32; 2]; 6] = [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ];
        let mut out = Vec::with_capacity(points.len() * 6);
        for p in points {
            for c in corners {
                let mut v = *p;
                v.tex_coords = c; // tells shader which corner
                let sz = if size_px > 0.0 { size_px } else { p.normal[2] };
                v.normal = [p.normal[0], p.normal[1], sz];
                out.push(v);
            }
        }
        out
    }

    /// Create an index buffer from index data
    pub fn create_index_buffer(&self, indices: &[u32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
            PipelineType::Scatter3 => {
                // For now, use points pipeline - will optimize later
                self.ensure_pipeline(PipelineType::Points);
            }
            PipelineType::Textured => {
                if self.image_pipeline.is_none() {
                    self.image_pipeline = Some(self.create_image_pipeline());
                }
            }
        }
    }

    /// Get a pipeline reference (pipeline must already exist)
    pub fn get_pipeline(&self, pipeline_type: PipelineType) -> &wgpu::RenderPipeline {
        match pipeline_type {
            PipelineType::Points => self.point_pipeline.as_ref().unwrap(),
            PipelineType::Lines => self.line_pipeline.as_ref().unwrap(),
            PipelineType::Triangles => self.triangle_pipeline.as_ref().unwrap(),
            PipelineType::Scatter3 => self.get_pipeline(PipelineType::Points),
            PipelineType::Textured => self.image_pipeline.as_ref().unwrap(),
        }
    }

    /// Create point rendering pipeline
    fn create_point_pipeline(&self) -> wgpu::RenderPipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Point Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::POINT.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Point Pipeline Layout"),
                bind_group_layouts: &[&self.uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }

    /// Create line rendering pipeline
    fn create_line_pipeline(&self) -> wgpu::RenderPipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Line Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::LINE.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Line Pipeline Layout"),
                bind_group_layouts: &[&self.uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }

    /// Create optimized direct rendering pipeline for precise viewport mapping
    fn create_direct_line_pipeline(&self) -> wgpu::RenderPipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Direct Line Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::LINE_DIRECT.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Direct Line Pipeline Layout"),
                bind_group_layouts: &[&self.direct_uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }

    /// Create optimized direct triangle pipeline (2D fills) for precise viewport mapping
    fn create_direct_triangle_pipeline(&self) -> wgpu::RenderPipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Direct Triangle Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::LINE_DIRECT.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Direct Triangle Pipeline Layout"),
                bind_group_layouts: &[&self.direct_uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Direct Triangle Pipeline"),
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
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }

    /// Create optimized direct point pipeline (instanced quads per point)
    fn create_direct_point_pipeline(&self) -> wgpu::RenderPipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Direct Point Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::POINT_DIRECT.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Direct Point Pipeline Layout"),
                bind_group_layouts: &[
                    &self.direct_uniform_bind_group_layout,
                    &self.point_style_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Direct Point Pipeline"),
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
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }

    /// Create style bind group for scatter points. Returns (buffer, bind_group).
    pub fn create_point_style_bind_group(
        &self,
        style: PointStyleUniforms,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Point Style Uniform Buffer"),
                contents: bytemuck::bytes_of(&style),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Point Style Bind Group"),
            layout: &self.point_style_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        (buffer, bind_group)
    }

    /// Create textured image pipeline (direct viewport mapping + sampled texture)
    fn create_image_pipeline(&self) -> wgpu::RenderPipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Image Direct Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::IMAGE_DIRECT.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Image Pipeline Layout"),
                bind_group_layouts: &[
                    &self.direct_uniform_bind_group_layout,
                    &self.image_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Image Pipeline"),
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
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }

    /// Create triangle rendering pipeline
    fn create_triangle_pipeline(&self) -> wgpu::RenderPipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Triangle Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::TRIANGLE.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Triangle Pipeline Layout"),
                bind_group_layouts: &[&self.uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    cull_mode: None, // Disable culling for 2D plotting
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None, // Disable depth testing for 2D plotting
                multisample: wgpu::MultisampleState {
                    count: self.msaa_sample_count,
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
        _depth_view: &'a wgpu::TextureView,
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
            depth_stencil_attachment: None, // No depth testing for 2D plotting
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

    /// Ensure direct triangle pipeline exists
    pub fn ensure_direct_triangle_pipeline(&mut self) {
        if self.direct_triangle_pipeline.is_none() {
            self.direct_triangle_pipeline = Some(self.create_direct_triangle_pipeline());
        }
    }

    /// Ensure direct point pipeline exists
    pub fn ensure_direct_point_pipeline(&mut self) {
        if self.direct_point_pipeline.is_none() {
            self.direct_point_pipeline = Some(self.create_direct_point_pipeline());
        }
    }

    /// Ensure image pipeline exists
    pub fn ensure_image_pipeline(&mut self) {
        if self.image_pipeline.is_none() {
            self.image_pipeline = Some(self.create_image_pipeline());
        }
    }

    /// Update transformation uniforms for direct viewport rendering
    pub fn update_direct_uniforms(
        &mut self,
        data_min: [f32; 2],
        data_max: [f32; 2],
        viewport_min: [f32; 2],
        viewport_max: [f32; 2],
        viewport_px: [f32; 2],
    ) {
        self.direct_uniforms =
            DirectUniforms::new(data_min, data_max, viewport_min, viewport_max, viewport_px);
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
    use glam::Vec2;

    /// Create vertices for a line from start to end point
    pub fn create_line(start: Vec3, end: Vec3, color: Vec4) -> Vec<Vertex> {
        vec![Vertex::new(start, color), Vertex::new(end, color)]
    }

    /// CPU polyline extrusion for thick lines (butt caps, miter joins simplified)
    /// Input: contiguous points. Output: triangle list vertices.
    pub fn extrude_polyline(points: &[Vec3], color: Vec4, width: f32) -> Vec<Vertex> {
        let mut out: Vec<Vertex> = Vec::new();
        if points.len() < 2 {
            return out;
        }
        let half_w = (width.max(1.0)) * 0.5;
        for i in 0..points.len() - 1 {
            let p0 = points[i];
            let p1 = points[i + 1];
            let dir = (p1 - p0).truncate();
            let len = (dir.x * dir.x + dir.y * dir.y).sqrt().max(1e-6);
            let nx = -dir.y / len;
            let ny = dir.x / len;
            let offset = Vec3::new(nx * half_w, ny * half_w, 0.0);
            // Quad corners in CCW
            let a = p0 - offset;
            let b = p0 + offset;
            let c = p1 + offset;
            let d = p1 - offset;
            // Two triangles: a-b-c and a-c-d
            out.push(Vertex::new(a, color));
            out.push(Vertex::new(b, color));
            out.push(Vertex::new(c, color));
            out.push(Vertex::new(a, color));
            out.push(Vertex::new(c, color));
            out.push(Vertex::new(d, color));
        }
        out
    }

    fn line_intersection(p: Vec2, r: Vec2, q: Vec2, s: Vec2) -> Option<Vec2> {
        let rxs = r.perp_dot(s);
        if rxs.abs() < 1e-6 {
            return None;
        }
        let t = (q - p).perp_dot(s) / rxs;
        Some(p + r * t)
    }

    /// Extrude polyline with join styles at internal vertices.
    pub fn extrude_polyline_with_join(
        points: &[Vec3],
        color: Vec4,
        width: f32,
        join: crate::plots::line::LineJoin,
    ) -> Vec<Vertex> {
        let mut out: Vec<Vertex> = Vec::new();
        if points.len() < 2 {
            return out;
        }
        let half_w = (width.max(1.0)) * 0.5;
        // Base quads
        out.extend(extrude_polyline(points, color, width));

        // Joins
        for i in 1..points.len() - 1 {
            let p_prev = points[i - 1];
            let p = points[i];
            let p_next = points[i + 1];
            let d0 = (p - p_prev).truncate();
            let d1 = (p_next - p).truncate();
            let l0 = d0.length().max(1e-6);
            let l1 = d1.length().max(1e-6);
            let n0 = Vec2::new(-d0.y / l0, d0.x / l0);
            let n1 = Vec2::new(-d1.y / l1, d1.x / l1);
            let turn = d0.perp_dot(d1); // >0 left turn, <0 right turn

            if turn > 1e-6 {
                // Left turn: outer side is left (use +n)
                let left0 = p.truncate() + n0 * half_w;
                let left1 = p.truncate() + n1 * half_w;
                match join {
                    crate::plots::line::LineJoin::Bevel => {
                        // Triangle wedge (p, left0, left1)
                        out.push(Vertex::new(p, color));
                        out.push(Vertex::new(left0.extend(0.0), color));
                        out.push(Vertex::new(left1.extend(0.0), color));
                    }
                    crate::plots::line::LineJoin::Miter => {
                        let dir_edge0 = (p.truncate() - p_prev.truncate()).normalize_or_zero();
                        let dir_edge1 = (p_next.truncate() - p.truncate()).normalize_or_zero();
                        let l_edge = line_intersection(left0, dir_edge0, left1, dir_edge1);
                        if let Some(miter) = l_edge {
                            // fill wedge left0-miter-left1
                            out.push(Vertex::new(left0.extend(0.0), color));
                            out.push(Vertex::new(miter.extend(0.0), color));
                            out.push(Vertex::new(left1.extend(0.0), color));
                        } else {
                            // fallback to bevel
                            out.push(Vertex::new(p, color));
                            out.push(Vertex::new(left0.extend(0.0), color));
                            out.push(Vertex::new(left1.extend(0.0), color));
                        }
                    }
                    crate::plots::line::LineJoin::Round => {
                        // Arc fan from left0 -> left1 around p
                        let center = p.truncate();
                        let a0 = (left0 - center).to_array();
                        let a1 = (left1 - center).to_array();
                        let ang0 = a0[1].atan2(a0[0]);
                        let mut ang1 = a1[1].atan2(a1[0]);
                        // Ensure CCW sweep
                        if ang1 < ang0 {
                            ang1 += std::f32::consts::TAU;
                        }
                        let steps = 10usize;
                        let dtheta = (ang1 - ang0) / steps as f32;
                        let r = half_w;
                        for k in 0..steps {
                            let theta0 = ang0 + dtheta * k as f32;
                            let theta1 = ang0 + dtheta * (k + 1) as f32;
                            let v0 =
                                Vec2::new(center.x + theta0.cos() * r, center.y + theta0.sin() * r);
                            let v1 =
                                Vec2::new(center.x + theta1.cos() * r, center.y + theta1.sin() * r);
                            out.push(Vertex::new(p, color));
                            out.push(Vertex::new(v0.extend(0.0), color));
                            out.push(Vertex::new(v1.extend(0.0), color));
                        }
                    }
                }
            } else if turn < -1e-6 {
                // Right turn: outer side is right (use -n)
                let right0 = p.truncate() - n0 * half_w;
                let right1 = p.truncate() - n1 * half_w;
                match join {
                    crate::plots::line::LineJoin::Bevel => {
                        out.push(Vertex::new(p, color));
                        out.push(Vertex::new(right1.extend(0.0), color));
                        out.push(Vertex::new(right0.extend(0.0), color));
                    }
                    crate::plots::line::LineJoin::Miter => {
                        let dir_edge0 = (p.truncate() - p_prev.truncate()).normalize_or_zero();
                        let dir_edge1 = (p_next.truncate() - p.truncate()).normalize_or_zero();
                        let l_edge = line_intersection(right0, dir_edge0, right1, dir_edge1);
                        if let Some(miter) = l_edge {
                            out.push(Vertex::new(right1.extend(0.0), color));
                            out.push(Vertex::new(miter.extend(0.0), color));
                            out.push(Vertex::new(right0.extend(0.0), color));
                        } else {
                            out.push(Vertex::new(p, color));
                            out.push(Vertex::new(right1.extend(0.0), color));
                            out.push(Vertex::new(right0.extend(0.0), color));
                        }
                    }
                    crate::plots::line::LineJoin::Round => {
                        let center = p.truncate();
                        let a0 = (right0 - center).to_array();
                        let a1 = (right1 - center).to_array();
                        let mut ang0 = a0[1].atan2(a0[0]);
                        let mut ang1 = a1[1].atan2(a1[0]);
                        // Ensure CW sweep becomes CCW by swapping
                        if ang0 < ang1 {
                            std::mem::swap(&mut ang0, &mut ang1);
                        }
                        let steps = 10usize;
                        let dtheta = (ang0 - ang1) / steps as f32;
                        let r = half_w;
                        for k in 0..steps {
                            let theta0 = ang0 - dtheta * k as f32;
                            let theta1 = ang0 - dtheta * (k + 1) as f32;
                            let v0 =
                                Vec2::new(center.x + theta0.cos() * r, center.y + theta0.sin() * r);
                            let v1 =
                                Vec2::new(center.x + theta1.cos() * r, center.y + theta1.sin() * r);
                            out.push(Vertex::new(p, color));
                            out.push(Vertex::new(v0.extend(0.0), color));
                            out.push(Vertex::new(v1.extend(0.0), color));
                        }
                    }
                }
            }
        }

        out
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
        points
            .iter()
            .zip(colors.iter())
            .map(|(&pos, &color)| Vertex::new(pos, color))
            .collect()
    }

    /// Create vertices for a parametric line plot (1px line segments)
    pub fn create_line_plot(x_data: &[f64], y_data: &[f64], color: Vec4) -> Vec<Vertex> {
        let mut vertices = Vec::new();

        for i in 1..x_data.len() {
            let start = Vec3::new(x_data[i - 1] as f32, y_data[i - 1] as f32, 0.0);
            let end = Vec3::new(x_data[i] as f32, y_data[i] as f32, 0.0);
            vertices.extend(create_line(start, end, color));
        }

        vertices
    }

    /// Create dashed/dotted line vertices by selectively including segments.
    /// Approximation: pattern is applied per original segment index.
    pub fn create_line_plot_dashed(
        x_data: &[f64],
        y_data: &[f64],
        color: Vec4,
        style: crate::plots::line::LineStyle,
    ) -> Vec<Vertex> {
        let mut vertices = Vec::new();
        for i in 1..x_data.len() {
            let include = match style {
                crate::plots::line::LineStyle::Solid => true,
                crate::plots::line::LineStyle::Dashed => (i % 4) < 2, // on,on,off,off
                crate::plots::line::LineStyle::Dotted => false,       // handled elsewhere as points
                crate::plots::line::LineStyle::DashDot => {
                    let m = i % 6;
                    m < 2 || m == 3 // on,on,off,on,off,off
                }
            };
            if include {
                let start = Vec3::new(x_data[i - 1] as f32, y_data[i - 1] as f32, 0.0);
                let end = Vec3::new(x_data[i] as f32, y_data[i] as f32, 0.0);
                vertices.extend(create_line(start, end, color));
            }
        }
        vertices
    }

    /// Create thick polyline as triangles (used when line width > 1)
    pub fn create_thick_polyline(
        x_data: &[f64],
        y_data: &[f64],
        color: Vec4,
        width_px: f32,
    ) -> Vec<Vertex> {
        let mut pts: Vec<Vec3> = Vec::with_capacity(x_data.len());
        for i in 0..x_data.len() {
            pts.push(Vec3::new(x_data[i] as f32, y_data[i] as f32, 0.0));
        }
        extrude_polyline(&pts, color, width_px)
    }

    /// Thick polyline with join style
    pub fn create_thick_polyline_with_join(
        x_data: &[f64],
        y_data: &[f64],
        color: Vec4,
        width_px: f32,
        join: crate::plots::line::LineJoin,
    ) -> Vec<Vertex> {
        let mut pts: Vec<Vec3> = Vec::with_capacity(x_data.len());
        for i in 0..x_data.len() {
            pts.push(Vec3::new(x_data[i] as f32, y_data[i] as f32, 0.0));
        }
        extrude_polyline_with_join(&pts, color, width_px, join)
    }

    /// Create dashed/dotted thick polyline by skipping segments in the extruder.
    pub fn create_thick_polyline_dashed(
        x_data: &[f64],
        y_data: &[f64],
        color: Vec4,
        width_px: f32,
        style: crate::plots::line::LineStyle,
    ) -> Vec<Vertex> {
        let mut out: Vec<Vertex> = Vec::new();
        if x_data.len() < 2 {
            return out;
        }
        let pts: Vec<Vec3> = x_data
            .iter()
            .zip(y_data.iter())
            .map(|(&x, &y)| Vec3::new(x as f32, y as f32, 0.0))
            .collect();
        for i in 0..pts.len() - 1 {
            let include = match style {
                crate::plots::line::LineStyle::Solid => true,
                crate::plots::line::LineStyle::Dashed => (i % 4) < 2,
                crate::plots::line::LineStyle::Dotted => false,
                crate::plots::line::LineStyle::DashDot => {
                    let m = i % 6;
                    m < 2 || m == 3
                }
            };
            if include {
                let seg = [pts[i], pts[i + 1]];
                out.extend(extrude_polyline(&seg, color, width_px));
            }
        }
        out
    }

    /// Square caps variant: extend endpoints by half width
    pub fn create_thick_polyline_square_caps(
        x_data: &[f64],
        y_data: &[f64],
        color: Vec4,
        width_px: f32,
    ) -> Vec<Vertex> {
        if x_data.len() < 2 {
            return Vec::new();
        }
        let mut pts: Vec<Vec3> = Vec::with_capacity(x_data.len());
        for i in 0..x_data.len() {
            pts.push(Vec3::new(x_data[i] as f32, y_data[i] as f32, 0.0));
        }
        // extend start
        let dir0 = (pts[1] - pts[0]).truncate();
        let len0 = (dir0.x * dir0.x + dir0.y * dir0.y).sqrt().max(1e-6);
        let ext0 = Vec3::new(
            -(dir0.x / len0) * (width_px * 0.5),
            -(dir0.y / len0) * (width_px * 0.5),
            0.0,
        );
        pts[0] = pts[0] + ext0;
        // extend end
        let n = pts.len();
        let dir1 = (pts[n - 1] - pts[n - 2]).truncate();
        let len1 = (dir1.x * dir1.x + dir1.y * dir1.y).sqrt().max(1e-6);
        let ext1 = Vec3::new(
            (dir1.x / len1) * (width_px * 0.5),
            (dir1.y / len1) * (width_px * 0.5),
            0.0,
        );
        pts[n - 1] = pts[n - 1] + ext1;
        extrude_polyline(&pts, color, width_px)
    }

    /// Round caps variant: square caps plus approximated semicircle fan at ends
    pub fn create_thick_polyline_round_caps(
        x_data: &[f64],
        y_data: &[f64],
        color: Vec4,
        width_px: f32,
        segments: usize,
    ) -> Vec<Vertex> {
        let mut base = create_thick_polyline_square_caps(x_data, y_data, color, width_px);
        if x_data.len() < 2 {
            return base;
        }
        let r = width_px * 0.5;
        // start fan
        let p0 = Vec3::new(x_data[0] as f32, y_data[0] as f32, 0.0);
        let p1 = Vec3::new(x_data[1] as f32, y_data[1] as f32, 0.0);
        let dir0 = (p1 - p0).truncate();
        let theta0 = dir0.y.atan2(dir0.x) + std::f32::consts::PI; // facing backward
        for i in 0..segments {
            let a0 = theta0 - std::f32::consts::PI * (i as f32 / segments as f32);
            let a1 = theta0 - std::f32::consts::PI * ((i + 1) as f32 / segments as f32);
            let v0 = Vec3::new(p0.x + a0.cos() * r, p0.y + a0.sin() * r, 0.0);
            let v1 = Vec3::new(p0.x + a1.cos() * r, p0.y + a1.sin() * r, 0.0);
            base.push(Vertex::new(p0, color));
            base.push(Vertex::new(v0, color));
            base.push(Vertex::new(v1, color));
        }
        // end fan
        let n = x_data.len();
        let q0 = Vec3::new(x_data[n - 2] as f32, y_data[n - 2] as f32, 0.0);
        let q1 = Vec3::new(x_data[n - 1] as f32, y_data[n - 1] as f32, 0.0);
        let dir1 = (q1 - q0).truncate();
        let theta1 = dir1.y.atan2(dir1.x);
        let center = q1;
        for i in 0..segments {
            let a0 = theta1 - std::f32::consts::PI * (i as f32 / segments as f32);
            let a1 = theta1 - std::f32::consts::PI * ((i + 1) as f32 / segments as f32);
            let v0 = Vec3::new(center.x + a0.cos() * r, center.y + a0.sin() * r, 0.0);
            let v1 = Vec3::new(center.x + a1.cos() * r, center.y + a1.sin() * r, 0.0);
            base.push(Vertex::new(center, color));
            base.push(Vertex::new(v0, color));
            base.push(Vertex::new(v1, color));
        }
        base
    }

    /// Create vertices for a scatter plot
    pub fn create_scatter_plot(x_data: &[f64], y_data: &[f64], color: Vec4) -> Vec<Vertex> {
        x_data
            .iter()
            .zip(y_data.iter())
            .map(|(&x, &y)| Vertex::new(Vec3::new(x as f32, y as f32, 0.0), color))
            .collect()
    }
}
