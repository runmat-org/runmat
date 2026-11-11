//! WGPU-based rendering backend for high-performance plotting
//!
//! This module provides GPU-accelerated rendering using WGPU, supporting
//! both desktop and web targets for maximum compatibility.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::core::DepthMode;

mod edge_stroke_pipeline;
mod instanced_pipeline;
use crate::core::instances::InstanceRaw;
use crate::{
    core::scene::{GpuIndirectDraw, GpuVertexBuffer},
    gpu::shaders,
};

mod direct;
mod pipelines;
mod resources;
mod types;
pub use types::*;

/// High-performance WGPU renderer for interactive plotting
pub struct WgpuRenderer {
    pub access: runmat_accelerate_api::WgpuDeviceAccess,
    pub surface_config: wgpu::SurfaceConfiguration,

    // Global MSAA sample count for pipelines/attachments
    pub msaa_sample_count: u32,

    // Rendering pipelines (traditional camera-based)
    point_pipeline: Option<wgpu::RenderPipeline>,
    line_pipeline: Option<wgpu::RenderPipeline>,
    line_no_depth_pipeline: Option<wgpu::RenderPipeline>,
    annotation_stroke_pipeline: Option<wgpu::RenderPipeline>,
    edge_stroke_pipeline: Option<wgpu::RenderPipeline>,
    edge_stroke_no_depth_pipeline: Option<wgpu::RenderPipeline>,
    /// Device pixels per CSS pixel for screen-space stroke widths.
    pub pixels_per_point: f32,
    triangle_pipeline: Option<wgpu::RenderPipeline>,
    /// Blended surfaces drawn after opaque ones: depth-tested, without depth writes, so a
    /// translucent surface never hides what lies behind it.
    translucent_triangle_pipeline: Option<wgpu::RenderPipeline>,
    /// Instanced prototypes and depth-probed edge strokes.
    extra_pipelines: instanced_pipeline::ExtraPipelines,

    // Direct rendering pipelines (optimized coordinate transformation)
    pub direct_line_pipeline: Option<wgpu::RenderPipeline>,
    pub direct_triangle_pipeline: Option<wgpu::RenderPipeline>,
    pub direct_point_pipeline: Option<wgpu::RenderPipeline>,
    image_pipeline: Option<wgpu::RenderPipeline>,
    image_bind_group_layout: wgpu::BindGroupLayout,
    image_sampler: wgpu::Sampler,
    point_style_bind_group_layout: wgpu::BindGroupLayout,
    marker_screen_bind_group_layout: wgpu::BindGroupLayout,
    marker_screen_uniform_buffer: wgpu::Buffer,
    marker_screen_bind_group: wgpu::BindGroup,
    axes_marker_screen_uniform_buffers: Vec<wgpu::Buffer>,
    axes_marker_screen_bind_groups: Vec<wgpu::BindGroup>,

    // Grid helper uniforms/pipeline (3D only)
    grid_uniform_buffer: wgpu::Buffer,
    pub grid_uniform_bind_group: wgpu::BindGroup,
    grid_uniform_bind_group_layout: wgpu::BindGroupLayout,
    axes_grid_uniform_buffers: Vec<wgpu::Buffer>,
    axes_grid_uniform_bind_groups: Vec<wgpu::BindGroup>,
    grid_plane_pipeline: Option<wgpu::RenderPipeline>,

    // Uniform resources (traditional)
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    axes_uniform_buffers: Vec<wgpu::Buffer>,
    axes_uniform_bind_groups: Vec<wgpu::BindGroup>,

    // Direct uniform resources (optimized coordinate transformation)
    direct_uniform_buffer: wgpu::Buffer,
    pub direct_uniform_bind_group: wgpu::BindGroup,
    direct_uniform_bind_group_layout: wgpu::BindGroupLayout,
    axes_direct_uniform_buffers: Vec<wgpu::Buffer>,
    axes_direct_uniform_bind_groups: Vec<wgpu::BindGroup>,

    // Current uniforms
    uniforms: Uniforms,
    direct_uniforms: DirectUniforms,

    // Depth resources (used by camera-based 3D rendering paths)
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<Arc<wgpu::TextureView>>,
    depth_extent: (u32, u32, u32), // (w, h, sample_count)

    // MSAA color resources for resolving into single-sampled targets.
    msaa_color_texture: Option<wgpu::Texture>,
    msaa_color_view: Option<Arc<wgpu::TextureView>>,
    msaa_color_extent: (u32, u32, u32), // (w, h, sample_count)

    /// Depth mapping mode for camera-based 3D pipelines.
    pub depth_mode: DepthMode,
}

impl WgpuRenderer {
    fn create_uniform_bind_group_for_buffer(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        buffer: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        emission
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
                label: Some(label),
            })
    }

    fn create_direct_uniform_bind_group_for_buffer(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        buffer: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        emission
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.direct_uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
                label: Some(label),
            })
    }

    fn create_grid_uniform_bind_group_for_buffer(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        buffer: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        emission
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.grid_uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
    }

    fn create_marker_screen_bind_group_for_buffer(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        buffer: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        emission
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.marker_screen_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
    }

    pub fn ensure_axes_uniform_capacity(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        axes_count: usize,
    ) {
        while self.axes_uniform_buffers.len() < axes_count {
            let idx = self.axes_uniform_buffers.len();
            let buffer = emission
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Axes Uniform Buffer {idx}")),
                    contents: bytemuck::cast_slice(&[Uniforms::new()]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let bind_group = self.create_uniform_bind_group_for_buffer(
                emission,
                &buffer,
                &format!("axes_uniform_bind_group_{idx}"),
            );
            self.axes_uniform_buffers.push(buffer);
            self.axes_uniform_bind_groups.push(bind_group);
        }
        while self.axes_direct_uniform_buffers.len() < axes_count {
            let idx = self.axes_direct_uniform_buffers.len();
            let buffer = emission
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Axes Direct Uniform Buffer {idx}")),
                    contents: bytemuck::cast_slice(&[DirectUniforms::new(
                        [0.0, 0.0],
                        [1.0, 1.0],
                        [-1.0, -1.0],
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [0, 0],
                    )]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let bind_group = self.create_direct_uniform_bind_group_for_buffer(
                emission,
                &buffer,
                &format!("axes_direct_uniform_bind_group_{idx}"),
            );
            self.axes_direct_uniform_buffers.push(buffer);
            self.axes_direct_uniform_bind_groups.push(bind_group);
        }
        while self.axes_grid_uniform_buffers.len() < axes_count {
            let idx = self.axes_grid_uniform_buffers.len();
            let buffer = emission
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Axes Grid Uniform Buffer {idx}")),
                    contents: bytemuck::cast_slice(&[GridUniforms::default()]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let bind_group = self.create_grid_uniform_bind_group_for_buffer(
                emission,
                &buffer,
                &format!("axes_grid_uniform_bind_group_{idx}"),
            );
            self.axes_grid_uniform_buffers.push(buffer);
            self.axes_grid_uniform_bind_groups.push(bind_group);
        }
        while self.axes_marker_screen_uniform_buffers.len() < axes_count {
            let idx = self.axes_marker_screen_uniform_buffers.len();
            let buffer = emission
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Axes Marker Screen Uniform Buffer {idx}")),
                    contents: bytemuck::cast_slice(&[MarkerScreenUniforms {
                        viewport_px: [1.0, 1.0],
                        pixels_per_point: 1.0,
                        _pad: 0.0,
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let bind_group = self.create_marker_screen_bind_group_for_buffer(
                emission,
                &buffer,
                &format!("axes_marker_screen_bind_group_{idx}"),
            );
            self.axes_marker_screen_uniform_buffers.push(buffer);
            self.axes_marker_screen_bind_groups.push(bind_group);
        }
    }

    /// Create a new WGPU renderer
    pub fn new(
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        surface_config: wgpu::SurfaceConfiguration,
    ) -> Self {
        let device = emission.device();
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
            [0, 0],       // log_flags
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
            // Enlarged images show each pixel as a flat cell, as MATLAB draws them.
            mag_filter: wgpu::FilterMode::Nearest,
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
        let marker_screen_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Marker Screen Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // Depth-probed edge strokes read the viewport in their fragment stage.
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let marker_screen_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Marker Screen Uniform Buffer"),
                contents: bytemuck::cast_slice(&[MarkerScreenUniforms {
                    viewport_px: [1.0, 1.0],
                    pixels_per_point: 1.0,
                    _pad: 0.0,
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let marker_screen_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Marker Screen Bind Group"),
            layout: &marker_screen_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: marker_screen_uniform_buffer.as_entire_binding(),
            }],
        });

        // Grid uniforms (3D helper plane)
        let grid_uniforms = GridUniforms::default();
        let grid_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Uniform Buffer"),
            contents: bytemuck::cast_slice(&[grid_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let grid_uniform_bind_group_layout =
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
                label: Some("grid_uniform_bind_group_layout"),
            });
        let grid_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_uniform_bind_group"),
            layout: &grid_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: grid_uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            access: emission.access().clone(),
            surface_config,
            msaa_sample_count: 1,
            point_pipeline: None,
            line_pipeline: None,
            line_no_depth_pipeline: None,
            annotation_stroke_pipeline: None,
            edge_stroke_pipeline: None,
            edge_stroke_no_depth_pipeline: None,
            pixels_per_point: 1.0,
            triangle_pipeline: None,
            translucent_triangle_pipeline: None,
            extra_pipelines: Default::default(),
            direct_line_pipeline: None,
            direct_triangle_pipeline: None,
            direct_point_pipeline: None,
            image_pipeline: None,
            image_bind_group_layout,
            image_sampler,
            point_style_bind_group_layout,
            marker_screen_bind_group_layout,
            marker_screen_uniform_buffer,
            marker_screen_bind_group,
            axes_marker_screen_uniform_buffers: Vec::new(),
            axes_marker_screen_bind_groups: Vec::new(),
            grid_uniform_buffer,
            grid_uniform_bind_group,
            grid_uniform_bind_group_layout,
            axes_grid_uniform_buffers: Vec::new(),
            axes_grid_uniform_bind_groups: Vec::new(),
            grid_plane_pipeline: None,
            uniform_buffer,
            uniform_bind_group,
            uniform_bind_group_layout,
            axes_uniform_buffers: Vec::new(),
            axes_uniform_bind_groups: Vec::new(),
            direct_uniform_buffer,
            direct_uniform_bind_group,
            direct_uniform_bind_group_layout,
            axes_direct_uniform_buffers: Vec::new(),
            axes_direct_uniform_bind_groups: Vec::new(),
            uniforms,
            direct_uniforms,
            depth_texture: None,
            depth_view: None,
            depth_extent: (0, 0, 0),
            msaa_color_texture: None,
            msaa_color_view: None,
            msaa_color_extent: (0, 0, 0),
            depth_mode: DepthMode::default(),
        }
    }

    pub fn update_grid_uniforms(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        uniforms: GridUniforms,
    ) {
        emission.queue().write_buffer(
            &self.grid_uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
        self.ensure_axes_uniform_capacity(emission, 1);
        emission.queue().write_buffer(
            &self.axes_grid_uniform_buffers[0],
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    pub fn update_grid_uniforms_for_axes(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        axes_index: usize,
        uniforms: GridUniforms,
    ) {
        self.ensure_axes_uniform_capacity(emission, axes_index + 1);
        emission.queue().write_buffer(
            &self.axes_grid_uniform_buffers[axes_index],
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    pub fn get_grid_uniform_bind_group_for_axes(&self, axes_index: usize) -> &wgpu::BindGroup {
        self.axes_grid_uniform_bind_groups
            .get(axes_index)
            .unwrap_or(&self.grid_uniform_bind_group)
    }

    /// Camera-relative studio lighting is restricted to retained CAD scenes. Over a dark
    /// background it adds a rim light, so dark parts keep their outline (`dark_background`).
    pub fn configure_cad_lighting(&mut self, view: Mat4, enabled: bool, dark_background: bool) {
        self.uniforms.cad_lighting = [u32::from(enabled), u32::from(dark_background), 0, 0];
        self.uniforms.cad_view_rotation = [
            [view.x_axis.x, view.x_axis.y, view.x_axis.z, 0.0],
            [view.y_axis.x, view.y_axis.y, view.y_axis.z, 0.0],
            [view.z_axis.x, view.z_axis.y, view.z_axis.z, 0.0],
        ];
    }

    /// Update uniform buffer with new matrices
    pub fn update_uniforms(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        view_proj: Mat4,
        model: Mat4,
    ) {
        self.uniforms.update_view_proj(view_proj);
        self.uniforms.update_model(model);

        emission.queue().write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
        self.ensure_axes_uniform_capacity(emission, 1);
        emission.queue().write_buffer(
            &self.axes_uniform_buffers[0],
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    pub fn update_uniforms_for_axes(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        axes_index: usize,
        view_proj: Mat4,
        model: Mat4,
    ) {
        self.ensure_axes_uniform_capacity(emission, axes_index + 1);
        let mut uniforms = self.uniforms;
        uniforms.update_view_proj(view_proj);
        uniforms.update_model(model);
        emission.queue().write_buffer(
            &self.axes_uniform_buffers[axes_index],
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    pub fn update_marker_screen_uniforms(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        viewport_px: [f32; 2],
    ) {
        let uniforms = MarkerScreenUniforms {
            viewport_px: [viewport_px[0].max(1.0), viewport_px[1].max(1.0)],
            pixels_per_point: self.pixels_per_point,
            _pad: 0.0,
        };
        emission.queue().write_buffer(
            &self.marker_screen_uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
        self.ensure_axes_uniform_capacity(emission, 1);
        emission.queue().write_buffer(
            &self.axes_marker_screen_uniform_buffers[0],
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    pub fn update_marker_screen_uniforms_for_axes(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        axes_index: usize,
        viewport_px: [f32; 2],
    ) {
        self.ensure_axes_uniform_capacity(emission, axes_index + 1);
        let uniforms = MarkerScreenUniforms {
            viewport_px: [viewport_px[0].max(1.0), viewport_px[1].max(1.0)],
            pixels_per_point: self.pixels_per_point,
            _pad: 0.0,
        };
        emission.queue().write_buffer(
            &self.axes_marker_screen_uniform_buffers[axes_index],
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    /// Get the uniform bind group for rendering
    pub fn get_uniform_bind_group(&self) -> &wgpu::BindGroup {
        &self.uniform_bind_group
    }

    pub fn get_uniform_bind_group_for_axes(&self, axes_index: usize) -> &wgpu::BindGroup {
        self.axes_uniform_bind_groups
            .get(axes_index)
            .unwrap_or(&self.uniform_bind_group)
    }

    pub fn get_marker_screen_bind_group(&self) -> &wgpu::BindGroup {
        &self.marker_screen_bind_group
    }

    pub fn get_marker_screen_bind_group_for_axes(&self, axes_index: usize) -> &wgpu::BindGroup {
        self.axes_marker_screen_bind_groups
            .get(axes_index)
            .unwrap_or(&self.marker_screen_bind_group)
    }

    /// Ensure pipeline exists for the specified type
    pub fn ensure_pipeline(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        pipeline_type: PipelineType,
    ) {
        match pipeline_type {
            PipelineType::Points => {
                if self.point_pipeline.is_none() {
                    self.point_pipeline = Some(self.create_point_pipeline(emission));
                }
            }
            PipelineType::Lines => {
                if self.line_pipeline.is_none() {
                    self.line_pipeline = Some(self.create_line_pipeline(emission));
                }
            }
            PipelineType::LinesNoDepth => {
                if self.line_no_depth_pipeline.is_none() {
                    self.line_no_depth_pipeline =
                        Some(self.create_line_no_depth_pipeline(emission));
                }
            }
            PipelineType::AnnotationStroke => {
                if self.annotation_stroke_pipeline.is_none() {
                    self.annotation_stroke_pipeline =
                        Some(self.create_annotation_stroke_pipeline(emission));
                }
            }
            PipelineType::EdgeStroke => {
                if self.edge_stroke_pipeline.is_none() {
                    self.edge_stroke_pipeline =
                        Some(self.create_edge_stroke_pipeline(emission, true));
                }
            }
            PipelineType::EdgeStrokeNoDepth => {
                if self.edge_stroke_no_depth_pipeline.is_none() {
                    self.edge_stroke_no_depth_pipeline =
                        Some(self.create_edge_stroke_pipeline(emission, false));
                }
            }
            PipelineType::Triangles => {
                if self.triangle_pipeline.is_none() {
                    self.triangle_pipeline = Some(self.create_triangle_pipeline(emission, true));
                }
                if self.translucent_triangle_pipeline.is_none() {
                    self.translucent_triangle_pipeline =
                        Some(self.create_triangle_pipeline(emission, false));
                }
            }
            PipelineType::Scatter3 => {
                // Scatter3 shares marker semantics with point-based marker plots.
                self.ensure_pipeline(emission, PipelineType::Points);
            }
            PipelineType::Textured => {
                if self.image_pipeline.is_none() {
                    self.image_pipeline = Some(self.create_image_pipeline(emission));
                }
            }
        }
    }

    /// Get a pipeline reference (pipeline must already exist)
    pub fn get_pipeline(&self, pipeline_type: PipelineType) -> &wgpu::RenderPipeline {
        match pipeline_type {
            PipelineType::Points => self.point_pipeline.as_ref().unwrap(),
            PipelineType::Lines => self.line_pipeline.as_ref().unwrap(),
            PipelineType::LinesNoDepth => self.line_no_depth_pipeline.as_ref().unwrap(),
            PipelineType::AnnotationStroke => self.annotation_stroke_pipeline.as_ref().unwrap(),
            PipelineType::EdgeStroke => self.edge_stroke_pipeline.as_ref().unwrap(),
            PipelineType::EdgeStrokeNoDepth => self.edge_stroke_no_depth_pipeline.as_ref().unwrap(),
            PipelineType::Triangles => self.triangle_pipeline.as_ref().unwrap(),
            PipelineType::Scatter3 => self.get_pipeline(PipelineType::Points),
            PipelineType::Textured => self.image_pipeline.as_ref().unwrap(),
        }
    }

    /// The pipeline that draws `render_data`. With `translucent_last` (the scene draws blended
    /// surfaces after opaque ones), a blended surface draws without depth writes.
    pub fn pipeline_for(
        &self,
        render_data: &crate::core::RenderData,
        translucent_last: bool,
    ) -> &wgpu::RenderPipeline {
        if translucent_last && crate::core::scene::is_translucent_surface(render_data) {
            self.translucent_triangle_pipeline.as_ref().unwrap()
        } else {
            self.get_pipeline(render_data.pipeline_type)
        }
    }

    pub fn ensure_grid_plane_pipeline(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) {
        if self.grid_plane_pipeline.is_none() {
            self.grid_plane_pipeline = Some(self.create_grid_plane_pipeline(emission));
        }
    }

    pub fn grid_plane_pipeline(&self) -> Option<&wgpu::RenderPipeline> {
        self.grid_plane_pipeline.as_ref()
    }
}

/// Utility functions for creating common vertex patterns
pub mod vertex_utils;
