//! The renderer's pipeline constructors, one per primitive kind.

use super::*;

impl WgpuRenderer {
    /// Create point rendering pipeline
    pub(super) fn create_point_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> wgpu::RenderPipeline {
        let shader = emission
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Point Billboard Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::POINT_BILLBOARD.into()),
            });

        let pipeline_layout =
            emission
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Point Pipeline Layout"),
                    bind_group_layouts: &[
                        &self.uniform_bind_group_layout,
                        &self.point_style_bind_group_layout,
                        &self.marker_screen_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        emission.device()
            .create_render_pipeline(&crate::wgpu_compat::wgpu_render_pipeline_descriptor! {
                label: Some("Point Billboard Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: crate::wgpu_compat::wgpu_vertex_state!(&shader, "vs_main", &[Vertex::desc()]),
                fragment: Some(crate::wgpu_compat::wgpu_fragment_state!(&shader, "fs_main", &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })])),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::depth_format(),
                    depth_write_enabled: true,
                    depth_compare: self.depth_compare(),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }

    /// Create line rendering pipeline
    pub(super) fn create_line_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> wgpu::RenderPipeline {
        self.create_camera_line_pipeline(emission, "Line Pipeline", true, self.depth_compare())
    }

    pub(super) fn create_line_no_depth_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> wgpu::RenderPipeline {
        self.create_camera_line_pipeline(
            emission,
            "Line No Depth Pipeline",
            false,
            wgpu::CompareFunction::Always,
        )
    }

    pub(super) fn create_annotation_stroke_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> wgpu::RenderPipeline {
        let shader = emission
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Geometry Annotation Stroke Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::ANNOTATION_STROKE.into()),
            });
        let layout = emission
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Geometry Annotation Stroke Pipeline Layout"),
                bind_group_layouts: &[
                    &self.uniform_bind_group_layout,
                    &self.marker_screen_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        emission.device().create_render_pipeline(&crate::wgpu_compat::wgpu_render_pipeline_descriptor! {
            label: Some("Geometry Annotation Stroke Pipeline"),
            layout: Some(&layout),
            vertex: crate::wgpu_compat::wgpu_vertex_state!(&shader, "vs_main", &[Vertex::desc()]),
            fragment: Some(crate::wgpu_compat::wgpu_fragment_state!(&shader, "fs_main", &[Some(wgpu::ColorTargetState {
                format: self.surface_config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })])),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::depth_format(),
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: self.msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        })
    }

    pub(super) fn create_camera_line_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        label: &'static str,
        depth_write_enabled: bool,
        depth_compare: wgpu::CompareFunction,
    ) -> wgpu::RenderPipeline {
        let shader = emission
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::LINE.into()),
            });

        let pipeline_layout =
            emission
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[&self.uniform_bind_group_layout],
                    push_constant_ranges: &[],
                });

        emission.device()
            .create_render_pipeline(&crate::wgpu_compat::wgpu_render_pipeline_descriptor! {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: crate::wgpu_compat::wgpu_vertex_state!(&shader, "vs_main", &[Vertex::desc()]),
                fragment: Some(crate::wgpu_compat::wgpu_fragment_state!(&shader, "fs_main", &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })])),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::depth_format(),
                    depth_write_enabled,
                    depth_compare,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }

    /// Create optimized direct rendering pipeline for precise viewport mapping
    pub(super) fn create_direct_line_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> wgpu::RenderPipeline {
        let shader = emission
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Direct Line Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::LINE_DIRECT.into()),
            });

        let pipeline_layout =
            emission
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Direct Line Pipeline Layout"),
                    bind_group_layouts: &[&self.direct_uniform_bind_group_layout],
                    push_constant_ranges: &[],
                });

        emission.device()
            .create_render_pipeline(&crate::wgpu_compat::wgpu_render_pipeline_descriptor! {
                label: Some("Direct Line Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: crate::wgpu_compat::wgpu_vertex_state!(&shader, "vs_main", &[Vertex::desc()]),
                fragment: Some(crate::wgpu_compat::wgpu_fragment_state!(&shader, "fs_main", &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })])),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                // This pipeline is used inside a render pass that has a depth attachment.
                // To be compatible with that pass, we must specify a matching depth format.
                // Use CompareFunction::Always + no writes to effectively disable depth testing.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::depth_format(),
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }

    /// Create optimized direct triangle pipeline (2D fills) for precise viewport mapping
    pub(super) fn create_direct_triangle_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> wgpu::RenderPipeline {
        let shader = emission
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Direct Triangle Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::LINE_DIRECT.into()),
            });

        let pipeline_layout =
            emission
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Direct Triangle Pipeline Layout"),
                    bind_group_layouts: &[&self.direct_uniform_bind_group_layout],
                    push_constant_ranges: &[],
                });

        emission.device()
            .create_render_pipeline(&crate::wgpu_compat::wgpu_render_pipeline_descriptor! {
                label: Some("Direct Triangle Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: crate::wgpu_compat::wgpu_vertex_state!(&shader, "vs_main", &[Vertex::desc()]),
                fragment: Some(crate::wgpu_compat::wgpu_fragment_state!(&shader, "fs_main", &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })])),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::depth_format(),
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }

    /// Create optimized direct point pipeline (instanced quads per point)
    pub(super) fn create_direct_point_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> wgpu::RenderPipeline {
        let shader = emission
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Direct Point Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::POINT_DIRECT.into()),
            });

        let pipeline_layout =
            emission
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Direct Point Pipeline Layout"),
                    bind_group_layouts: &[
                        &self.direct_uniform_bind_group_layout,
                        &self.point_style_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        emission.device()
            .create_render_pipeline(&crate::wgpu_compat::wgpu_render_pipeline_descriptor! {
                label: Some("Direct Point Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: crate::wgpu_compat::wgpu_vertex_state!(&shader, "vs_main", &[Vertex::desc()]),
                fragment: Some(crate::wgpu_compat::wgpu_fragment_state!(&shader, "fs_main", &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })])),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::depth_format(),
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
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
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        style: PointStyleUniforms,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = emission
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Point Style Uniform Buffer"),
                contents: bytemuck::bytes_of(&style),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = emission
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
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
    pub(super) fn create_image_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> wgpu::RenderPipeline {
        let shader = emission
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Image Direct Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::IMAGE_DIRECT.into()),
            });

        let pipeline_layout =
            emission
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Image Pipeline Layout"),
                    bind_group_layouts: &[
                        &self.direct_uniform_bind_group_layout,
                        &self.image_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        emission.device()
            .create_render_pipeline(&crate::wgpu_compat::wgpu_render_pipeline_descriptor! {
                label: Some("Image Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: crate::wgpu_compat::wgpu_vertex_state!(&shader, "vs_main", &[Vertex::desc()]),
                fragment: Some(crate::wgpu_compat::wgpu_fragment_state!(&shader, "fs_main", &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })])),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::depth_format(),
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }

    /// Create triangle rendering pipeline
    pub(super) fn create_triangle_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> wgpu::RenderPipeline {
        self.create_triangle_pipeline_variant(emission, false)
    }

    pub(super) fn create_grid_plane_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> wgpu::RenderPipeline {
        let shader = emission
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Grid Plane Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::GRID_PLANE.into()),
            });

        let pipeline_layout =
            emission
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Grid Plane Pipeline Layout"),
                    bind_group_layouts: &[
                        &self.uniform_bind_group_layout,
                        &self.grid_uniform_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        emission.device()
            .create_render_pipeline(&crate::wgpu_compat::wgpu_render_pipeline_descriptor! {
                label: Some("Grid Plane Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: crate::wgpu_compat::wgpu_vertex_state!(&shader, "vs_main", &[Vertex::desc()]),
                fragment: Some(crate::wgpu_compat::wgpu_fragment_state!(&shader, "fs_main", &[Some(wgpu::ColorTargetState {
                        format: self.surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })])),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::depth_format(),
                    depth_write_enabled: false,
                    depth_compare: self.depth_compare(),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: self.msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
    }
}
