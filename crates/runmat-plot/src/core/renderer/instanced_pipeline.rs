//! Pipelines for instanced CAD prototypes and for depth-probed edge strokes, and the per-frame
//! resources they bind: instance buffers and the scene-depth bind group.
use super::{shaders, InstanceRaw, Vertex, WgpuRenderer};
use crate::core::DepthMode;
use crate::gpu::shaders::vertex::edge_stroke::EdgeStrokeShader;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// The extra pipelines, built on first use and dropped with the others when the depth mode or
/// MSAA count changes.
#[derive(Default)]
pub(crate) struct ExtraPipelines {
    triangles_instanced: Option<wgpu::RenderPipeline>,
    /// Depth-tested edge strokes keyed by (instanced, depth probed).
    strokes: HashMap<(bool, bool), wgpu::RenderPipeline>,
    /// The scene-depth layout for probed strokes, for the MSAA state it was made for.
    depth_layout: Option<(bool, wgpu::BindGroupLayout)>,
}

impl WgpuRenderer {
    fn multisampled(&self) -> bool {
        self.msaa_sample_count > 1
    }

    fn ensure_depth_probe_layout(&mut self, emission: &runmat_accelerate_api::WgpuEmission<'_>) {
        let multisampled = self.multisampled();
        if self
            .extra_pipelines
            .depth_layout
            .as_ref()
            .is_some_and(|(m, _)| *m == multisampled)
        {
            return;
        }
        let layout = emission
            .device()
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Edge Stroke Scene Depth"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                }],
            });
        self.extra_pipelines.depth_layout = Some((multisampled, layout));
        self.extra_pipelines
            .strokes
            .retain(|(_, probed), _| !probed);
    }

    /// Builds the instanced triangle pipeline and the depth-tested stroke variants.
    pub fn ensure_extra_pipelines(&mut self, emission: &runmat_accelerate_api::WgpuEmission<'_>) {
        self.ensure_depth_probe_layout(emission);
        if self.extra_pipelines.triangles_instanced.is_none() {
            let pipeline = self.create_triangle_pipeline_variant(emission, true, true);
            self.extra_pipelines.triangles_instanced = Some(pipeline);
        }
        for instanced in [false, true] {
            for probed in [false, true] {
                if self
                    .extra_pipelines
                    .strokes
                    .contains_key(&(instanced, probed))
                {
                    continue;
                }
                let variant = EdgeStrokeShader {
                    depth_tested: true,
                    reversed_z: self.depth_mode == DepthMode::ReversedZ,
                    instanced,
                    depth_probe: probed.then_some(self.multisampled()),
                };
                let layout = self.extra_pipelines.depth_layout.as_ref().map(|(_, l)| l);
                let pipeline = self.create_edge_stroke_pipeline_variant(
                    emission,
                    variant,
                    if probed { layout } else { None },
                );
                self.extra_pipelines
                    .strokes
                    .insert((instanced, probed), pipeline);
            }
        }
    }

    pub fn instanced_triangle_pipeline(&self) -> Option<&wgpu::RenderPipeline> {
        self.extra_pipelines.triangles_instanced.as_ref()
    }

    /// A depth-tested edge stroke pipeline: `instanced` reads placements, `probed` samples the
    /// scene depth at group 2.
    pub fn edge_stroke_variant(
        &self,
        instanced: bool,
        probed: bool,
    ) -> Option<&wgpu::RenderPipeline> {
        self.extra_pipelines.strokes.get(&(instanced, probed))
    }

    /// The scene depth as a probed stroke's group 2. The depth attachment must be read-only in
    /// the pass that binds it.
    pub fn depth_probe_bind_group(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        depth_view: &wgpu::TextureView,
    ) -> Option<wgpu::BindGroup> {
        let (_, layout) = self.extra_pipelines.depth_layout.as_ref()?;
        Some(
            emission
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Edge Stroke Scene Depth"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    }],
                }),
        )
    }

    pub fn create_instance_buffer(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        instances: &[InstanceRaw],
    ) -> wgpu::Buffer {
        emission
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(instances),
                usage: wgpu::BufferUsages::VERTEX,
            })
    }

    /// The triangle pipeline, plain or instanced (a second, per-instance vertex buffer).
    pub(super) fn create_triangle_pipeline_variant(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        instanced: bool,
        depth_write_enabled: bool,
    ) -> wgpu::RenderPipeline {
        let source = if instanced {
            shaders::vertex::triangle::instanced()
        } else {
            shaders::vertex::TRIANGLE.to_string()
        };
        let shader = emission
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Triangle Shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline_layout =
            emission
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Triangle Pipeline Layout"),
                    bind_group_layouts: &[&self.uniform_bind_group_layout],
                    push_constant_ranges: &[],
                });
        let buffers = if instanced {
            vec![Vertex::desc(), InstanceRaw::desc()]
        } else {
            vec![Vertex::desc()]
        };
        emission.device().create_render_pipeline(&crate::wgpu_compat::wgpu_render_pipeline_descriptor! {
            label: Some("Triangle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: crate::wgpu_compat::wgpu_vertex_state!(&shader, "vs_main", &buffers),
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
                depth_write_enabled,
                depth_compare: self.depth_compare(),
                stencil: wgpu::StencilState::default(),
                // Translucent surfaces sit a hair behind coincident opaque ones: where context
                // geometry touches a result face, the result shows untinted instead of
                // z-fighting with it.
                bias: if depth_write_enabled {
                    wgpu::DepthBiasState::default()
                } else {
                    let away = match self.depth_mode {
                        DepthMode::Standard => 1,
                        DepthMode::ReversedZ => -1,
                    };
                    wgpu::DepthBiasState {
                        constant: 8 * away,
                        slope_scale: 2.0 * away as f32,
                        clamp: 0.0,
                    }
                },
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
