//! The render pipelines for screen-space geometry edge strokes.
use super::{shaders, InstanceRaw, Vertex, WgpuRenderer};
use crate::core::DepthMode;
use crate::gpu::shaders::vertex::edge_stroke::EdgeStrokeShader;

impl WgpuRenderer {
    /// Edge strokes are alpha-blended triangles widened in the vertex shader. The depth-tested
    /// variant moves each stroke toward the camera by a fixed share of its view depth (the
    /// shader's `EDGE_DEPTH_SHARE`), so an edge drawn on its own face wins the depth test at any
    /// distance while an edge a few millimetres under another surface stays hidden; a small
    /// slope bias covers faces seen edge-on. It doesn't write depth, so crossing edges and the
    /// translucent faces drawn after them composite normally.
    pub(super) fn create_edge_stroke_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        depth_tested: bool,
    ) -> wgpu::RenderPipeline {
        self.create_edge_stroke_pipeline_variant(
            emission,
            EdgeStrokeShader {
                depth_tested,
                reversed_z: self.depth_mode == DepthMode::ReversedZ,
                instanced: false,
                depth_probe: None,
            },
            None,
        )
    }

    /// Any edge stroke variant: instanced strokes read a second (instance) vertex buffer, and
    /// depth-probed strokes bind the scene depth at group 2 (`probe_layout`).
    pub(super) fn create_edge_stroke_pipeline_variant(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        variant: EdgeStrokeShader,
        probe_layout: Option<&wgpu::BindGroupLayout>,
    ) -> wgpu::RenderPipeline {
        let depth_tested = variant.depth_tested;
        let label = if depth_tested {
            "Geometry Edge Stroke Pipeline"
        } else {
            "Geometry Edge Stroke No Depth Pipeline"
        };
        let shader = emission
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(
                    shaders::vertex::edge_stroke::source_for(variant).into(),
                ),
            });
        let mut layouts = vec![
            &self.uniform_bind_group_layout,
            &self.marker_screen_bind_group_layout,
        ];
        if let Some(probe) = probe_layout {
            layouts.push(probe);
        }
        let layout = emission
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &layouts,
                push_constant_ranges: &[],
            });
        let buffers = if variant.instanced {
            vec![Vertex::desc(), InstanceRaw::desc()]
        } else {
            vec![Vertex::desc()]
        };
        let toward_camera = match self.depth_mode {
            DepthMode::Standard => -1,
            DepthMode::ReversedZ => 1,
        };
        let (depth_compare, bias) = if depth_tested {
            (
                self.depth_compare(),
                wgpu::DepthBiasState {
                    constant: 0,
                    slope_scale: toward_camera as f32 * 1.0,
                    clamp: 0.0,
                },
            )
        } else {
            (
                wgpu::CompareFunction::Always,
                wgpu::DepthBiasState::default(),
            )
        };
        emission.device().create_render_pipeline(&crate::wgpu_compat::wgpu_render_pipeline_descriptor! {
            label: Some(label),
            layout: Some(&layout),
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
                depth_write_enabled: false,
                depth_compare,
                stencil: wgpu::StencilState::default(),
                bias,
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
