//! The render pipeline for screen-space geometry edge strokes.
use super::{shaders, Vertex, WgpuRenderer};
use crate::core::DepthMode;

impl WgpuRenderer {
    /// Edge strokes are alpha-blended triangles widened in the vertex shader. The depth-tested
    /// variant is biased toward the camera, so an edge drawn on its own face wins the depth test
    /// against that face and doesn't flicker; it doesn't write depth, so crossing edges and the
    /// translucent faces drawn after them composite normally.
    pub(super) fn create_edge_stroke_pipeline(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        depth_tested: bool,
    ) -> wgpu::RenderPipeline {
        let label = if depth_tested {
            "Geometry Edge Stroke Pipeline"
        } else {
            "Geometry Edge Stroke No Depth Pipeline"
        };
        let shader = emission
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(shaders::vertex::EDGE_STROKE.into()),
            });
        let layout = emission
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[
                    &self.uniform_bind_group_layout,
                    &self.marker_screen_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let toward_camera = match self.depth_mode {
            DepthMode::Standard => -1,
            DepthMode::ReversedZ => 1,
        };
        let (depth_compare, bias) = if depth_tested {
            (
                self.depth_compare(),
                wgpu::DepthBiasState {
                    constant: toward_camera * 64,
                    slope_scale: toward_camera as f32 * 1.5,
                    clamp: 0.0,
                },
            )
        } else {
            (wgpu::CompareFunction::Always, wgpu::DepthBiasState::default())
        };
        emission.device().create_render_pipeline(&crate::wgpu_compat::wgpu_render_pipeline_descriptor! {
            label: Some(label),
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
