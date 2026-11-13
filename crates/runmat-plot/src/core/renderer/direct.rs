//! Render passes, vertex draws and the direct (viewport-mapped) pipelines' uniforms.

use super::*;

impl WgpuRenderer {
    /// Begin a render pass
    pub fn begin_render_pass<'a>(
        &'a self,
        _emission: &'a runmat_accelerate_api::WgpuEmission<'_>,
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
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        render_pass: &mut wgpu::RenderPass<'a>,
        pipeline_type: PipelineType,
        vertex_buffer: &'a wgpu::Buffer,
        draw: VertexDraw<'a>,
    ) {
        // Ensure the pipeline exists first
        self.ensure_pipeline(emission, pipeline_type);

        // Now get the pipeline and render
        let pipeline = self.get_pipeline(pipeline_type);
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

        match draw {
            VertexDraw::Indirect(indirect) => {
                render_pass.draw_indirect(indirect.args.as_ref(), indirect.offset);
            }
            VertexDraw::Indexed {
                index_buffer,
                index_count,
            } => {
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..index_count, 0, 0..1);
            }
            VertexDraw::Direct { vertex_count } => {
                render_pass.draw(0..vertex_count, 0..1);
            }
        }
    }

    /// Ensure direct line pipeline exists
    pub fn ensure_direct_line_pipeline(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) {
        if self.direct_line_pipeline.is_none() {
            self.direct_line_pipeline = Some(self.create_direct_line_pipeline(emission));
        }
    }

    /// Ensure direct triangle pipeline exists
    pub fn ensure_direct_triangle_pipeline(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) {
        if self.direct_triangle_pipeline.is_none() {
            self.direct_triangle_pipeline = Some(self.create_direct_triangle_pipeline(emission));
        }
    }

    /// Ensure direct point pipeline exists
    pub fn ensure_direct_point_pipeline(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) {
        if self.direct_point_pipeline.is_none() {
            self.direct_point_pipeline = Some(self.create_direct_point_pipeline(emission));
        }
    }

    /// Ensure image pipeline exists
    pub fn ensure_image_pipeline(&mut self, emission: &runmat_accelerate_api::WgpuEmission<'_>) {
        if self.image_pipeline.is_none() {
            self.image_pipeline = Some(self.create_image_pipeline(emission));
        }
    }

    /// Update transformation uniforms for direct viewport rendering
    pub fn update_direct_uniforms(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        uniforms: DirectUniforms,
    ) {
        self.direct_uniforms = uniforms;
        emission.queue().write_buffer(
            &self.direct_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.direct_uniforms]),
        );
        self.ensure_axes_uniform_capacity(emission, 1);
        emission.queue().write_buffer(
            &self.axes_direct_uniform_buffers[0],
            0,
            bytemuck::cast_slice(&[self.direct_uniforms]),
        );
    }

    pub fn update_direct_uniforms_for_axes(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        axes_index: usize,
        uniforms: DirectUniforms,
    ) {
        self.ensure_axes_uniform_capacity(emission, axes_index + 1);
        emission.queue().write_buffer(
            &self.axes_direct_uniform_buffers[axes_index],
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    pub fn get_direct_uniform_bind_group_for_axes(&self, axes_index: usize) -> &wgpu::BindGroup {
        self.axes_direct_uniform_bind_groups
            .get(axes_index)
            .unwrap_or(&self.direct_uniform_bind_group)
    }
}
