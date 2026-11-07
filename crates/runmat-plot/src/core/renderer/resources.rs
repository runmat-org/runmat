//! Depth, MSAA, texture and buffer resources of the renderer.

use super::*;

impl WgpuRenderer {
    pub fn set_depth_mode(&mut self, mode: DepthMode) {
        if self.depth_mode != mode {
            self.depth_mode = mode;
            // Pipelines depend on depth compare; rebuild.
            self.point_pipeline = None;
            self.line_pipeline = None;
            self.line_no_depth_pipeline = None;
            self.annotation_stroke_pipeline = None;
            self.edge_stroke_pipeline = None;
            self.edge_stroke_no_depth_pipeline = None;
            self.triangle_pipeline = None;
            self.translucent_triangle_pipeline = None;
            self.extra_pipelines = Default::default();
            self.direct_line_pipeline = None;
            self.direct_triangle_pipeline = None;
            self.direct_point_pipeline = None;
            self.image_pipeline = None;
            self.grid_plane_pipeline = None;
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
            self.line_no_depth_pipeline = None;
            self.annotation_stroke_pipeline = None;
            self.edge_stroke_pipeline = None;
            self.edge_stroke_no_depth_pipeline = None;
            self.triangle_pipeline = None;
            self.translucent_triangle_pipeline = None;
            self.extra_pipelines = Default::default();
            self.direct_line_pipeline = None;
            self.direct_triangle_pipeline = None;
            self.direct_point_pipeline = None;
            self.image_pipeline = None;
            self.grid_plane_pipeline = None;
            // Depth attachment must match sample count.
            self.depth_texture = None;
            self.depth_view = None;
            self.depth_extent = (0, 0, 0);
            self.msaa_color_texture = None;
            self.msaa_color_view = None;
            self.msaa_color_extent = (0, 0, 0);
        }
    }

    /// 32-bit float depth on every target (WebGPU supports it everywhere). With reversed-Z its
    /// precision is relative to view distance, so a 3 mm skin over a frame still separates at
    /// 60 m; a 24-bit buffer lets edges under the skin show through on a large assembly.
    pub(crate) fn depth_format() -> wgpu::TextureFormat {
        wgpu::TextureFormat::Depth32Float
    }

    pub(super) fn depth_compare(&self) -> wgpu::CompareFunction {
        match self.depth_mode {
            DepthMode::Standard => wgpu::CompareFunction::LessEqual,
            DepthMode::ReversedZ => wgpu::CompareFunction::GreaterEqual,
        }
    }

    pub fn ensure_depth_view(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> Arc<wgpu::TextureView> {
        let width = self.surface_config.width.max(1);
        let height = self.surface_config.height.max(1);
        let samples = self.msaa_sample_count.max(1);
        if self.depth_view.is_none() || self.depth_extent != (width, height, samples) {
            let texture = emission.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("runmat_depth_texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format: Self::depth_format(),
                // Probed edge strokes read it (as a read-only attachment's texture).
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.depth_texture = Some(texture);
            self.depth_view = Some(Arc::new(view));
            self.depth_extent = (width, height, samples);
        }
        self.depth_view
            .as_ref()
            .cloned()
            .expect("depth view missing")
    }

    pub fn ensure_msaa_color_view(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
    ) -> Arc<wgpu::TextureView> {
        let width = self.surface_config.width.max(1);
        let height = self.surface_config.height.max(1);
        let samples = self.msaa_sample_count.max(1);
        if self.msaa_color_view.is_none() || self.msaa_color_extent != (width, height, samples) {
            let texture = emission.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("runmat_msaa_color_plot"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.msaa_color_texture = Some(texture);
            self.msaa_color_view = Some(Arc::new(view));
            self.msaa_color_extent = (width, height, samples);
        }
        self.msaa_color_view
            .as_ref()
            .cloned()
            .expect("msaa color view missing")
    }

    /// Create a GPU texture and bind group for an RGBA8 image
    pub fn create_image_texture_and_bind_group(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::BindGroup) {
        let texture = emission.device().create_texture(&wgpu::TextureDescriptor {
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
        emission.queue().write_texture(
            crate::wgpu_compat::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            crate::wgpu_compat::TexelCopyBufferLayout {
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
        let bind_group = emission
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
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
    pub fn create_vertex_buffer(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        vertices: &[Vertex],
    ) -> wgpu::Buffer {
        emission
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            })
    }

    /// Choose the most efficient vertex buffer source for the provided data.
    pub fn vertex_buffer_from_sources(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        gpu: Option<&GpuVertexBuffer>,
        cpu_vertices: &[Vertex],
    ) -> Option<runmat_accelerate_api::WgpuBufferOwner> {
        if let Some(buffer) = gpu {
            Some(buffer.buffer.clone())
        } else if !cpu_vertices.is_empty() {
            Some(runmat_accelerate_api::WgpuBufferOwner::new(
                self.create_vertex_buffer(emission, cpu_vertices),
            ))
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
    pub fn create_index_buffer(
        &self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        indices: &[u32],
    ) -> wgpu::Buffer {
        emission
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            })
    }
}
