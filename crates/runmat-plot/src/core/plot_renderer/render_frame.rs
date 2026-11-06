//! Rendering a frame: the whole figure, a scene to a target, or one camera to a viewport.

use super::*;

impl PlotRenderer {
    /// Render the current scene to a texture/surface
    pub fn render(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        encoder: &mut wgpu::CommandEncoder,
        target: RenderTarget<'_>,
        config: &PlotRenderConfig,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();

        self.wgpu_renderer.ensure_msaa(config.msaa_samples);

        // Update WGPU uniforms from primary axes camera
        let aspect_ratio = config.width as f32 / config.height as f32;
        let mut cam = self.camera().clone();
        cam.update_aspect_ratio(aspect_ratio);
        let view_proj_matrix = cam.view_proj_matrix();
        self.wgpu_renderer
            .configure_cad_lighting(
                cam.view_matrix(),
                self.last_geometry_scene.is_some(),
                !self.theme_background_is_light(),
            );
        let model_matrix = self.axes_model_matrix(0);
        self.wgpu_renderer
            .update_uniforms(emission, view_proj_matrix, model_matrix);

        // Collect all render data and create vertex buffers first (outside render pass)
        let mut render_items = Vec::new();
        let mut total_vertices = 0;
        let mut total_triangles = 0;

        // Instanced nodes draw their placements in view at their level of detail.
        let mut instance_buffers: Vec<Option<(wgpu::Buffer, u32)>> = Vec::new();
        let viewport_px = glam::Vec2::new(config.width.max(1) as f32, config.height.max(1) as f32);
        self.wgpu_renderer.ensure_extra_pipelines(emission);
        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if let Some((vertex_buffer, index_buffer)) =
                    self.prepare_buffers_for_render_data(emission, node.id, render_data)
                {
                    self.wgpu_renderer
                        .ensure_pipeline(emission, render_data.pipeline_type);
                    instance_buffers.push(node.instances.as_ref().map(|set| {
                        let shown = set.visible(view_proj_matrix * model_matrix, viewport_px);
                        let count = shown.len() as u32;
                        (
                            self.wgpu_renderer.create_instance_buffer(emission, &shown),
                            count,
                        )
                    }));
                    render_items.push((render_data, vertex_buffer, index_buffer));

                    total_vertices += render_data.vertex_count();
                    if let Some(indices) = &render_data.indices {
                        total_triangles += indices.len() / 3;
                    }
                }
            }
        }

        // Pre-create image bind groups and set direct uniforms once (for textured items)
        let mut image_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());
        let has_textured_items = render_items.iter().any(|(render_data, _, _)| {
            render_data.pipeline_type == crate::core::PipelineType::Textured
        });
        if has_textured_items {
            // Ensure image pipeline once to avoid mutable borrow during pass.
            self.wgpu_renderer.ensure_image_pipeline(emission);
            let mut inferred_bounds: Option<(f64, f64, f64, f64)> = None;
            for (render_data, _, _) in &render_items {
                let Some(bounds) = render_data.bounds.as_ref() else {
                    continue;
                };
                let min_x = bounds.min.x as f64;
                let max_x = bounds.max.x as f64;
                let min_y = bounds.min.y as f64;
                let max_y = bounds.max.y as f64;
                inferred_bounds = Some(match inferred_bounds {
                    Some((x0, x1, y0, y1)) => {
                        (x0.min(min_x), x1.max(max_x), y0.min(min_y), y1.max(max_y))
                    }
                    None => (min_x, max_x, min_y, max_y),
                });
            }

            let (mut x_min, mut x_max, mut y_min, mut y_max) = self
                .data_bounds
                .or(inferred_bounds)
                .unwrap_or((-1.0, 1.0, -1.0, 1.0));
            // Avoid zero ranges in the direct image shader (division by data_range).
            if (x_max - x_min).abs() < f64::EPSILON {
                x_min -= 0.5;
                x_max += 0.5;
            }
            if (y_max - y_min).abs() < f64::EPSILON {
                y_min -= 0.5;
                y_max += 0.5;
            }
            log::trace!(
                target: "runmat_plot",
                "direct uniforms bounds x=({}, {}) y=({}, {}) size=({}, {})",
                x_min,
                x_max,
                y_min,
                y_max,
                config.width,
                config.height
            );
            let (ndc_y_min, ndc_y_max) = self.overlay_y_direction_for_axes(0).ndc_endpoints();
            self.wgpu_renderer.update_direct_uniforms(
                emission,
                DirectUniforms::new(
                    [x_min as f32, y_min as f32],
                    [x_max as f32, y_max as f32],
                    [-1.0, ndc_y_min],
                    [1.0, ndc_y_max],
                    [config.width as f32, config.height as f32],
                    [0, 0],
                ),
            );
        }
        for (render_data, _vb, _ib) in &render_items {
            if render_data.pipeline_type == crate::core::PipelineType::Textured {
                if let Some(crate::core::scene::ImageData::Rgba8 {
                    width,
                    height,
                    data,
                }) = &render_data.image
                {
                    let (_tex, _view, img_bg) = self
                        .wgpu_renderer
                        .create_image_texture_and_bind_group(emission, *width, *height, data);
                    image_bind_groups.push(Some(img_bg));
                } else {
                    image_bind_groups.push(None);
                }
            } else {
                image_bind_groups.push(None);
            }
        }
        let mut point_style_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in &render_items {
            if matches!(
                render_data.pipeline_type,
                crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
            ) {
                let style = crate::core::renderer::PointStyleUniforms {
                    face_color: render_data.material.albedo.to_array(),
                    edge_color: render_data.material.emissive.to_array(),
                    edge_thickness_px: render_data.material.roughness,
                    marker_shape: render_data.material.metallic as u32,
                    _pad: [0.0, 0.0],
                };
                let (_buf, bg) = self
                    .wgpu_renderer
                    .create_point_style_bind_group(emission, style);
                point_style_bind_groups.push(Some(bg));
            } else {
                point_style_bind_groups.push(None);
            }
        }
        // Expand CPU marker vertices to billboard quads.
        let mut point_buffers: Vec<Option<(wgpu::Buffer, usize)>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in &render_items {
            if matches!(
                render_data.pipeline_type,
                crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
            ) && !render_data.vertices.is_empty()
            {
                let expanded = self
                    .wgpu_renderer
                    .create_direct_point_vertices(&render_data.vertices, 0.0);
                let buffer = self.wgpu_renderer.create_vertex_buffer(emission, &expanded);
                point_buffers.push(Some((buffer, expanded.len())));
            } else {
                point_buffers.push(None);
            }
        }
        self.wgpu_renderer.update_marker_screen_uniforms(
            emission,
            [config.width.max(1) as f32, config.height.max(1) as f32],
        );

        // Create render pass
        {
            let depth_view = self.wgpu_renderer.ensure_depth_view(emission);
            let use_msaa = self.wgpu_renderer.msaa_sample_count > 1;
            let mut cached_msaa_view: Option<Arc<wgpu::TextureView>> = None;

            let (color_view, resolve_target) = if use_msaa {
                if let Some(explicit_resolve_target) = target.resolve_target {
                    (target.view, Some(explicit_resolve_target))
                } else {
                    cached_msaa_view = Some(self.wgpu_renderer.ensure_msaa_color_view(emission));
                    (
                        cached_msaa_view
                            .as_ref()
                            .expect("msaa color view should exist")
                            .as_ref(),
                        Some(target.view),
                    )
                }
            } else {
                (target.view, target.resolve_target)
            };

            let depth_clear = match self.wgpu_renderer.depth_mode {
                crate::core::DepthMode::Standard => 1.0,
                crate::core::DepthMode::ReversedZ => 0.0,
            };
            // Depth-tested edge strokes draw in a second pass that reads the scene's depth
            // (see `edge_stroke::EdgeStrokeShader::depth_probe`).
            let depth_probe = render_items
                .iter()
                .any(|(d, _, _)| d.pipeline_type == crate::core::PipelineType::EdgeStroke)
                .then(|| {
                    self.wgpu_renderer
                        .depth_probe_bind_group(emission, &depth_view)
                })
                .flatten();
            let _keep_msaa_view_alive = &cached_msaa_view;
            for stroke_pass in [false, true] {
                if stroke_pass && depth_probe.is_none() {
                    break;
                }
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(if stroke_pass {
                        "Plot Edge Stroke Pass"
                    } else {
                        "Plot Render Pass"
                    }),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target,
                        ops: wgpu::Operations {
                            load: if stroke_pass {
                                wgpu::LoadOp::Load
                            } else {
                                wgpu::LoadOp::Clear(wgpu::Color {
                                    r: config.background_color.x as f64,
                                    g: config.background_color.y as f64,
                                    b: config.background_color.z as f64,
                                    a: config.background_color.w as f64,
                                })
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        // Read-only in the stroke pass, which samples it.
                        depth_ops: (!stroke_pass).then_some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(depth_clear),
                            store: if depth_probe.is_some() {
                                wgpu::StoreOp::Store
                            } else {
                                wgpu::StoreOp::Discard
                            },
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                // Now render all items with proper bind group setup
                for (i, (render_data, vertex_buffer, index_buffer)) in
                    render_items.iter().enumerate()
                {
                    let probed_stroke = depth_probe.is_some()
                        && render_data.pipeline_type == crate::core::PipelineType::EdgeStroke;
                    if probed_stroke != stroke_pass {
                        continue;
                    }
                    let instanced = instance_buffers[i].as_ref();
                    if instanced.is_some_and(|(_, count)| *count == 0) {
                        continue;
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        // On wasm, "blank but drawing" is often caused by bad vertex data (NaNs/alpha=0)
                        // or using the wrong pipeline. Emit a single summary per item.
                        if log::log_enabled!(log::Level::Debug) {
                            if let Some(v0) = render_data.vertices.first() {
                                log::debug!(
                                    target: "runmat_plot",
                                    "wasm draw item: pipeline={:?} verts={} v0.pos=({:.3},{:.3},{:.3}) v0.color=({:.3},{:.3},{:.3},{:.3})",
                                    render_data.pipeline_type,
                                    render_data.vertices.len(),
                                    v0.position[0],
                                    v0.position[1],
                                    v0.position[2],
                                    v0.color[0],
                                    v0.color[1],
                                    v0.color[2],
                                    v0.color[3],
                                );
                            } else if render_data.gpu_vertices.is_some() {
                                log::debug!(
                                    target: "runmat_plot",
                                    "wasm draw item: pipeline={:?} using gpu_vertices vertex_count={}",
                                    render_data.pipeline_type,
                                    render_data.vertex_count(),
                                );
                            } else {
                                log::debug!(
                                    target: "runmat_plot",
                                    "wasm draw item: pipeline={:?} has no vertices",
                                    render_data.pipeline_type
                                );
                            }
                        }
                    }

                    // Get the appropriate pipeline for this render data (pipeline ensured above)
                    if render_data.pipeline_type == crate::core::PipelineType::Textured {
                        // Ensure image pipeline
                        let pipeline = self.wgpu_renderer.get_pipeline(render_data.pipeline_type);
                        render_pass.set_pipeline(pipeline);
                        // Bind direct uniforms at set(0)
                        // Use data bounds for image mapping
                        render_pass.set_bind_group(
                            0,
                            &self.wgpu_renderer.direct_uniform_bind_group,
                            &[],
                        );
                        if let Some(ref img_bg) = image_bind_groups[i] {
                            render_pass.set_bind_group(1, img_bg, &[]);
                        }
                    } else {
                        let variant = match render_data.pipeline_type {
                            crate::core::PipelineType::Triangles if instanced.is_some() => {
                                self.wgpu_renderer.instanced_triangle_pipeline()
                            }
                            crate::core::PipelineType::EdgeStroke
                                if instanced.is_some() || probed_stroke =>
                            {
                                self.wgpu_renderer
                                    .edge_stroke_variant(instanced.is_some(), probed_stroke)
                            }
                            _ => None,
                        };
                        let pipeline = variant.unwrap_or_else(|| {
                            self.wgpu_renderer.get_pipeline(render_data.pipeline_type)
                        });
                        render_pass.set_pipeline(pipeline);
                        // Set the uniform bind group (required by shaders)
                        render_pass.set_bind_group(
                            0,
                            self.wgpu_renderer.get_uniform_bind_group(),
                            &[],
                        );
                        if probed_stroke {
                            if let Some(depth) = depth_probe.as_ref() {
                                render_pass.set_bind_group(2, depth, &[]);
                            }
                        }
                        if matches!(
                            render_data.pipeline_type,
                            crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
                        ) {
                            if let Some(ref bg) = point_style_bind_groups[i] {
                                render_pass.set_bind_group(1, bg, &[]);
                            }
                            render_pass.set_bind_group(
                                2,
                                self.wgpu_renderer.get_marker_screen_bind_group(),
                                &[],
                            );
                        } else if render_data.pipeline_type.binds_screen_uniforms() {
                            render_pass.set_bind_group(
                                1,
                                self.wgpu_renderer.get_marker_screen_bind_group(),
                                &[],
                            );
                        }
                    }

                    let is_markers = matches!(
                        render_data.pipeline_type,
                        crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
                    );
                    if is_markers {
                        if let Some((ref expanded, _len)) = point_buffers[i] {
                            render_pass.set_vertex_buffer(0, expanded.slice(..));
                        } else {
                            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        }
                    } else {
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    }
                    if let Some((buffer, _)) = instanced {
                        render_pass.set_vertex_buffer(1, buffer.slice(..));
                    }
                    let placements = instanced.map(|(_, count)| 0..*count);

                    if let Some(index_buffer) = index_buffer {
                        render_pass
                            .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        if let Some(indices) = &render_data.indices {
                            for (index_range, instance_range) in
                                indexed_draw_ranges(&render_data.draw_calls, indices.len())
                            {
                                log::trace!(target: "runmat_plot", "draw indexed count={}", index_range.len());
                                render_pass.draw_indexed(
                                    index_range,
                                    0,
                                    placements.clone().unwrap_or(instance_range),
                                );
                            }
                        }
                    } else {
                        log::trace!(target: "runmat_plot", "draw direct vertices");
                        if let Some((args, offset)) = Self::gpu_indirect_args(render_data) {
                            render_pass.draw_indirect(args, offset);
                            continue;
                        }
                        if is_markers {
                            if let Some((_, len)) = point_buffers[i] {
                                render_pass.draw(0..len as u32, 0..1);
                                continue;
                            }
                        }
                        // Use draw_calls from render_data for proper vertex range handling
                        for draw_call in &render_data.draw_calls {
                            log::trace!(
                                target: "runmat_plot",
                                "draw vertices offset={} count={} instances={}",
                                draw_call.vertex_offset,
                                draw_call.vertex_count,
                                draw_call.instance_count
                            );
                            render_pass.draw(
                                draw_call.vertex_offset as u32
                                    ..(draw_call.vertex_offset + draw_call.vertex_count) as u32,
                                placements
                                    .clone()
                                    .unwrap_or(0..draw_call.instance_count as u32),
                            );
                        }
                    }
                }
                // Drop render_pass at end of scope.
            }
        }

        let render_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            axes_viewports_px: vec![(0, 0, config.width.max(1), config.height.max(1))],
            vertex_count: total_vertices,
            triangle_count: total_triangles,
            render_time_ms: render_time,
        })
    }

    /// Shared scene orchestration for non-overlay render targets.
    ///
    /// For single-axes figures this follows the direct full-target render path.
    /// For subplot grids, it renders each axes into a deterministic tiled viewport layout.
    pub fn render_scene_to_target(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        config: &PlotRenderConfig,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let (rows, cols) = self.figure_axes_grid();
        let axes_count = self.figure_axes_count();
        log::debug!(
            "runmat-plot: renderer.scene_to_target.start rows={} cols={} axes_count={} width={} height={}",
            rows,
            cols,
            axes_count,
            config.width,
            config.height
        );
        if axes_count <= 1
            && !self.axes_position_explicit_for_axes(0)
            && self.axes_grid_cell_for_axes(0).is_none()
        {
            log::debug!("runmat-plot: renderer.scene_to_target.branch_single_axes");
            return self.render(
                emission,
                encoder,
                RenderTarget {
                    view: target_view,
                    resolve_target: None,
                },
                config,
            );
        }

        let viewports = self.compute_axes_viewports_px(config.width.max(1), config.height.max(1));
        log::debug!(
            "runmat-plot: renderer.scene_to_target.branch_subplot_axes viewports={}",
            viewports.len()
        );
        self.render_axes_to_viewports(
            emission,
            encoder,
            target_view,
            &viewports,
            config.msaa_samples.max(1),
            config,
        )?;
        let stats = self.scene.statistics();
        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            axes_viewports_px: viewports,
            vertex_count: stats.total_vertices,
            triangle_count: stats.total_triangles,
            render_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub fn compute_axes_viewports_px(
        &self,
        total_width: u32,
        total_height: u32,
    ) -> Vec<(u32, u32, u32, u32)> {
        self.last_figure
            .as_ref()
            .map(|figure| {
                crate::core::axes_layout::figure_viewports_px(figure, total_width, total_height)
            })
            .unwrap_or_else(|| vec![(0, 0, total_width.max(1), total_height.max(1))])
    }

    /// Render using the camera-based pipeline into a viewport region with a scissor rectangle.
    /// This preserves existing contents (Load) and draws only inside the viewport rectangle.
    #[allow(clippy::too_many_arguments)]
    pub fn render_camera_to_viewport(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        viewport_scissor: (u32, u32, u32, u32),
        config: &PlotRenderConfig,
        camera: &Camera,
        axes_index: usize,
        clear_background: bool,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        log::debug!(
            "runmat-plot: renderer.camera_to_viewport.start axes_index={} viewport=({}, {}, {}, {}) clear_background={}",
            axes_index,
            viewport_scissor.0,
            viewport_scissor.1,
            viewport_scissor.2,
            viewport_scissor.3,
            clear_background
        );
        let use_msaa = config.msaa_samples.max(1) > 1;
        self.wgpu_renderer.ensure_msaa(config.msaa_samples);
        let msaa_view_keepalive = if use_msaa {
            Some(self.wgpu_renderer.ensure_msaa_color_view(emission))
        } else {
            None
        };
        let render_target = if let Some(msaa_view) = msaa_view_keepalive.as_ref() {
            RenderTarget {
                view: msaa_view.as_ref(),
                resolve_target: Some(target_view),
            }
        } else {
            RenderTarget {
                view: target_view,
                resolve_target: None,
            }
        };
        self.render_camera_to_target_viewport(
            emission,
            encoder,
            render_target,
            viewport_scissor,
            config,
            camera,
            axes_index,
            clear_background,
        )
    }
}
