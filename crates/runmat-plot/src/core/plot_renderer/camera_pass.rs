//! One camera's render pass into a target viewport: render items, helpers, grid and draws.

use super::*;

impl PlotRenderer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_camera_to_target_viewport(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        encoder: &mut wgpu::CommandEncoder,
        target: RenderTarget<'_>,
        viewport_scissor: (u32, u32, u32, u32),
        config: &PlotRenderConfig,
        camera: &Camera,
        axes_index: usize,
        clear_background: bool,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();

        // Apply MSAA preference into pipelines
        self.wgpu_renderer.ensure_msaa(config.msaa_samples);
        self.wgpu_renderer.set_depth_mode(config.depth_mode);

        // Ensure a depth attachment exists for camera-based 3D rendering.
        // This is a no-op for pure 2D direct-mapped pipelines, but is required for correct
        // occlusion in 3D plots (surf/mesh/scatter3).
        let depth_view = self.wgpu_renderer.ensure_depth_view(emission);

        // Update standard uniforms from the provided camera
        let aspect_ratio = (config.width.max(1)) as f32 / (config.height.max(1)) as f32;
        let mut cam = camera.clone();
        cam.update_aspect_ratio(aspect_ratio);
        cam.depth_mode = config.depth_mode;
        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.camera_ready axes_index={} aspect_ratio={} msaa_samples={}",
            axes_index,
            aspect_ratio,
            config.msaa_samples
        );

        // Dynamic clip planes (CAD-like): keep near/far tight to visible bounds to avoid
        // clipping surprises and depth precision collapse on huge datasets.
        if config.clip_policy.dynamic {
            let mut bounds: Option<crate::core::scene::BoundingBox> = None;
            for node in self.scene.get_visible_nodes() {
                if let Some(rd) = &node.render_data {
                    if let Some(b) = rd.bounds {
                        bounds = Some(bounds.map_or(b, |acc| acc.union(&b)));
                    }
                }
            }
            if let Some(b) = bounds {
                cam.update_clip_planes_from_world_aabb(b.min, b.max, &config.clip_policy);
            }
        }
        let view_proj_matrix = cam.view_proj_matrix();
        self.wgpu_renderer
            .configure_cad_lighting(
                cam.view_matrix(),
                self.last_geometry_scene.is_some(),
                !self.theme_background_is_light(),
            );
        let model_matrix = self.axes_model_matrix(axes_index);
        self.wgpu_renderer.update_uniforms_for_axes(
            emission,
            axes_index,
            view_proj_matrix,
            model_matrix,
        );
        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.uniforms_updated axes_index={}",
            axes_index
        );

        let (mut sx, mut sy, mut sw, mut sh) = viewport_scissor;
        let target_w = self.wgpu_renderer.surface_config.width.max(1);
        let target_h = self.wgpu_renderer.surface_config.height.max(1);
        if sx >= target_w || sy >= target_h {
            return Ok(RenderResult {
                success: true,
                data_bounds: self.data_bounds,
                axes_viewports_px: vec![(0, 0, target_w, target_h)],
                vertex_count: 0,
                triangle_count: 0,
                render_time_ms: 0.0,
            });
        }
        sx = sx.min(target_w.saturating_sub(1));
        sy = sy.min(target_h.saturating_sub(1));
        sw = sw.max(1).min(target_w.saturating_sub(sx).max(1));
        sh = sh.max(1).min(target_h.saturating_sub(sy).max(1));
        let is_2d = matches!(
            cam.projection,
            crate::core::camera::ProjectionType::Orthographic { .. }
        );
        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.viewport_normalized axes_index={} viewport=({}, {}, {}, {}) is_2d={}",
            axes_index,
            sx,
            sy,
            sw,
            sh,
            is_2d
        );
        log::debug!(
            target: "runmat_plot.draw_camera",
            "draw camera axes_index={axes_index} is_2d={is_2d} viewport=({sx}, {sy}, {sw}, {sh}) projection={:?} cfg_wh=({}, {})",
            cam.projection,
            config.width,
            config.height
        );

        // 3D helpers: a horizontal world grid and world-axis ticks. Keeping the
        // grid on XY gives it a stable meaning as the camera moves. `axis off` hides them all;
        // a plot keeps its axis lines under `grid off`, while CAD geometry's world triad follows
        // the grid toggle. Built before the scene's render items borrow the scene.
        let axes_visible = self
            .overlay_axes_style_for_axes(axes_index)
            .is_none_or(|style| style.visible);
        let world_helpers = (!is_2d
            && axes_visible
            && (self.overlay_show_grid_for_axes(axes_index)
                || !self.axes_shows_geometry(axes_index)))
        .then(|| {
            self.world_helpers_3d(
                emission,
                &cam,
                axes_index,
                view_proj_matrix,
                model_matrix,
                (sw, sh),
            )
        });

        // The 2D grid draws under the data (2D orthographic only). Built before the scene's
        // render items borrow the scene.
        let log_flags = self.direct_log_flags_for_axes(axes_index);
        let grid_vb_opt = if is_2d {
            self.grid_2d_buffer(emission, axes_index, (sw, sh), log_flags)
        } else {
            None
        };

        // Prepare render items outside the pass
        let mut owned_render_data: Vec<Box<crate::core::RenderData>> = Vec::new();
        let mut render_items = Vec::new();
        let mut instances_by_item: std::collections::HashMap<
            *const crate::core::RenderData,
            std::sync::Arc<crate::core::NodeInstances>,
        > = Default::default();
        self.wgpu_renderer.ensure_extra_pipelines(emission);
        let mut grid_plane_buffers: Option<(wgpu::Buffer, wgpu::Buffer)> = None;
        let mut total_vertices = 0usize;
        let mut total_triangles = 0usize;
        let render_item_logs = render_item_diagnostics_enabled();
        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.collect_render_items.start axes_index={}",
            axes_index
        );
        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if render_item_logs && node.axes_index == axes_index {
                    log::debug!(
                        target: "runmat_plot.draw_item",
                        "draw item axes_index={} node_axes_index={} pipeline={:?} vertex_count={} has_indices={} has_bounds={} gpu_vertices={}",
                        axes_index,
                        node.axes_index,
                        render_data.pipeline_type,
                        render_data.vertex_count(),
                        render_data.indices.is_some(),
                        render_data.bounds.is_some(),
                        render_data.gpu_vertices.is_some()
                    );
                }
                if let Some((vb, ib)) =
                    self.prepare_buffers_for_render_data(emission, node.id, render_data)
                {
                    self.wgpu_renderer
                        .ensure_pipeline(emission, render_data.pipeline_type);
                    total_vertices += render_data.vertex_count();
                    if let Some(indices) = &render_data.indices {
                        total_triangles += indices.len() / 3;
                    }
                    if let Some(set) = node.instances.as_ref() {
                        instances_by_item
                            .insert(render_data as *const crate::core::RenderData, set.clone());
                    }
                    render_items.push((render_data, vb, ib));
                }
            }
        }
        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.collect_render_items.ok axes_index={} items={} total_vertices={} total_triangles={}",
            axes_index,
            render_items.len(),
            total_vertices,
            total_triangles
        );

        if let Some((helper_vertices, plane)) = world_helpers {
            grid_plane_buffers = plane;
            if !helper_vertices.is_empty() {
                let rd = Box::new(crate::core::RenderData {
                    pipeline_type: crate::core::PipelineType::Lines,
                    vertices: helper_vertices,
                    indices: None,
                    gpu_vertices: None,
                    bounds: None,
                    material: crate::core::Material::default(),
                    draw_calls: vec![crate::core::DrawCall {
                        vertex_offset: 0,
                        vertex_count: 0, // filled below
                        index_offset: None,
                        index_count: None,
                        instance_count: 1,
                    }],
                    image: None,
                });
                owned_render_data.push(rd);
                let idx = owned_render_data.len() - 1;
                // Fill vertex_count now that vertices are owned.
                let vcount = owned_render_data[idx].vertices.len();
                if let Some(dc) = owned_render_data[idx].draw_calls.get_mut(0) {
                    dc.vertex_count = vcount;
                }
                let vb = runmat_accelerate_api::WgpuBufferOwner::new(
                    self.wgpu_renderer
                        .create_vertex_buffer(emission, &owned_render_data[idx].vertices),
                );
                // Draw helpers first (under data, depth-tested).
                let rd_ref: &crate::core::RenderData = &owned_render_data[idx];
                render_items.insert(0, (rd_ref, vb, None));
                total_vertices += vcount;
            }
        }

        // Instanced nodes draw their placements in view at their level of detail.
        let instance_buffers: Vec<Option<(wgpu::Buffer, u32)>> = render_items
            .iter()
            .map(|(render_data, _, _)| {
                instances_by_item
                    .get(&(*render_data as *const crate::core::RenderData))
                    .map(|set| {
                        let shown = set.visible(
                            view_proj_matrix * model_matrix,
                            glam::Vec2::new(sw.max(1) as f32, sh.max(1) as f32),
                        );
                        let count = shown.len() as u32;
                        (
                            self.wgpu_renderer.create_instance_buffer(emission, &shown),
                            count,
                        )
                    })
            })
            .collect();
        // Precompute expanded point buffers to keep them alive across the render pass
        let mut point_buffers: Vec<Option<(wgpu::Buffer, usize)>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in render_items.iter() {
            if matches!(
                render_data.pipeline_type,
                crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
            ) && !render_data.vertices.is_empty()
            {
                let expanded = self
                    .wgpu_renderer
                    // size_px=0.0 => use per-vertex normal.z sizes
                    .create_direct_point_vertices(&render_data.vertices, 0.0);
                let buf = self.wgpu_renderer.create_vertex_buffer(emission, &expanded);
                point_buffers.push(Some((buf, expanded.len())));
            } else {
                point_buffers.push(None);
            }
        }
        // Precreate image bind groups for textured items to avoid lifetime issues
        let has_textured_items = render_items.iter().any(|(render_data, _vb, _ib)| {
            render_data.pipeline_type == crate::core::PipelineType::Textured
        });
        if has_textured_items {
            self.wgpu_renderer.ensure_image_pipeline(emission);
        }
        let mut image_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());

        for (render_data, _vb, _ib) in render_items.iter() {
            if render_data.pipeline_type == crate::core::PipelineType::Textured {
                if let Some(crate::core::scene::ImageData::Rgba8 {
                    width,
                    height,
                    data,
                }) = &render_data.image
                {
                    let (_t, _v, bg) = self
                        .wgpu_renderer
                        .create_image_texture_and_bind_group(emission, *width, *height, data);
                    image_bind_groups.push(Some(bg));
                } else {
                    image_bind_groups.push(None);
                }
            } else {
                image_bind_groups.push(None);
            }
        }
        // Precreate point style bind groups for points to match pipeline layout [direct uniforms, point style]
        let mut point_style_bind_groups: Vec<Option<wgpu::BindGroup>> =
            Vec::with_capacity(render_items.len());
        for (render_data, _vb, _ib) in render_items.iter() {
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

        // Before the pass: configure direct uniforms and ensure pipelines
        let bounds_opt = if is_2d {
            match cam.projection {
                crate::core::camera::ProjectionType::Orthographic {
                    left,
                    right,
                    bottom,
                    top,
                    ..
                } => Some((left as f64, right as f64, bottom as f64, top as f64)),
                _ => self.data_bounds,
            }
        } else {
            None
        };
        if is_2d {
            if let Some((l, r, b, t)) = bounds_opt {
                let (ndc_y_min, ndc_y_max) = self
                    .overlay_y_direction_for_axes(axes_index)
                    .ndc_endpoints();
                self.wgpu_renderer.update_direct_uniforms_for_axes(
                    emission,
                    axes_index,
                    DirectUniforms::new(
                        [l as f32, b as f32],
                        [r as f32, t as f32],
                        [-1.0, ndc_y_min],
                        [1.0, ndc_y_max],
                        [sw.max(1) as f32, sh.max(1) as f32],
                        log_flags,
                    ),
                );
            }
            self.wgpu_renderer.ensure_direct_triangle_pipeline(emission);
            self.wgpu_renderer.ensure_direct_line_pipeline(emission);
            self.wgpu_renderer.ensure_direct_point_pipeline(emission);
        } else {
            // 3D: ensure camera-based pipelines exist so surfaces rotate with the camera.
            self.wgpu_renderer
                .ensure_pipeline(emission, crate::core::PipelineType::Triangles);
            self.wgpu_renderer
                .ensure_pipeline(emission, crate::core::PipelineType::Lines);
            self.wgpu_renderer
                .ensure_pipeline(emission, crate::core::PipelineType::LinesNoDepth);
            self.wgpu_renderer
                .ensure_pipeline(emission, crate::core::PipelineType::EdgeStroke);
            self.wgpu_renderer
                .ensure_pipeline(emission, crate::core::PipelineType::EdgeStrokeNoDepth);
            self.wgpu_renderer
                .ensure_pipeline(emission, crate::core::PipelineType::Points);
        }
        self.wgpu_renderer.update_marker_screen_uniforms_for_axes(
            emission,
            axes_index,
            [sw.max(1) as f32, sh.max(1) as f32],
        );

        // Begin pass with Load (preserve egui)
        {
            // Prepare MSAA render target if enabled
            let use_msaa = self.wgpu_renderer.msaa_sample_count > 1;
            log::debug!(
                "runmat-plot: renderer.camera_to_target_viewport.render_pass_start axes_index={} use_msaa={} clear_background={}",
                axes_index,
                use_msaa,
                clear_background
            );

            // Depth-tested edge strokes draw in a second pass that reads the scene's depth, so an
            // edge under a surface (a fuselage's frames under its skin) stays hidden across the
            // stroke's whole width (see `edge_stroke::EdgeStrokeShader::depth_probe`).
            let depth_probe = (!is_2d
                && render_items
                    .iter()
                    .any(|(d, _, _)| d.pipeline_type == crate::core::PipelineType::EdgeStroke))
            .then(|| {
                self.wgpu_renderer
                    .depth_probe_bind_group(emission, &depth_view)
            })
            .flatten();
            for stroke_pass in [false, true] {
                if stroke_pass && depth_probe.is_none() {
                    break;
                }
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(if stroke_pass {
                        "Plot Camera Viewport Edge Stroke Pass"
                    } else {
                        "Plot Camera Viewport Pass"
                    }),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target.view,
                        resolve_target: if use_msaa {
                            target.resolve_target
                        } else {
                            None
                        },
                        ops: wgpu::Operations {
                            load: if clear_background && !stroke_pass {
                                wgpu::LoadOp::Clear(wgpu::Color {
                                    r: config.background_color.x as f64,
                                    g: config.background_color.y as f64,
                                    b: config.background_color.z as f64,
                                    a: config.background_color.w as f64,
                                })
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view.as_ref(),
                        // Read-only in the stroke pass, which samples it.
                        depth_ops: (!stroke_pass).then_some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(match config.depth_mode {
                                DepthMode::Standard => 1.0,
                                DepthMode::ReversedZ => 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                // Apply viewport and scissor rectangle to draw into the plot rect
                render_pass.set_viewport(
                    sx as f32,
                    sy as f32,
                    sw.max(1) as f32,
                    sh.max(1) as f32,
                    0.0,
                    1.0,
                );
                render_pass.set_scissor_rect(sx, sy, sw.max(1), sh.max(1));
                log::debug!(
                "runmat-plot: renderer.camera_to_target_viewport.render_pass_ready axes_index={} viewport=({}, {}, {}, {})",
                axes_index,
                sx,
                sy,
                sw.max(1),
                sh.max(1)
            );
                if let Some(ref vb_grid) = grid_vb_opt.as_ref().filter(|_| !stroke_pass) {
                    if let Some(ref pipeline) = self.wgpu_renderer.direct_line_pipeline {
                        log::debug!(
                        "runmat-plot: renderer.camera_to_target_viewport.draw_grid_start axes_index={} vertex_buffer_size={}",
                        axes_index,
                        vb_grid.size()
                    );
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(
                            0,
                            self.wgpu_renderer
                                .get_direct_uniform_bind_group_for_axes(axes_index),
                            &[],
                        );
                        render_pass.set_vertex_buffer(0, vb_grid.slice(..));
                        // Each grid line is two vertices (LineList)
                        // Draw full buffer
                        // Note: vertex count equals number of vertices
                        // wgpu will interpret as lines via pipeline topology
                        render_pass.draw(
                            0..(vb_grid.size() / std::mem::size_of::<Vertex>() as u64) as u32,
                            0..1,
                        );
                        log::debug!(
                        "runmat-plot: renderer.camera_to_target_viewport.draw_grid_ok axes_index={}",
                        axes_index
                    );
                    }
                }

                // Use direct pipelines for precise 2D mapping inside the viewport
                let use_direct_for_triangles = is_2d;
                let use_direct_for_lines = is_2d;
                let direct_tri_pipeline = if use_direct_for_triangles && bounds_opt.is_some() {
                    self.wgpu_renderer
                        .direct_triangle_pipeline
                        .as_ref()
                        .map(|p| p as *const wgpu::RenderPipeline)
                } else {
                    None
                };
                let direct_line_pipeline = if use_direct_for_lines && bounds_opt.is_some() {
                    self.wgpu_renderer
                        .direct_line_pipeline
                        .as_ref()
                        .map(|p| p as *const wgpu::RenderPipeline)
                } else {
                    None
                };
                let direct_point_pipeline = if is_2d && bounds_opt.is_some() {
                    self.wgpu_renderer
                        .direct_point_pipeline
                        .as_ref()
                        .map(|p| p as *const wgpu::RenderPipeline)
                } else {
                    None
                };

                for (idx, (render_data, vertex_buffer, index_buffer)) in
                    render_items.iter().enumerate()
                {
                    let probed_stroke = depth_probe.is_some()
                        && render_data.pipeline_type == crate::core::PipelineType::EdgeStroke;
                    if probed_stroke != stroke_pass {
                        continue;
                    }
                    let instanced = instance_buffers[idx].as_ref();
                    if instanced.is_some_and(|(_, count)| *count == 0) {
                        continue;
                    }
                    let placements = instanced.map(|(_, count)| 0..*count);
                    let is_triangles = matches!(
                        render_data.pipeline_type,
                        crate::core::PipelineType::Triangles
                    );
                    let is_lines = matches!(
                        render_data.pipeline_type,
                        crate::core::PipelineType::Lines | crate::core::PipelineType::LinesNoDepth
                    );
                    let is_points = matches!(
                        render_data.pipeline_type,
                        crate::core::PipelineType::Points | crate::core::PipelineType::Scatter3
                    );
                    let is_textured = matches!(
                        render_data.pipeline_type,
                        crate::core::PipelineType::Textured
                    );
                    // Use direct mapping for lines/triangles/points for correct pixel-sized markers in GUI
                    let use_direct = is_2d
                        && ((use_direct_for_triangles && is_triangles)
                            || (use_direct_for_lines && is_lines)
                            || is_points)
                        && bounds_opt.is_some();
                    if render_item_logs {
                        log::debug!(
                        "runmat-plot: renderer.camera_to_target_viewport.draw_item_start axes_index={} item_index={} pipeline={:?} use_direct={} textured={} indexed={} draw_calls={} point_buffer={} ",
                        axes_index,
                        idx,
                        render_data.pipeline_type,
                        use_direct,
                        is_textured,
                        index_buffer.is_some(),
                        render_data.draw_calls.len(),
                        point_buffers[idx].is_some()
                    );
                    }

                    if use_direct {
                        // Safe because we only read pointers here within pass
                        let pipeline_ref: &wgpu::RenderPipeline = unsafe {
                            if is_triangles {
                                direct_tri_pipeline.unwrap().as_ref().unwrap()
                            } else if is_lines {
                                direct_line_pipeline.unwrap().as_ref().unwrap()
                            } else {
                                direct_point_pipeline.unwrap().as_ref().unwrap()
                            }
                        };
                        let uniform_bg = self
                            .wgpu_renderer
                            .get_direct_uniform_bind_group_for_axes(axes_index);
                        render_pass.set_pipeline(pipeline_ref);
                        render_pass.set_bind_group(0, uniform_bg, &[]);
                        if is_points {
                            if let Some(ref bg) = point_style_bind_groups[idx] {
                                render_pass.set_bind_group(1, bg, &[]);
                            }
                        }
                        if render_item_logs {
                            log::debug!(
                            "runmat-plot: renderer.camera_to_target_viewport.draw_item_pipeline_ready axes_index={} item_index={} branch=direct",
                            axes_index,
                            idx
                        );
                        }
                    } else if is_textured {
                        let pipeline = self
                            .wgpu_renderer
                            .get_pipeline(crate::core::PipelineType::Textured);
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(
                            0,
                            self.wgpu_renderer
                                .get_direct_uniform_bind_group_for_axes(axes_index),
                            &[],
                        );
                        if let Some(ref bg) = image_bind_groups[idx] {
                            render_pass.set_bind_group(1, bg, &[]);
                        }
                        if render_item_logs {
                            log::debug!(
                            "runmat-plot: renderer.camera_to_target_viewport.draw_item_pipeline_ready axes_index={} item_index={} branch=textured",
                            axes_index,
                            idx
                        );
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
                            self.wgpu_renderer
                                .pipeline_for(render_data, self.scene.translucent_last())
                        });
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(
                            0,
                            self.wgpu_renderer
                                .get_uniform_bind_group_for_axes(axes_index),
                            &[],
                        );
                        if let Some(depth) = depth_probe.as_ref().filter(|_| probed_stroke) {
                            render_pass.set_bind_group(2, depth, &[]);
                        }
                        if is_points {
                            if let Some(ref bg) = point_style_bind_groups[idx] {
                                render_pass.set_bind_group(1, bg, &[]);
                            }
                            render_pass.set_bind_group(
                                2,
                                self.wgpu_renderer
                                    .get_marker_screen_bind_group_for_axes(axes_index),
                                &[],
                            );
                        } else if render_data.pipeline_type.binds_screen_uniforms() {
                            render_pass.set_bind_group(
                                1,
                                self.wgpu_renderer
                                    .get_marker_screen_bind_group_for_axes(axes_index),
                                &[],
                            );
                        }
                        if render_item_logs {
                            log::debug!(
                            "runmat-plot: renderer.camera_to_target_viewport.draw_item_pipeline_ready axes_index={} item_index={} branch=standard",
                            axes_index,
                            idx
                        );
                        }
                    }

                    if is_points && use_direct {
                        if let Some((ref buf, len)) = point_buffers[idx] {
                            render_pass.set_vertex_buffer(0, buf.slice(..));
                            render_pass.draw(0..len as u32, 0..1);
                            if render_item_logs {
                                log::debug!(
                                "runmat-plot: renderer.camera_to_target_viewport.draw_item_ok axes_index={} item_index={} mode=direct_points vertices={}",
                                axes_index,
                                idx,
                                len
                            );
                            }
                            continue;
                        }
                    } else if is_points {
                        if let Some((ref buf, _len)) = point_buffers[idx] {
                            render_pass.set_vertex_buffer(0, buf.slice(..));
                        } else {
                            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        }
                    } else {
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    }
                    if let Some((buffer, _)) = instanced {
                        render_pass.set_vertex_buffer(1, buffer.slice(..));
                    }
                    if let Some(index_buffer_ref) = index_buffer {
                        render_pass.set_index_buffer(
                            index_buffer_ref.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        if let Some(indices) = &render_data.indices {
                            for (index_range, instance_range) in
                                indexed_draw_ranges(&render_data.draw_calls, indices.len())
                            {
                                render_pass.draw_indexed(
                                    index_range,
                                    0,
                                    placements.clone().unwrap_or(instance_range),
                                );
                            }
                            if render_item_logs {
                                log::debug!(
                                "runmat-plot: renderer.camera_to_target_viewport.draw_item_ok axes_index={} item_index={} mode=indexed indices={}",
                                axes_index,
                                idx,
                                indices.len()
                            );
                            }
                        }
                    } else {
                        if is_points {
                            if let Some((_, len)) = point_buffers[idx] {
                                render_pass.draw(0..len as u32, 0..1);
                                continue;
                            }
                        }
                        for dc in &render_data.draw_calls {
                            render_pass.draw(
                                dc.vertex_offset as u32
                                    ..(dc.vertex_offset + dc.vertex_count) as u32,
                                placements.clone().unwrap_or(0..dc.instance_count as u32),
                            );
                            if render_item_logs {
                                log::debug!(
                                "runmat-plot: renderer.camera_to_target_viewport.draw_call_ok axes_index={} item_index={} mode=draw vertex_offset={} vertex_count={} instances={}",
                                axes_index,
                                idx,
                                dc.vertex_offset,
                                dc.vertex_count,
                                dc.instance_count
                            );
                            }
                        }
                    }
                }

                // Draw procedural 3D grid plane after data, depth-tested (no depth writes).
                if let Some((ref vb, ref ib)) = grid_plane_buffers.as_ref().filter(|_| !stroke_pass)
                {
                    if let Some(pipeline) = self.wgpu_renderer.grid_plane_pipeline() {
                        log::debug!(
                        "runmat-plot: renderer.camera_to_target_viewport.draw_grid_plane_start axes_index={}",
                        axes_index
                    );
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_bind_group(
                            0,
                            self.wgpu_renderer
                                .get_uniform_bind_group_for_axes(axes_index),
                            &[],
                        );
                        render_pass.set_bind_group(
                            1,
                            self.wgpu_renderer
                                .get_grid_uniform_bind_group_for_axes(axes_index),
                            &[],
                        );
                        render_pass.set_vertex_buffer(0, vb.slice(..));
                        render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..6, 0, 0..1);
                        log::debug!(
                        "runmat-plot: renderer.camera_to_target_viewport.draw_grid_plane_ok axes_index={}",
                        axes_index
                    );
                    }
                }
            }
        }

        log::debug!(
            "runmat-plot: renderer.camera_to_target_viewport.ok axes_index={} total_vertices={} total_triangles={}",
            axes_index,
            total_vertices,
            total_triangles
        );

        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            axes_viewports_px: vec![(sx, sy, sw, sh)],
            vertex_count: total_vertices,
            triangle_count: total_triangles,
            render_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
        })
    }
}
