//! Data bounds, camera fitting, view presets, and rendering into a viewport.

use super::*;

impl PlotRenderer {
    /// Calculate data bounds from scene
    pub fn calculate_data_bounds(&mut self) -> Option<(f64, f64, f64, f64)> {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if let Some(bounds) = render_data.bounds {
                    min_x = min_x.min(bounds.min.x as f64);
                    max_x = max_x.max(bounds.max.x as f64);
                    min_y = min_y.min(bounds.min.y as f64);
                    max_y = max_y.max(bounds.max.y as f64);
                    continue;
                }
                for vertex in &render_data.vertices {
                    let x = vertex.position[0] as f64;
                    let y = vertex.position[1] as f64;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }

        if min_x != f64::INFINITY && max_x != f64::NEG_INFINITY {
            // Add a small margin around data for readability without making
            // the plotted curve appear to exceed the labeled axis range.
            let x_range = (max_x - min_x).max(0.1);
            let y_range = (max_y - min_y).max(0.1);
            let x_margin = x_range * 0.04;
            let y_margin = y_range * 0.04;

            let bounds = (
                min_x - x_margin,
                max_x + x_margin,
                min_y - y_margin,
                max_y + y_margin,
            );

            // println!("Calculated data bounds: {:?}", bounds); // Too noisy
            self.data_bounds = Some(bounds);
            Some(bounds)
        } else {
            self.data_bounds = None;
            None
        }
    }

    /// Fit camera to show all data.
    ///
    /// Returns `true` if a fit was applied (i.e. bounds existed).
    pub fn fit_camera_to_data(&mut self) -> bool {
        if self.axes_cameras.len() > 1 {
            return self.fit_cameras_to_axes_data();
        }

        if self.axes_has_3d_content(0) {
            let Some(bounds) = self.display_bounds_3d_for_axes(0) else {
                return false;
            };
            let center = (bounds.min + bounds.max) * 0.5;
            let mut cam = Camera::new();
            cam.axis_scale = self.axes_3d_axis_scale(0, bounds);
            cam.target = center * cam.axis_scale;
            cam.up = Vec3::Z;
            cam.position = cam.target + Vec3::new(1.0, -1.0, 1.0);
            cam.fit_bounds(bounds.min, bounds.max);
            if let Some(axis_cam) = self.axes_cameras.get_mut(0) {
                *axis_cam = cam;
            }
            return true;
        }

        if let Some((x_min, x_max, y_min, y_max)) = self.display_bounds_for_axes(0) {
            // Match the camera bounds exactly to data bounds to align with overlay grid
            let mut cam = Self::create_default_camera();
            let l = x_min as f32;
            let r = x_max as f32;
            let b = y_min as f32;
            let t = y_max as f32;
            if let crate::core::camera::ProjectionType::Orthographic {
                ref mut left,
                ref mut right,
                ref mut bottom,
                ref mut top,
                ..
            } = cam.projection
            {
                *left = l;
                *right = r;
                *bottom = b;
                *top = t;
            }
            cam.position.z = 1.0;
            cam.target.z = 0.0;
            cam.mark_dirty();

            if let Some(axis_cam) = self.axes_cameras.get_mut(0) {
                *axis_cam = cam;
            }
            return true;
        }
        false
    }

    /// Explicit "Fit Extents" action (CAD-like). Fits the camera to current data once.
    pub fn fit_extents(&mut self) {
        let _ = if self.figure_axes_grid().0 * self.figure_axes_grid().1 > 1 {
            self.fit_cameras_to_axes_data()
        } else {
            self.fit_camera_to_data()
        };
        if self.last_figure.is_some() {
            self.arm_figure_camera_fits();
        }
        self.clear_all_axes_camera_interaction();
        self.camera_auto_fit = false;
        self.needs_update = true;
    }

    pub fn reset_geometry_view(&mut self) {
        self.set_camera_view_preset(CameraViewPreset::Perspective);
    }

    pub fn set_camera_view_preset(&mut self, preset: CameraViewPreset) {
        let bounds_by_axes: Vec<Option<BoundingBox>> = (0..self.axes_cameras.len())
            .map(|idx| {
                if self.axes_has_3d_content(idx) {
                    self.display_bounds_3d_for_axes(idx)
                } else {
                    self.axes_bounds(idx)
                }
            })
            .collect();
        let display_bounds: PerAxesViewBounds = (0..self.axes_cameras.len())
            .map(|idx| self.display_bounds_for_axes(idx))
            .collect();

        for (idx, camera) in self.axes_cameras.iter_mut().enumerate() {
            if matches!(
                camera.projection,
                crate::core::camera::ProjectionType::Perspective { .. }
            ) {
                Self::apply_perspective_view_preset(camera, preset, bounds_by_axes[idx]);
            } else if let Some((x_min, x_max, y_min, y_max)) = display_bounds[idx] {
                let mut cam = Self::create_default_camera();
                if let crate::core::camera::ProjectionType::Orthographic {
                    ref mut left,
                    ref mut right,
                    ref mut bottom,
                    ref mut top,
                    ..
                } = cam.projection
                {
                    *left = x_min as f32;
                    *right = x_max as f32;
                    *bottom = y_min as f32;
                    *top = y_max as f32;
                }
                cam.position.z = 1.0;
                cam.target.z = 0.0;
                cam.mark_dirty();
                *camera = cam;
            }
        }

        if self.last_figure.is_some() {
            self.arm_figure_camera_fits();
        }
        self.clear_all_axes_camera_interaction();
        self.camera_auto_fit = false;
        self.needs_update = true;
    }

    pub(super) fn apply_perspective_view_preset(
        camera: &mut Camera,
        preset: CameraViewPreset,
        bounds: Option<BoundingBox>,
    ) {
        let (direction, up) = Self::view_preset_orientation(preset);
        camera.up = up;

        if let Some(bounds) = bounds {
            let center = (bounds.min + bounds.max) * 0.5 * camera.axis_scale;
            camera.target = center;
            camera.position = center + direction;
            camera.fit_bounds(bounds.min, bounds.max);
        } else {
            let distance = (camera.position - camera.target).length().max(1.0);
            camera.position = camera.target + direction * distance;
            camera.mark_dirty();
        }
    }

    pub(super) fn view_preset_orientation(preset: CameraViewPreset) -> (Vec3, Vec3) {
        match preset {
            CameraViewPreset::Perspective => (Vec3::new(1.0, -1.0, 1.0).normalize(), Vec3::Z),
            CameraViewPreset::Top => (Vec3::Z, Vec3::Y),
            CameraViewPreset::Bottom => (-Vec3::Z, Vec3::Y),
            CameraViewPreset::Front => (-Vec3::Y, Vec3::Z),
            CameraViewPreset::Back => (Vec3::Y, Vec3::Z),
            CameraViewPreset::Left => (-Vec3::X, Vec3::Z),
            CameraViewPreset::Right => (Vec3::X, Vec3::Z),
        }
    }

    /// Explicit "Reset Camera" action. Restores the default orientation without re-framing.
    ///
    /// For 3D, this resets the view direction around the current data center (or current target)
    /// while preserving the current zoom distance.
    /// For 2D, this is equivalent to Fit Extents (since "home" without data bounds is rarely useful).
    pub fn reset_camera_position(&mut self) {
        let data_centers: Vec<Option<Vec3>> = (0..self.axes_cameras.len())
            .map(|idx| {
                if self.axes_has_3d_content(idx) {
                    self.display_bounds_3d_for_axes(idx)
                } else {
                    self.axes_bounds(idx)
                }
                .map(|b| (b.min + b.max) * 0.5)
            })
            .collect();
        let display_bounds: PerAxesViewBounds = (0..self.axes_cameras.len())
            .map(|idx| self.display_bounds_for_axes(idx))
            .collect();
        for (idx, c) in self.axes_cameras.iter_mut().enumerate() {
            if matches!(
                c.projection,
                crate::core::camera::ProjectionType::Perspective { .. }
            ) {
                c.reset_perspective_orientation(data_centers[idx]);
            } else if let Some((x_min, x_max, y_min, y_max)) = display_bounds[idx] {
                let mut cam = Self::create_default_camera();
                if let crate::core::camera::ProjectionType::Orthographic {
                    ref mut left,
                    ref mut right,
                    ref mut bottom,
                    ref mut top,
                    ..
                } = cam.projection
                {
                    *left = x_min as f32;
                    *right = x_max as f32;
                    *bottom = y_min as f32;
                    *top = y_max as f32;
                }
                cam.position.z = 1.0;
                cam.target.z = 0.0;
                cam.mark_dirty();
                *c = cam;
            }
        }
        self.clear_all_axes_camera_interaction();
        self.camera_auto_fit = false;
        self.needs_update = true;
    }

    /// Render the current scene to a specific viewport within a texture/surface
    pub fn render_to_viewport(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        _viewport: (f32, f32, f32, f32), // (x, y, width, height) in framebuffer coordinates
        clear_background: bool,
        background_color: Option<glam::Vec4>,
    ) -> Result<RenderResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();

        // Collect render data and create buffers first
        let mut render_items = Vec::new();
        let mut total_vertices = 0;
        let mut total_triangles = 0;

        for node in self.scene.get_visible_nodes() {
            if let Some(render_data) = &node.render_data {
                if let Some(vertex_buffer) = self.wgpu_renderer.vertex_buffer_from_sources(
                    emission,
                    render_data.gpu_vertices.as_ref(),
                    &render_data.vertices,
                ) {
                    self.wgpu_renderer
                        .ensure_pipeline(emission, render_data.pipeline_type);

                    log::trace!(
                        target: "runmat_plot",
                        "upload vertices={}, draw_calls={}",
                        render_data.vertex_count(),
                        render_data.draw_calls.len()
                    );

                    render_items.push((render_data, vertex_buffer));
                    total_vertices += render_data.vertex_count();

                    if render_data.pipeline_type == crate::core::PipelineType::Triangles {
                        total_triangles += render_data.vertex_count() / 3;
                    }
                }
            }
        }

        // Update uniforms
        let mut cam = self.camera().clone();
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

        // Create render pass (respect MSAA)
        let use_msaa = self.wgpu_renderer.msaa_sample_count > 1;
        let msaa_view_opt = if use_msaa {
            let tex = emission.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("runmat_msaa_color_camera"),
                size: wgpu::Extent3d {
                    width: self.wgpu_renderer.surface_config.width,
                    height: self.wgpu_renderer.surface_config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.wgpu_renderer.msaa_sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: self.wgpu_renderer.surface_config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            Some(tex.create_view(&wgpu::TextureViewDescriptor::default()))
        } else {
            None
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Viewport Plot Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: msaa_view_opt.as_ref().unwrap_or(target_view),
                resolve_target: if use_msaa { Some(target_view) } else { None },
                ops: wgpu::Operations {
                    load: if clear_background {
                        wgpu::LoadOp::Clear(wgpu::Color {
                            r: background_color.map_or(0.08, |c| c.x as f64),
                            g: background_color.map_or(0.09, |c| c.y as f64),
                            b: background_color.map_or(0.11, |c| c.z as f64),
                            a: background_color.map_or(1.0, |c| c.w as f64),
                        })
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // Apply viewport scissor to match overlay plot rect
        let (vx, vy, vw, vh) = _viewport;
        render_pass.set_viewport(vx, vy, vw, vh, 0.0, 1.0);

        // Configure direct-uniforms for precise data-to-NDC mapping within this viewport
        let sw = self.wgpu_renderer.surface_config.width as f32;
        let sh = self.wgpu_renderer.surface_config.height as f32;
        let ndc_left = (vx / sw) * 2.0 - 1.0;
        let ndc_right = ((vx + vw) / sw) * 2.0 - 1.0;
        let ndc_top = 1.0 - (vy / sh) * 2.0;
        let ndc_bottom = 1.0 - ((vy + vh) / sh) * 2.0;
        let direction = self.overlay_y_direction_for_axes(0);
        let (ndc_bottom, ndc_top) = (
            direction.orient(ndc_bottom, ndc_bottom, ndc_top),
            direction.orient(ndc_top, ndc_bottom, ndc_top),
        );

        // data_bounds passed in from caller: (x_min, y_min, x_max, y_max)
        let (x_min, y_min, x_max, y_max) = (0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64);
        self.wgpu_renderer.update_direct_uniforms(
            emission,
            DirectUniforms::new(
                [x_min as f32, y_min as f32],
                [x_max as f32, y_max as f32],
                [ndc_left, ndc_bottom],
                [ndc_right, ndc_top],
                [sw, sh],
                [0, 0],
            ),
        );

        // Continue with specific pipelines below (implementation omitted here)
        drop(render_pass);

        let render_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(RenderResult {
            success: true,
            data_bounds: self.data_bounds,
            axes_viewports_px: vec![(
                vx.round().max(0.0) as u32,
                vy.round().max(0.0) as u32,
                vw.round().max(1.0) as u32,
                vh.round().max(1.0) as u32,
            )],
            vertex_count: total_vertices,
            triangle_count: total_triangles,
            render_time_ms: render_time,
        })
    }
}
