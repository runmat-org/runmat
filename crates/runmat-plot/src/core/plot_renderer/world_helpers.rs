//! The 3D world helpers drawn under a scene: a procedural ground grid on XY and the origin
//! triad with tick marks (labels are drawn in the overlay).

use super::*;

impl PlotRenderer {
    /// The helper lines (axis lines and ticks, in data units) and the ground grid's quad
    /// buffers for a 3D axes. The overlay labels the same ticks from [`Self::axis_frame_3d`].
    pub(super) fn world_helpers_3d(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        cam: &Camera,
        axes_index: usize,
        view_proj_matrix: Mat4,
        model_matrix: Mat4,
        (sw, sh): (u32, u32),
    ) -> (Vec<Vertex>, Option<(wgpu::Buffer, wgpu::Buffer)>) {
        let mut grid_plane_buffers: Option<(wgpu::Buffer, wgpu::Buffer)> = None;
        let view_proj = view_proj_matrix * model_matrix;
        let inv_view_proj = view_proj.inverse();
        // The camera lives in axis-scaled space (3-D axes stretch to a cube); this helper is
        // built in data units, so read the camera's target and position back in data units.
        let axis_scale = cam.axis_scale.max(Vec3::splat(1.0e-12));
        let target_data = cam.target / axis_scale;
        let position_data = cam.position / axis_scale;
        let scene_bounds = self
            .scene
            .get_visible_nodes()
            .into_iter()
            .filter(|node| node.axes_index == axes_index)
            .filter_map(|node| node.render_data.as_ref().and_then(|data| data.bounds))
            .reduce(|a, b| a.union(&b));
        let plane_level = self
            .last_geometry_scene
            .as_ref()
            .map(|scene| scene.bounds)
            .or(scene_bounds)
            .filter(|bounds| bounds.min.is_finite() && bounds.max.is_finite())
            .map(|bounds| bounds.min.z - bounds.size().length().max(1.0) * 0.001)
            .unwrap_or(target_data.z);

        let on_grid = |x: f32, y: f32| -> Vec3 { Vec3::new(x, y, plane_level) };

        let unproject = |ndc_x: f32, ndc_y: f32, ndc_z: f32| -> Option<Vec3> {
            let clip = Vec4::new(ndc_x, ndc_y, ndc_z, 1.0);
            let world = inv_view_proj * clip;
            if !world.w.is_finite() || world.w.abs() < 1e-6 {
                return None;
            }
            let p = world.truncate() / world.w;
            if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
                Some(p)
            } else {
                None
            }
        };

        let ray_intersect_grid = |ndc_x: f32, ndc_y: f32| -> Option<Vec3> {
            // Use a near/far pair in clip space to form a ray.
            let p0 = unproject(ndc_x, ndc_y, -1.0)?;
            let p1 = unproject(ndc_x, ndc_y, 1.0)?;
            let dir = p1 - p0;
            if !dir.z.is_finite() || dir.z.abs() < 1e-8 {
                return None;
            }
            let t = (plane_level - p0.z) / dir.z;
            if !t.is_finite() || t <= 0.0 {
                return None;
            }
            Some(p0 + dir * t)
        };

        let mut plane_pts: Vec<Vec3> = Vec::new();
        for (nx, ny) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            if let Some(p) = ray_intersect_grid(nx, ny) {
                plane_pts.push(p);
            }
        }

        // Fallback region if we couldn't intersect enough rays (camera nearly parallel to plane).
        let mut min_x = 0.0_f32;
        let mut max_x = 1.0_f32;
        let mut min_y = 0.0_f32;
        let mut max_y = 1.0_f32;

        if plane_pts.len() >= 2 {
            min_x = plane_pts.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
            max_x = plane_pts
                .iter()
                .map(|p| p.x)
                .fold(f32::NEG_INFINITY, f32::max);
            min_y = plane_pts.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
            max_y = plane_pts
                .iter()
                .map(|p| p.y)
                .fold(f32::NEG_INFINITY, f32::max);
        } else if let crate::core::camera::ProjectionType::Perspective { fov, .. } = cam.projection
        {
            let dist = (cam.position - cam.target).length().max(1e-3);
            let extent = (dist * (0.5 * fov).tan() * 1.25).max(0.5);
            min_x = target_data.x - extent / axis_scale.x;
            max_x = target_data.x + extent / axis_scale.x;
            min_y = target_data.y - extent / axis_scale.y;
            max_y = target_data.y + extent / axis_scale.y;
        }

        // Expand a bit so grid lines don't pop at edges.
        let dx = (max_x - min_x).abs().max(1e-3);
        let dy = (max_y - min_y).abs().max(1e-3);
        let margin_x = dx * 0.04;
        let margin_y = dy * 0.04;
        min_x -= margin_x;
        max_x += margin_x;
        min_y -= margin_y;
        max_y += margin_y;

        let project_to_px = |p: Vec3| -> Option<(f32, f32)> {
            let clip = view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
            if !clip.w.is_finite() || clip.w.abs() < 1e-6 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            if !(ndc.x.is_finite() && ndc.y.is_finite()) {
                return None;
            }
            let px = ((ndc.x + 1.0) * 0.5) * (sw.max(1) as f32);
            let py = ((1.0 - ndc.y) * 0.5) * (sh.max(1) as f32);
            Some((px, py))
        };

        let nice_step = |raw: f64| -> f64 {
            if !raw.is_finite() || raw <= 0.0 {
                return 1.0;
            }
            let pow10 = 10.0_f64.powf(raw.log10().floor());
            let norm = raw / pow10;
            let mult = if norm <= 1.0 {
                1.0
            } else if norm <= 2.0 {
                2.0
            } else if norm <= 5.0 {
                5.0
            } else {
                10.0
            };
            mult * pow10
        };

        // Use the more visible ground direction for spacing. A nearly
        // edge-on X or Y axis should not make every major cell enormous.
        let center = on_grid(target_data.x, target_data.y);
        let px_per_world = {
            let a = project_to_px(center);
            let bx = project_to_px(center + Vec3::X);
            let by = project_to_px(center + Vec3::Y);
            match (a, bx, by) {
                (Some((ax, ay)), Some((bx, by)), Some((cx, cy))) => (bx - ax)
                    .hypot(by - ay)
                    .max((cx - ax).hypot(cy - ay))
                    .max(1e-3),
                _ => 1.0,
            }
        };
        let desired_major_px = 96.0_f64;
        let major_step = nice_step((desired_major_px / (px_per_world as f64)).max(1e-6));
        let mut minor_step = major_step / 10.0;
        if !minor_step.is_finite() || minor_step <= 0.0 {
            minor_step = major_step.max(1.0);
        }

        // Cap minor line density to avoid noisy/perf-heavy grids.
        let max_minor_lines = 180.0;
        let minor_count_x = (dx as f64 / minor_step).abs();
        let minor_count_y = (dy as f64 / minor_step).abs();
        if minor_count_x > max_minor_lines || minor_count_y > max_minor_lines {
            minor_step = (major_step / 5.0).max(major_step); // effectively disable minors
        }

        let mut helper_vertices: Vec<Vertex> = Vec::new();
        let mut push_line = |a: Vec3, b: Vec3, color: Vec4| {
            helper_vertices.push(Vertex::new(a, color));
            helper_vertices.push(Vertex::new(b, color));
        };

        // World-axis helper for spatial awareness at model scale. Each axis has its own
        // tick step, since 3-D axes may scale each data axis differently.
        // Each axis reaches the far edge of the data from the origin, with about four
        // ticks; without data bounds, the step follows the on-screen scale instead.
        let frame = self.axis_frame_3d(axes_index).unwrap_or_else(|| {
            // No finite bounds: size the triad from the on-screen scale at the origin.
            let steps = Self::world_axis_steps(&project_to_px, center, &nice_step);
            let lens = steps.map(|step| {
                (step as f32 * 5.0)
                    .clamp((step as f32).min(0.5), (dx.max(dy) * 0.6).max(step as f32))
            });
            AxisFrame3d::origin_triad(steps, lens)
        });
        // Procedural ground grid (depth-tested, no depth writes).
        let show_major_grid = self.overlay_show_grid_for_axes(axes_index);
        let show_minor_grid = self.overlay_show_minor_grid_for_axes(axes_index);
        if show_major_grid || show_minor_grid {
            let theme = self.theme.build_theme();
            let bg = theme.get_background_color();
            let grid = theme.get_grid_color();
            let bg_luma = 0.2126 * bg.x + 0.7152 * bg.y + 0.0722 * bg.z;
            let mut major_rgb = [grid.x, grid.y, grid.z];
            let mut minor_rgb = [grid.x, grid.y, grid.z];
            let mut major_alpha = grid.w.clamp(0.22, 0.34);
            let mut minor_alpha = (grid.w * 0.55).clamp(0.07, 0.16);
            if bg_luma <= 0.62 {
                major_rgb = [grid.x * 0.80, grid.y * 0.80, grid.z * 0.80];
                minor_rgb = [grid.x * 0.68, grid.y * 0.68, grid.z * 0.68];
            }
            if bg_luma > 0.62 {
                major_rgb = [grid.x * 0.45, grid.y * 0.45, grid.z * 0.45];
                minor_rgb = [grid.x * 0.33, grid.y * 0.33, grid.z * 0.33];
                major_alpha = major_alpha.max(0.32);
                minor_alpha = minor_alpha.max(0.14);
            }
            if !show_major_grid {
                major_alpha = 0.0;
            }
            if !show_minor_grid {
                minor_alpha = 0.0;
            }
            // A plot's grid fills the floor of its axis limits with lines at its tick values,
            // as MATLAB's plot box does. CAD geometry keeps the view-filling ground grid that
            // fades out around the model.
            let grid_steps = if frame.world_origin {
                GridPlaneExtent {
                    min: (min_x, min_y),
                    max: (max_x, max_y),
                    major: [major_step as f32; 2],
                    minor: [minor_step as f32; 2],
                    fade: [
                        (0.85 * dx.max(dy)).max(major_step as f32),
                        (1.40 * dx.max(dy)).max((major_step as f32) * 2.0),
                    ],
                }
            } else {
                let major = [frame.steps[0] as f32, frame.steps[1] as f32];
                GridPlaneExtent {
                    min: (frame.origin.x, frame.origin.y),
                    max: (frame.origin.x + frame.lens[0], frame.origin.y + frame.lens[1]),
                    major,
                    minor: major.map(|step| step / 5.0),
                    fade: [f32::MAX * 0.5, f32::MAX],
                }
            };
            self.wgpu_renderer.ensure_grid_plane_pipeline(emission);
            self.wgpu_renderer.update_grid_uniforms_for_axes(
                emission,
                axes_index,
                crate::core::renderer::GridUniforms {
                    major_step_x: grid_steps.major[0],
                    minor_step_x: grid_steps.minor[0],
                    fade_start: grid_steps.fade[0],
                    fade_end: grid_steps.fade[1],
                    camera_pos: position_data.to_array(),
                    major_step_y: grid_steps.major[1],
                    target_pos: [target_data.x, target_data.y, plane_level],
                    minor_step_y: grid_steps.minor[1],
                    major_color: [major_rgb[0], major_rgb[1], major_rgb[2], major_alpha],
                    minor_color: [minor_rgb[0], minor_rgb[1], minor_rgb[2], minor_alpha],
                },
            );

            let quad_vertices = [
                Vertex::new(on_grid(grid_steps.min.0, grid_steps.min.1), Vec4::ONE),
                Vertex::new(on_grid(grid_steps.max.0, grid_steps.min.1), Vec4::ONE),
                Vertex::new(on_grid(grid_steps.max.0, grid_steps.max.1), Vec4::ONE),
                Vertex::new(on_grid(grid_steps.min.0, grid_steps.max.1), Vec4::ONE),
            ];
            let quad_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
            let vb = self
                .wgpu_renderer
                .create_vertex_buffer(emission, &quad_vertices);
            let ib = self
                .wgpu_renderer
                .create_index_buffer(emission, &quad_indices);
            grid_plane_buffers = Some((vb, ib));
        }

        // A result scene leaves the world triad out (see `shows_world_triad`).
        if self.shows_world_triad(axes_index) {
            let origin = frame.origin;
            let col_x = Vec4::new(0.92, 0.25, 0.25, 0.85);
            let col_y = Vec4::new(0.35, 0.90, 0.45, 0.85);
            let col_z = Vec4::new(0.35, 0.62, 0.98, 0.85);
            push_line(origin, origin + Vec3::new(frame.lens[0], 0.0, 0.0), col_x);
            push_line(origin, origin + Vec3::new(0.0, frame.lens[1], 0.0), col_y);
            push_line(
                frame.z_base,
                frame.z_base + Vec3::new(0.0, 0.0, frame.lens[2]),
                col_z,
            );

            // Tick marks on the axis lines (major step only). Labels are drawn in the overlay
            // from the same frame so they stay crisp; these marks anchor them in depth.
            for (axis_index, (axis, perp, perp_index, col)) in [
                (Vec3::X, Vec3::Y, 1, col_x),
                (Vec3::Y, Vec3::X, 0, col_y),
                (Vec3::Z, Vec3::X, 0, col_z),
            ]
            .into_iter()
            .enumerate()
            {
                // A tick mark extends along the perpendicular axis, so its length is measured in
                // that axis's units: 3-D axes may scale x, y and z differently.
                let perp_len = frame.lens[perp_index];
                let perp_step = frame.steps[perp_index] as f32;
                // `f32::clamp` panics if min > max; keep the minimum below the maximum.
                let tick_max = (perp_step * 0.25).max(1.0e-6);
                let tick_min = (perp_len * 0.002).min(tick_max);
                let tick_len = (perp_len * 0.02).clamp(tick_min, tick_max);
                for (offset, _value) in frame.ticks(axis_index) {
                    let p = frame.axis_start(axis_index) + axis * offset;
                    push_line(
                        p - perp * tick_len,
                        p + perp * tick_len,
                        Vec4::new(col.x, col.y, col.z, col.w * 0.85),
                    );
                }
            }
        }

        (helper_vertices, grid_plane_buffers)
    }
}
