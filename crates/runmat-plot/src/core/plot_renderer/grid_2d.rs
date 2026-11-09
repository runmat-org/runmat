//! The 2D grid drawn under an axes' data: major and minor lines (log, polar and explicit
//! ticks included), and the direct uniforms that map the axes' limits to the viewport.

use super::*;

impl PlotRenderer {
    /// The 2D grid's line vertices in a GPU buffer, when the axes shows a grid and has view
    /// bounds; `None` draws no grid.
    pub(super) fn grid_2d_buffer(
        &mut self,
        emission: &runmat_accelerate_api::WgpuEmission<'_>,
        axes_index: usize,
        (sw, sh): (u32, u32),
        log_flags: [u32; 2],
    ) -> Option<wgpu::Buffer> {
        let show_major_grid = self.overlay_show_grid_for_axes(axes_index);
        let show_minor_grid = self.overlay_show_minor_grid_for_axes(axes_index);
        if !(show_major_grid || show_minor_grid) {
            return None;
        }
        let mut grid_vb_opt: Option<wgpu::Buffer> = None;
        if let Some((mut l, mut r, mut b, mut t)) = self.view_bounds_for_axes(axes_index) {
            if self.overlay_axes_kind_for_axes(axes_index) == AxesKind::Polar {
                let radius = l.abs().max(r.abs()).max(b.abs()).max(t.abs()).max(1e-6);
                l = -radius;
                r = radius;
                b = -radius;
                t = radius;
            }
            // Reflect output coordinates after log mapping; limits stay ordered.
            let (ndc_y_min, ndc_y_max) = self
                .overlay_y_direction_for_axes(axes_index)
                .ndc_endpoints();
            // Update direct uniforms mapping for viewport
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
            self.wgpu_renderer.ensure_direct_line_pipeline(emission);

            let x_range = (r - l).max(1e-6);
            let y_range = (t - b).max(1e-6);
            let x_step = plot_utils::calculate_tick_interval(x_range);
            let y_step = plot_utils::calculate_tick_interval(y_range);
            let mut grid_vertices: Vec<Vertex> = Vec::new();
            // Custom ticks (xticks/yticks) set where the overlay draws tick labels, so major
            // grid lines must use the same values; the evenly spaced minor grid is skipped.
            let explicit_x_ticks = self.overlay_x_ticks_for_axes(axes_index);
            let explicit_y_ticks = self.overlay_y_ticks_for_axes(axes_index);
            // Log axes are labeled at whole decades; the grid follows those, with minor
            // lines at 2..9 times each decade. Even spacing in data units would bunch up.
            let (x_log, y_log) = (log_flags[0] != 0, log_flags[1] != 0);
            let log_major_x =
                (x_log && explicit_x_ticks.is_none()).then(|| Self::log_grid_values(l, r, false));
            let log_major_y =
                (y_log && explicit_y_ticks.is_none()).then(|| Self::log_grid_values(b, t, false));
            let explicit_x_ticks = explicit_x_ticks.or(log_major_x);
            let explicit_y_ticks = explicit_y_ticks.or(log_major_y);
            let g = 80.0_f32 / 255.0_f32;
            let major_col = Vec4::new(g, g, g, 1.0);
            let minor_col = Vec4::new(g, g, g, 0.42);
            if self.overlay_axes_kind_for_axes(axes_index) == AxesKind::Polar {
                let radius = l.abs().max(r.abs()).max(b.abs()).max(t.abs()).max(1e-6);
                let step = plot_utils::calculate_tick_interval(radius);
                Self::push_polar_grid_lines(
                    &mut grid_vertices,
                    radius,
                    step,
                    show_major_grid,
                    show_minor_grid,
                    major_col,
                    minor_col,
                );
            } else {
                if show_minor_grid && x_log {
                    Self::push_vertical_grid_lines_at(
                        &mut grid_vertices,
                        &Self::log_grid_values(l, r, true),
                        (l, r),
                        (b, t),
                        minor_col,
                    );
                } else if show_minor_grid
                    && explicit_x_ticks.is_none()
                    && x_step.is_finite()
                    && x_step > 0.0
                {
                    let minor_step = (x_step / 5.0).max(f64::EPSILON);
                    Self::push_vertical_grid_lines(
                        &mut grid_vertices,
                        (l, r),
                        (b, t),
                        minor_step,
                        minor_col,
                        Some((x_step, minor_step * 0.25)),
                    );
                }
                if show_minor_grid && y_log {
                    Self::push_horizontal_grid_lines_at(
                        &mut grid_vertices,
                        &Self::log_grid_values(b, t, true),
                        (b, t),
                        (l, r),
                        minor_col,
                    );
                } else if show_minor_grid
                    && explicit_y_ticks.is_none()
                    && y_step.is_finite()
                    && y_step > 0.0
                {
                    let minor_step = (y_step / 5.0).max(f64::EPSILON);
                    Self::push_horizontal_grid_lines(
                        &mut grid_vertices,
                        (b, t),
                        (l, r),
                        minor_step,
                        minor_col,
                        Some((y_step, minor_step * 0.25)),
                    );
                }
                if let (true, Some(ticks)) = (show_major_grid, explicit_x_ticks.as_ref()) {
                    Self::push_vertical_grid_lines_at(
                        &mut grid_vertices,
                        ticks,
                        (l, r),
                        (b, t),
                        major_col,
                    );
                } else if show_major_grid && x_step.is_finite() && x_step > 0.0 {
                    Self::push_vertical_grid_lines(
                        &mut grid_vertices,
                        (l, r),
                        (b, t),
                        x_step,
                        major_col,
                        None,
                    );
                }
                if let (true, Some(ticks)) = (show_major_grid, explicit_y_ticks.as_ref()) {
                    Self::push_horizontal_grid_lines_at(
                        &mut grid_vertices,
                        ticks,
                        (b, t),
                        (l, r),
                        major_col,
                    );
                } else if show_major_grid && y_step.is_finite() && y_step > 0.0 {
                    Self::push_horizontal_grid_lines(
                        &mut grid_vertices,
                        (b, t),
                        (l, r),
                        y_step,
                        major_col,
                        None,
                    );
                }
            }
            if !grid_vertices.is_empty() {
                grid_vb_opt = Some(
                    self.wgpu_renderer
                        .create_vertex_buffer(emission, &grid_vertices),
                );
            }
        }
        grid_vb_opt
    }
}
