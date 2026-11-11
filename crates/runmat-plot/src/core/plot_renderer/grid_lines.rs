//! 2D grid lines for an axes: major, minor, log and polar.

use super::*;

impl PlotRenderer {
    /// Overlay configuration getters
    pub fn overlay_show_grid(&self) -> bool {
        self.figure_show_grid
    }

    pub fn set_overlay_grid_enabled(&mut self, enabled: bool) {
        self.figure_show_grid = enabled;
        self.needs_update = true;
    }

    pub fn overlay_show_minor_grid(&self) -> bool {
        self.figure_show_minor_grid
    }
    pub fn overlay_show_grid_for_axes(&self, axes_index: usize) -> bool {
        self.last_figure
            .as_ref()
            .and_then(|f| f.axes_metadata(axes_index))
            .map(|m| m.grid_enabled)
            .unwrap_or(self.figure_show_grid)
    }
    pub fn overlay_show_minor_grid_for_axes(&self, axes_index: usize) -> bool {
        Self::minor_grid_for_axes(
            self.last_figure.as_ref(),
            self.figure_show_minor_grid,
            axes_index,
        )
    }

    pub fn overlay_axes_kind_for_axes(&self, axes_index: usize) -> AxesKind {
        self.last_figure
            .as_ref()
            .map(|f| f.axes_kind(axes_index))
            .unwrap_or(AxesKind::Cartesian)
    }

    pub(super) fn minor_grid_for_axes(
        last_figure: Option<&crate::plots::Figure>,
        figure_show_minor_grid: bool,
        axes_index: usize,
    ) -> bool {
        last_figure
            .map(|f| f.minor_grid_enabled_for_axes(axes_index))
            .unwrap_or(figure_show_minor_grid)
    }

    pub(super) fn push_vertical_grid_lines(
        vertices: &mut Vec<Vertex>,
        x_bounds: (f64, f64),
        y_bounds: (f64, f64),
        step: f64,
        color: Vec4,
        skip_major: Option<(f64, f64)>,
    ) {
        Self::for_each_grid_position(x_bounds.0, x_bounds.1, step, skip_major, |x| {
            let x = x as f32;
            let y0 = y_bounds.0 as f32;
            let y1 = y_bounds.1 as f32;
            if x.is_finite() && y0.is_finite() && y1.is_finite() {
                vertices.push(Vertex::new(Vec3::new(x, y0, 0.0), color));
                vertices.push(Vertex::new(Vec3::new(x, y1, 0.0), color));
            }
        });
    }

    pub(super) fn push_horizontal_grid_lines(
        vertices: &mut Vec<Vertex>,
        y_bounds: (f64, f64),
        x_bounds: (f64, f64),
        step: f64,
        color: Vec4,
        skip_major: Option<(f64, f64)>,
    ) {
        Self::for_each_grid_position(y_bounds.0, y_bounds.1, step, skip_major, |y| {
            let y = y as f32;
            let x0 = x_bounds.0 as f32;
            let x1 = x_bounds.1 as f32;
            if y.is_finite() && x0.is_finite() && x1.is_finite() {
                vertices.push(Vertex::new(Vec3::new(x0, y, 0.0), color));
                vertices.push(Vertex::new(Vec3::new(x1, y, 0.0), color));
            }
        });
    }

    /// Vertical grid lines at explicit tick values inside `x_bounds`.
    /// Grid values on a log axis spanning `lo..hi`: whole decades, or the 2..9 multiples
    /// between them when `minor` is set. Empty unless both limits are positive.
    pub(super) fn log_grid_values(lo: f64, hi: f64, minor: bool) -> Vec<f64> {
        let (lo, hi) = (lo.min(hi), lo.max(hi));
        if !(lo > 0.0 && hi.is_finite()) {
            return Vec::new();
        }
        let first = lo.log10().floor() as i32;
        let last = hi.log10().ceil() as i32;
        let multiples: &[f64] = if minor {
            &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        } else {
            &[1.0]
        };
        (first..=last)
            .flat_map(|decade| {
                let base = 10f64.powi(decade);
                multiples.iter().map(move |m| m * base)
            })
            .filter(|value| *value >= lo && *value <= hi)
            .collect()
    }

    pub(super) fn push_vertical_grid_lines_at(
        vertices: &mut Vec<Vertex>,
        ticks: &[f64],
        x_bounds: (f64, f64),
        y_bounds: (f64, f64),
        color: Vec4,
    ) {
        let (lo, hi) = (x_bounds.0.min(x_bounds.1), x_bounds.0.max(x_bounds.1));
        for &x in ticks
            .iter()
            .filter(|x| x.is_finite() && **x >= lo && **x <= hi)
        {
            vertices.push(Vertex::new(
                Vec3::new(x as f32, y_bounds.0 as f32, 0.0),
                color,
            ));
            vertices.push(Vertex::new(
                Vec3::new(x as f32, y_bounds.1 as f32, 0.0),
                color,
            ));
        }
    }

    /// Horizontal grid lines at explicit tick values inside `y_bounds`.
    pub(super) fn push_horizontal_grid_lines_at(
        vertices: &mut Vec<Vertex>,
        ticks: &[f64],
        y_bounds: (f64, f64),
        x_bounds: (f64, f64),
        color: Vec4,
    ) {
        let (lo, hi) = (y_bounds.0.min(y_bounds.1), y_bounds.0.max(y_bounds.1));
        for &y in ticks
            .iter()
            .filter(|y| y.is_finite() && **y >= lo && **y <= hi)
        {
            vertices.push(Vertex::new(
                Vec3::new(x_bounds.0 as f32, y as f32, 0.0),
                color,
            ));
            vertices.push(Vertex::new(
                Vec3::new(x_bounds.1 as f32, y as f32, 0.0),
                color,
            ));
        }
    }

    pub(super) fn push_polar_grid_lines(
        vertices: &mut Vec<Vertex>,
        radius: f64,
        major_step: f64,
        show_major: bool,
        show_minor: bool,
        major_color: Vec4,
        minor_color: Vec4,
    ) {
        if !radius.is_finite() || radius <= 0.0 || !major_step.is_finite() || major_step <= 0.0 {
            return;
        }

        if show_minor {
            let minor_step = (major_step / 5.0).max(f64::EPSILON);
            Self::for_each_grid_position(
                minor_step,
                radius,
                minor_step,
                Some((major_step, minor_step * 0.25)),
                |r| {
                    Self::push_circle_grid_line(vertices, r, minor_color);
                },
            );
        }
        if show_major {
            Self::for_each_grid_position(major_step, radius, major_step, None, |r| {
                Self::push_circle_grid_line(vertices, r, major_color);
            });
            for i in 0..12 {
                let theta = i as f64 * std::f64::consts::TAU / 12.0;
                let x = (theta.cos() * radius) as f32;
                let y = (theta.sin() * radius) as f32;
                if x.is_finite() && y.is_finite() {
                    vertices.push(Vertex::new(Vec3::ZERO, major_color));
                    vertices.push(Vertex::new(Vec3::new(x, y, 0.0), major_color));
                }
            }
        }
    }

    pub(super) fn push_circle_grid_line(vertices: &mut Vec<Vertex>, radius: f64, color: Vec4) {
        if !radius.is_finite() || radius <= 0.0 {
            return;
        }
        const SEGMENTS: usize = 96;
        for i in 0..SEGMENTS {
            let theta0 = i as f64 * std::f64::consts::TAU / SEGMENTS as f64;
            let theta1 = (i + 1) as f64 * std::f64::consts::TAU / SEGMENTS as f64;
            let p0 = Vec3::new(
                (theta0.cos() * radius) as f32,
                (theta0.sin() * radius) as f32,
                0.0,
            );
            let p1 = Vec3::new(
                (theta1.cos() * radius) as f32,
                (theta1.sin() * radius) as f32,
                0.0,
            );
            if p0.is_finite() && p1.is_finite() {
                vertices.push(Vertex::new(p0, color));
                vertices.push(Vertex::new(p1, color));
            }
        }
    }

    pub(super) fn for_each_grid_position(
        min: f64,
        max: f64,
        step: f64,
        skip_major: Option<(f64, f64)>,
        mut visit: impl FnMut(f64),
    ) {
        if !min.is_finite() || !max.is_finite() || !step.is_finite() || step <= 0.0 || min > max {
            return;
        }
        let start = (min / step).ceil() * step;
        if !start.is_finite() {
            return;
        }

        let mut value = start;
        for _ in 0..MAX_2D_GRID_LINES_PER_AXIS {
            if value > max {
                break;
            }
            let is_major = skip_major
                .map(|(major_step, tolerance)| {
                    major_step.is_finite()
                        && major_step > 0.0
                        && (value - (value / major_step).round() * major_step).abs() <= tolerance
                })
                .unwrap_or(false);
            if !is_major {
                visit(value);
            }

            let next = value + step;
            if !next.is_finite() || next <= value {
                break;
            }
            value = next;
        }
    }
}
