use crate::core::{Camera, PipelineType, ProjectionType, RenderData};
use crate::plots::{Figure, PlotElement};
use crate::styling::PlotThemeConfig;
use font8x8::{UnicodeFonts, BASIC_FONTS};
use glam::{Vec2, Vec3, Vec4};

#[derive(Clone, Debug)]
struct AxesView {
    viewport: (u32, u32, u32, u32),
    plot_rect: (u32, u32, u32, u32),
    bounds_2d: (f32, f32, f32, f32),
    bounds_3d: (Vec3, Vec3),
    camera_3d: Option<Camera>,
    has_3d_content: bool,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    z_label: Option<String>,
    title_scale: u32,
    label_scale: u32,
    tick_scale: u32,
    show_grid: bool,
    show_box: bool,
}

#[derive(Clone, Copy, Debug)]
struct ScreenVertex {
    x: f32,
    y: f32,
    z: f32,
    color: [u8; 4],
}

struct Canvas {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    depth: Vec<f32>,
}

impl Canvas {
    fn new(width: u32, height: u32, background: [u8; 4]) -> Self {
        let mut pixels = vec![0u8; (width.max(1) * height.max(1) * 4) as usize];
        for px in pixels.chunks_exact_mut(4) {
            px.copy_from_slice(&background);
        }
        let depth = vec![f32::INFINITY; (width.max(1) * height.max(1)) as usize];
        Self {
            width: width.max(1),
            height: height.max(1),
            pixels,
            depth,
        }
    }

    fn rgba(self) -> Vec<u8> {
        self.pixels
    }

    fn blend_pixel(&mut self, x: i32, y: i32, rgba: [u8; 4], depth: f32, use_depth: bool) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let idx = (y as u32 * self.width + x as u32) as usize;
        if use_depth {
            if !depth.is_finite() || depth >= self.depth[idx] {
                return;
            }
            self.depth[idx] = depth;
        }

        let p = idx * 4;
        let src_a = rgba[3] as f32 / 255.0;
        let dst_a = self.pixels[p + 3] as f32 / 255.0;
        let out_a = src_a + dst_a * (1.0 - src_a);
        if out_a <= f32::EPSILON {
            self.pixels[p..p + 4].copy_from_slice(&[0, 0, 0, 0]);
            return;
        }
        for (i, src_u8) in rgba.iter().take(3).enumerate() {
            let src = *src_u8 as f32 / 255.0;
            let dst = self.pixels[p + i] as f32 / 255.0;
            let out = (src * src_a + dst * dst_a * (1.0 - src_a)) / out_a;
            self.pixels[p + i] = (out.clamp(0.0, 1.0) * 255.0) as u8;
        }
        self.pixels[p + 3] = (out_a.clamp(0.0, 1.0) * 255.0) as u8;
    }

    fn draw_disc(&mut self, center: Vec2, radius: f32, rgba: [u8; 4], depth: f32, use_depth: bool) {
        let r = radius.max(0.5);
        let min_x = (center.x - r).floor() as i32;
        let max_x = (center.x + r).ceil() as i32;
        let min_y = (center.y - r).floor() as i32;
        let max_y = (center.y + r).ceil() as i32;
        let rr = r * r;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 + 0.5 - center.x;
                let dy = y as f32 + 0.5 - center.y;
                if dx * dx + dy * dy <= rr {
                    self.blend_pixel(x, y, rgba, depth, use_depth);
                }
            }
        }
    }

    fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, rgba: [u8; 4]) {
        if w <= 0 || h <= 0 {
            return;
        }
        let x0 = x.max(0);
        let y0 = y.max(0);
        let x1 = (x + w).min(self.width as i32);
        let y1 = (y + h).min(self.height as i32);
        for yy in y0..y1 {
            for xx in x0..x1 {
                self.blend_pixel(xx, yy, rgba, 0.0, false);
            }
        }
    }

    fn stroke_rect(&mut self, x: i32, y: i32, w: i32, h: i32, rgba: [u8; 4], width_px: f32) {
        let l = x as f32;
        let r = (x + w - 1) as f32;
        let t = y as f32;
        let b = (y + h - 1) as f32;
        let c = rgba;
        self.draw_line(
            ScreenVertex {
                x: l,
                y: t,
                z: 0.0,
                color: c,
            },
            ScreenVertex {
                x: r,
                y: t,
                z: 0.0,
                color: c,
            },
            width_px,
            0,
            false,
        );
        self.draw_line(
            ScreenVertex {
                x: r,
                y: t,
                z: 0.0,
                color: c,
            },
            ScreenVertex {
                x: r,
                y: b,
                z: 0.0,
                color: c,
            },
            width_px,
            0,
            false,
        );
        self.draw_line(
            ScreenVertex {
                x: r,
                y: b,
                z: 0.0,
                color: c,
            },
            ScreenVertex {
                x: l,
                y: b,
                z: 0.0,
                color: c,
            },
            width_px,
            0,
            false,
        );
        self.draw_line(
            ScreenVertex {
                x: l,
                y: b,
                z: 0.0,
                color: c,
            },
            ScreenVertex {
                x: l,
                y: t,
                z: 0.0,
                color: c,
            },
            width_px,
            0,
            false,
        );
    }

    fn draw_line(
        &mut self,
        a: ScreenVertex,
        b: ScreenVertex,
        width_px: f32,
        style_code: i32,
        use_depth: bool,
    ) {
        let radius = width_px.max(1.0) * 0.5;
        let segments = dash_segments(a, b, style_code, radius.max(1.0));
        for (s0, s1) in segments {
            self.draw_capsule_segment(s0, s1, radius, use_depth);
        }
    }

    fn draw_capsule_segment(
        &mut self,
        a: ScreenVertex,
        b: ScreenVertex,
        radius: f32,
        use_depth: bool,
    ) {
        let min_x = (a.x.min(b.x) - radius - 1.0).floor() as i32;
        let max_x = (a.x.max(b.x) + radius + 1.0).ceil() as i32;
        let min_y = (a.y.min(b.y) - radius - 1.0).floor() as i32;
        let max_y = (a.y.max(b.y) + radius + 1.0).ceil() as i32;

        let av = Vec2::new(a.x, a.y);
        let bv = Vec2::new(b.x, b.y);
        let ab = bv - av;
        let ab_len2 = ab.length_squared().max(1e-8);

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let t = ((p - av).dot(ab) / ab_len2).clamp(0.0, 1.0);
                let closest = av + ab * t;
                let dist = p.distance(closest);
                if dist > radius + 1.0 {
                    continue;
                }
                let coverage = (radius + 1.0 - dist).clamp(0.0, 1.0);
                if coverage <= 0.0 {
                    continue;
                }

                let depth = a.z + (b.z - a.z) * t;
                let mut color = lerp_rgba(a.color, b.color, t);
                color[3] = ((color[3] as f32) * coverage).round().clamp(0.0, 255.0) as u8;
                self.blend_pixel(x, y, color, depth, use_depth);
            }
        }
    }

    fn fill_triangle(
        &mut self,
        v0: ScreenVertex,
        v1: ScreenVertex,
        v2: ScreenVertex,
        use_depth: bool,
    ) {
        let min_x = v0.x.min(v1.x).min(v2.x).floor() as i32;
        let max_x = v0.x.max(v1.x).max(v2.x).ceil() as i32;
        let min_y = v0.y.min(v1.y).min(v2.y).floor() as i32;
        let max_y = v0.y.max(v1.y).max(v2.y).ceil() as i32;

        let p0 = Vec2::new(v0.x, v0.y);
        let p1 = Vec2::new(v1.x, v1.y);
        let p2 = Vec2::new(v2.x, v2.y);
        let area = edge_fn(p0, p1, p2);
        if area.abs() <= f32::EPSILON {
            return;
        }

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let w0 = edge_fn(p1, p2, p) / area;
                let w1 = edge_fn(p2, p0, p) / area;
                let w2 = edge_fn(p0, p1, p) / area;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let depth = w0 * v0.z + w1 * v1.z + w2 * v2.z;
                let color = blend_barycentric_rgba(v0.color, v1.color, v2.color, w0, w1, w2);
                self.blend_pixel(x, y, color, depth, use_depth);
            }
        }
    }
}

fn edge_fn(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
}

fn blend_barycentric_rgba(
    c0: [u8; 4],
    c1: [u8; 4],
    c2: [u8; 4],
    w0: f32,
    w1: f32,
    w2: f32,
) -> [u8; 4] {
    let mix = |a: u8, b: u8, c: u8| -> u8 {
        (a as f32 * w0 + b as f32 * w1 + c as f32 * w2)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    [
        mix(c0[0], c1[0], c2[0]),
        mix(c0[1], c1[1], c2[1]),
        mix(c0[2], c1[2], c2[2]),
        mix(c0[3], c1[3], c2[3]),
    ]
}

fn lerp_rgba(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    let mix = |x: u8, y: u8| -> u8 {
        (x as f32 + (y as f32 - x as f32) * t)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    [
        mix(a[0], b[0]),
        mix(a[1], b[1]),
        mix(a[2], b[2]),
        mix(a[3], b[3]),
    ]
}

fn with_alpha(color: [u8; 4], alpha_scale: f32) -> [u8; 4] {
    let mut out = color;
    out[3] = ((out[3] as f32) * alpha_scale).round().clamp(0.0, 255.0) as u8;
    out
}

fn dash_segments(
    a: ScreenVertex,
    b: ScreenVertex,
    style_code: i32,
    width_px: f32,
) -> Vec<(ScreenVertex, ScreenVertex)> {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= 1e-5 {
        return vec![(a, b)];
    }
    let pattern = match style_code {
        1 => vec![(6.0 * width_px, true), (6.0 * width_px, false)],
        2 => vec![(1.5 * width_px, true), (5.0 * width_px, false)],
        3 => vec![
            (6.0 * width_px, true),
            (4.0 * width_px, false),
            (1.5 * width_px, true),
            (4.0 * width_px, false),
        ],
        _ => return vec![(a, b)],
    };

    let mut out = Vec::new();
    let mut s = 0.0f32;
    let mut pi = 0usize;
    while s < len {
        let (step, draw) = pattern[pi % pattern.len()];
        let e = (s + step.max(1.0)).min(len);
        if draw {
            let t0 = s / len;
            let t1 = e / len;
            out.push((lerp_screen(a, b, t0), lerp_screen(a, b, t1)));
        }
        s = e;
        pi += 1;
    }
    out
}

fn lerp_screen(a: ScreenVertex, b: ScreenVertex, t: f32) -> ScreenVertex {
    ScreenVertex {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        z: a.z + (b.z - a.z) * t,
        color: lerp_rgba(a.color, b.color, t),
    }
}

fn to_u8_rgba(color: [f32; 4]) -> [u8; 4] {
    [
        (color[0].clamp(0.0, 1.0) * 255.0) as u8,
        (color[1].clamp(0.0, 1.0) * 255.0) as u8,
        (color[2].clamp(0.0, 1.0) * 255.0) as u8,
        (color[3].clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

fn is_default_figure_bg(bg: Vec4) -> bool {
    const EPS: f32 = 1e-3;
    (bg.x - 1.0).abs() <= EPS
        && (bg.y - 1.0).abs() <= EPS
        && (bg.z - 1.0).abs() <= EPS
        && (bg.w - 1.0).abs() <= EPS
}

fn compute_tiled_viewports(
    width: u32,
    height: u32,
    rows: usize,
    cols: usize,
) -> Vec<(u32, u32, u32, u32)> {
    if rows == 0 || cols == 0 {
        return vec![(0, 0, width.max(1), height.max(1))];
    }
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let cell_w = (width / cols_u).max(1);
    let cell_h = (height / rows_u).max(1);
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows_u {
        for c in 0..cols_u {
            let x = c * cell_w;
            let y = r * cell_h;
            let w = if c + 1 == cols_u {
                width.saturating_sub(x).max(1)
            } else {
                cell_w
            };
            let h = if r + 1 == rows_u {
                height.saturating_sub(y).max(1)
            } else {
                cell_h
            };
            out.push((x, y, w, h));
        }
    }
    out
}

fn compute_plot_rect(viewport: (u32, u32, u32, u32), has_3d: bool) -> (u32, u32, u32, u32) {
    let (vx, vy, vw, vh) = viewport;
    let left = if has_3d { 48 } else { 62 };
    let right = 24;
    let top = if has_3d { 34 } else { 40 };
    let bottom = if has_3d { 48 } else { 54 };

    let px = vx + left.min(vw.saturating_sub(2));
    let py = vy + top.min(vh.saturating_sub(2));
    let pw = vw
        .saturating_sub(left + right)
        .max(vw.saturating_sub(2).max(1));
    let ph = vh
        .saturating_sub(top + bottom)
        .max(vh.saturating_sub(2).max(1));
    (px, py, pw.max(1), ph.max(1))
}

fn project_2d(
    pos: Vec3,
    plot_rect: (u32, u32, u32, u32),
    bounds: (f32, f32, f32, f32),
    color: [u8; 4],
) -> ScreenVertex {
    let (x_min, x_max, y_min, y_max) = bounds;
    let xr = (x_max - x_min).max(1e-6);
    let yr = (y_max - y_min).max(1e-6);
    let tx = ((pos.x - x_min) / xr).clamp(0.0, 1.0);
    let ty = ((pos.y - y_min) / yr).clamp(0.0, 1.0);
    let sx = plot_rect.0 as f32 + tx * plot_rect.2.max(1) as f32;
    let sy = plot_rect.1 as f32 + (1.0 - ty) * plot_rect.3.max(1) as f32;
    ScreenVertex {
        x: sx,
        y: sy,
        z: 0.0,
        color,
    }
}

fn project_3d(
    pos: Vec3,
    plot_rect: (u32, u32, u32, u32),
    camera: &Camera,
    color: [u8; 4],
) -> Option<ScreenVertex> {
    let mut cam = camera.clone();
    cam.update_aspect_ratio((plot_rect.2.max(1) as f32) / (plot_rect.3.max(1) as f32));
    let vp = cam.view_proj_matrix();
    let clip = vp * pos.extend(1.0);
    if clip.w.abs() <= 1e-6 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.z < -1.2 || ndc.z > 1.2 {
        return None;
    }
    let sx = plot_rect.0 as f32 + (ndc.x * 0.5 + 0.5) * plot_rect.2.max(1) as f32;
    let sy = plot_rect.1 as f32 + (1.0 - (ndc.y * 0.5 + 0.5)) * plot_rect.3.max(1) as f32;
    let depth = ndc.z * 0.5 + 0.5;
    Some(ScreenVertex {
        x: sx,
        y: sy,
        z: depth,
        color,
    })
}

fn axes_has_3d_content(figure: &Figure, axes_index: usize) -> bool {
    figure
        .plots()
        .zip(figure.plot_axes_indices().iter().copied())
        .any(|(plot, idx)| {
            idx == axes_index
                && match plot {
                    PlotElement::Surface(surface) => !surface.image_mode,
                    PlotElement::Patch(patch) => {
                        patch.force_3d() || patch.vertices().iter().any(|p| p.z.abs() > 1e-6)
                    }
                    PlotElement::Line3(_) | PlotElement::Scatter3(_) => true,
                    _ => false,
                }
        })
}

fn choose_axes_bounds(
    figure: &Figure,
    axes_index: usize,
    render_data: &[(usize, RenderData)],
) -> (f32, f32, f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for (ax, rd) in render_data {
        if *ax != axes_index {
            continue;
        }
        if let Some(bounds) = rd.bounds {
            min_x = min_x.min(bounds.min.x);
            max_x = max_x.max(bounds.max.x);
            min_y = min_y.min(bounds.min.y);
            max_y = max_y.max(bounds.max.y);
        }
    }

    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        min_x = -1.0;
        max_x = 1.0;
        min_y = -1.0;
        max_y = 1.0;
    }

    if let Some(meta) = figure.axes_metadata(axes_index) {
        if let Some((l, r)) = meta.x_limits {
            min_x = l as f32;
            max_x = r as f32;
        }
        if let Some((b, t)) = meta.y_limits {
            min_y = b as f32;
            max_y = t as f32;
        }
        if meta.axis_equal {
            let cx = (min_x + max_x) * 0.5;
            let cy = (min_y + max_y) * 0.5;
            let size = (max_x - min_x).abs().max((max_y - min_y).abs()).max(0.1);
            min_x = cx - size * 0.5;
            max_x = cx + size * 0.5;
            min_y = cy - size * 0.5;
            max_y = cy + size * 0.5;
        }
    }

    (min_x, max_x, min_y, max_y)
}

fn choose_axes_bounds_3d(
    axes_index: usize,
    render_data: &[(usize, RenderData)],
    bounds_2d: (f32, f32, f32, f32),
) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    for (ax, rd) in render_data {
        if *ax != axes_index {
            continue;
        }
        if let Some(bounds) = rd.bounds {
            min = min.min(bounds.min);
            max = max.max(bounds.max);
        }
    }

    if !min.x.is_finite() || !max.x.is_finite() {
        (
            Vec3::new(bounds_2d.0, bounds_2d.2, -1.0),
            Vec3::new(bounds_2d.1, bounds_2d.3, 1.0),
        )
    } else {
        if (max.z - min.z).abs() < 1e-6 {
            min.z -= 0.5;
            max.z += 0.5;
        }
        (min, max)
    }
}

fn default_3d_camera_for_bounds(min: Vec3, max: Vec3) -> Camera {
    let center = (min + max) * 0.5;
    let extent = (max - min).abs();
    let radius = extent.length().max(1e-3) * 0.5;

    let mut cam = Camera::new();
    let fov = match cam.projection {
        ProjectionType::Perspective { fov, .. } => fov.max(0.2),
        _ => 45.0f32.to_radians(),
    };
    let distance = (radius / (fov * 0.5).tan()).max(radius * 2.5) * 1.05;

    let dir = Vec3::new(1.0, -1.0, 0.8).normalize_or_zero();
    cam.target = center;
    cam.position = center + dir * distance;
    cam.up = Vec3::Z;
    cam
}

fn choose_axes_camera(
    figure: &Figure,
    axes_index: usize,
    axes_cameras: Option<&[Camera]>,
    min: Vec3,
    max: Vec3,
) -> Camera {
    if let Some(cams) = axes_cameras {
        if let Some(cam) = cams.get(axes_index) {
            return cam.clone();
        }
    }

    let mut cam = default_3d_camera_for_bounds(min, max);

    if let Some(meta) = figure.axes_metadata(axes_index) {
        if let (Some(az), Some(el)) = (meta.view_azimuth_deg, meta.view_elevation_deg) {
            cam.set_view_angles_deg(az, el);
        }
    }

    cam
}

fn get_axes_title_and_labels(
    figure: &Figure,
    axes_index: usize,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    let meta = figure.axes_metadata(axes_index);
    let title = meta
        .and_then(|m| m.title.as_ref())
        .or(figure.title.as_ref())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let x_label = meta
        .and_then(|m| m.x_label.as_ref())
        .or(figure.x_label.as_ref())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let y_label = meta
        .and_then(|m| m.y_label.as_ref())
        .or(figure.y_label.as_ref())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let z_label = meta
        .and_then(|m| m.z_label.as_ref())
        .or(figure.z_label.as_ref())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    (title, x_label, y_label, z_label)
}

fn text_scale_from_font_size(font_size: Option<f32>, default_scale: u32) -> u32 {
    let base = font_size.unwrap_or((default_scale.max(1) * 8) as f32);
    ((base / 8.0).round() as i32).clamp(1, 4) as u32
}

fn get_axes_style_and_display_prefs(
    figure: &Figure,
    axes_index: usize,
) -> (u32, u32, u32, bool, bool) {
    let Some(meta) = figure.axes_metadata(axes_index) else {
        return (2, 2, 1, figure.grid_enabled, figure.box_enabled);
    };

    let title_scale = text_scale_from_font_size(meta.title_style.font_size, 2);
    let label_font = meta
        .x_label_style
        .font_size
        .or(meta.y_label_style.font_size)
        .or(meta.z_label_style.font_size);
    let label_scale = text_scale_from_font_size(label_font, 2);
    let tick_scale = text_scale_from_font_size(label_font.map(|s| (s - 2.0).max(8.0)), 1);

    (
        title_scale,
        label_scale,
        tick_scale,
        meta.grid_enabled,
        meta.box_enabled,
    )
}

fn project_vertex(vertex: &crate::core::Vertex, axes: &AxesView) -> Option<ScreenVertex> {
    let pos = Vec3::from_array(vertex.position);
    let color = to_u8_rgba(vertex.color);
    if axes.has_3d_content {
        axes.camera_3d
            .as_ref()
            .and_then(|camera| project_3d(pos, axes.plot_rect, camera, color))
    } else {
        Some(project_2d(pos, axes.plot_rect, axes.bounds_2d, color))
    }
}

fn draw_bitmap_text(canvas: &mut Canvas, x: i32, y: i32, text: &str, scale: u32, color: [u8; 4]) {
    let mut cursor_x = x;
    let sc = scale.max(1) as i32;
    let fallback = BASIC_FONTS.get('?').unwrap_or([0u8; 8]);

    for ch in text.chars() {
        let glyph = BASIC_FONTS
            .get(ch)
            .or_else(|| BASIC_FONTS.get(' '))
            .unwrap_or(fallback);
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..8i32 {
                if ((bits >> col) & 1) == 0 {
                    continue;
                }
                for sy in 0..sc {
                    for sx in 0..sc {
                        canvas.blend_pixel(
                            cursor_x + col * sc + sx,
                            y + row as i32 * sc + sy,
                            color,
                            0.0,
                            false,
                        );
                    }
                }
            }
        }
        cursor_x += 8 * sc + sc;
    }
}

fn draw_text_centered(
    canvas: &mut Canvas,
    center_x: i32,
    y: i32,
    text: &str,
    scale: u32,
    color: [u8; 4],
) {
    let sc = scale.max(1) as i32;
    let text_w = (text.chars().count() as i32) * (8 * sc + sc);
    let x = center_x - text_w / 2;
    draw_bitmap_text(canvas, x, y, text, scale, color);
}

fn format_tick(v: f32) -> String {
    if !v.is_finite() {
        return "nan".to_string();
    }
    let abs = v.abs();
    if abs >= 1000.0 || (abs > 0.0 && abs < 0.01) {
        format!("{v:.2e}")
    } else {
        format!("{v:.3}")
    }
}

fn draw_2d_axes_decorations(canvas: &mut Canvas, axes: &AxesView) {
    let frame_color = [162, 170, 184, 255];
    let grid_color = [104, 114, 130, 110];
    let text_color = [212, 220, 234, 255];

    let (px, py, pw, ph) = axes.plot_rect;
    let left = px as i32;
    let right = (px + pw.saturating_sub(1)) as i32;
    let top = py as i32;
    let bottom = (py + ph.saturating_sub(1)) as i32;

    if axes.show_grid {
        for i in 0..=6 {
            let t = i as f32 / 6.0;
            let x = (left as f32 + t * (right - left) as f32).round() as i32;
            let y = (top as f32 + t * (bottom - top) as f32).round() as i32;

            let gv = ScreenVertex {
                x: x as f32,
                y: top as f32,
                z: 0.0,
                color: grid_color,
            };
            let gv2 = ScreenVertex {
                x: x as f32,
                y: bottom as f32,
                z: 0.0,
                color: grid_color,
            };
            canvas.draw_line(gv, gv2, 1.0, 0, false);

            let gh = ScreenVertex {
                x: left as f32,
                y: y as f32,
                z: 0.0,
                color: grid_color,
            };
            let gh2 = ScreenVertex {
                x: right as f32,
                y: y as f32,
                z: 0.0,
                color: grid_color,
            };
            canvas.draw_line(gh, gh2, 1.0, 0, false);
        }
    }

    if axes.show_box {
        let corners = [
            (left as f32, top as f32),
            (right as f32, top as f32),
            (right as f32, bottom as f32),
            (left as f32, bottom as f32),
        ];
        for i in 0..4 {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            canvas.draw_line(
                ScreenVertex {
                    x: a.0,
                    y: a.1,
                    z: 0.0,
                    color: frame_color,
                },
                ScreenVertex {
                    x: b.0,
                    y: b.1,
                    z: 0.0,
                    color: frame_color,
                },
                1.2,
                0,
                false,
            );
        }
    }

    let (x_min, x_max, y_min, y_max) = axes.bounds_2d;
    let tick_sc = axes.tick_scale as i32;
    for i in 0..=4 {
        let t = i as f32 / 4.0;
        let x = (left as f32 + t * (right - left) as f32).round() as i32;
        let y = (top as f32 + t * (bottom - top) as f32).round() as i32;

        let xv = x_min + t * (x_max - x_min);
        let yv = y_max - t * (y_max - y_min);

        draw_bitmap_text(
            canvas,
            x - 12 * tick_sc,
            bottom + 6 + tick_sc,
            &format_tick(xv),
            axes.tick_scale,
            with_alpha(text_color, 0.9),
        );
        draw_bitmap_text(
            canvas,
            left - 56 * tick_sc,
            y - 4 * tick_sc,
            &format_tick(yv),
            axes.tick_scale,
            with_alpha(text_color, 0.9),
        );
    }

    if let Some(title) = &axes.title {
        draw_text_centered(
            canvas,
            (axes.viewport.0 + axes.viewport.2 / 2) as i32,
            axes.viewport.1 as i32 + 6,
            title,
            axes.title_scale,
            text_color,
        );
    }
    if let Some(x_label) = &axes.x_label {
        let label_sc = axes.label_scale as i32;
        draw_text_centered(
            canvas,
            (axes.viewport.0 + axes.viewport.2 / 2) as i32,
            (axes.viewport.1 + axes.viewport.3).saturating_sub((12 + 10 * label_sc) as u32) as i32,
            x_label,
            axes.label_scale,
            text_color,
        );
    }
    if let Some(y_label) = &axes.y_label {
        let label_sc = axes.label_scale as i32;
        draw_bitmap_text(
            canvas,
            axes.viewport.0 as i32 + 6,
            (axes.viewport.1 + axes.viewport.3 / 2).saturating_sub((8 * label_sc) as u32) as i32,
            y_label,
            axes.label_scale,
            text_color,
        );
    }
}

fn draw_3d_axes_decorations(canvas: &mut Canvas, axes: &AxesView) {
    let floor_grid_minor = [44, 54, 70, 68];
    let axis_x_color = [235, 80, 80, 230];
    let axis_y_color = [90, 220, 120, 230];
    let axis_z_color = [90, 160, 255, 230];
    let text_color = [212, 220, 234, 255];

    let (bmin, bmax) = axes.bounds_3d;
    let Some(cam) = axes.camera_3d.as_ref() else {
        return;
    };

    let origin_component = |lo: f32, hi: f32| -> f32 {
        if lo <= 0.0 && hi >= 0.0 {
            0.0
        } else {
            lo
        }
    };
    let ox = origin_component(bmin.x, bmax.x);
    let oy = origin_component(bmin.y, bmax.y);
    let oz = origin_component(bmin.z, bmax.z);
    let floor_z = oz;

    if axes.show_grid {
        let divisions = 28usize;
        for i in 0..=divisions {
            let t = i as f32 / divisions as f32;
            let x = bmin.x + t * (bmax.x - bmin.x);
            let y = bmin.y + t * (bmax.y - bmin.y);

            let gx0 = Vec3::new(x, bmin.y, floor_z);
            let gx1 = Vec3::new(x, bmax.y, floor_z);
            let gy0 = Vec3::new(bmin.x, y, floor_z);
            let gy1 = Vec3::new(bmax.x, y, floor_z);

            let Some(a0) = project_3d(gx0, axes.plot_rect, cam, floor_grid_minor) else {
                continue;
            };
            let Some(a1) = project_3d(gx1, axes.plot_rect, cam, floor_grid_minor) else {
                continue;
            };
            canvas.draw_line(a0, a1, 0.9, 0, false);

            let Some(b0) = project_3d(gy0, axes.plot_rect, cam, floor_grid_minor) else {
                continue;
            };
            let Some(b1) = project_3d(gy1, axes.plot_rect, cam, floor_grid_minor) else {
                continue;
            };
            canvas.draw_line(b0, b1, 0.9, 0, false);
        }
    }

    let x_end = if bmax.x >= ox {
        Vec3::new(bmax.x, oy, floor_z)
    } else {
        Vec3::new(bmin.x, oy, floor_z)
    };
    let y_end = if bmax.y >= oy {
        Vec3::new(ox, bmax.y, floor_z)
    } else {
        Vec3::new(ox, bmin.y, floor_z)
    };
    let z_end = if bmax.z >= oz {
        Vec3::new(ox, oy, bmax.z)
    } else {
        Vec3::new(ox, oy, bmin.z)
    };
    let origin = Vec3::new(ox, oy, oz);

    if let (Some(o), Some(xp)) = (
        project_3d(origin, axes.plot_rect, cam, axis_x_color),
        project_3d(x_end, axes.plot_rect, cam, axis_x_color),
    ) {
        canvas.draw_line(o, xp, 1.8, 0, false);
        draw_bitmap_text(
            canvas,
            xp.x as i32 + 6,
            xp.y as i32 + 2,
            axes.x_label.as_deref().unwrap_or("x"),
            axes.label_scale,
            axis_x_color,
        );
    }
    if let (Some(o), Some(yp)) = (
        project_3d(origin, axes.plot_rect, cam, axis_y_color),
        project_3d(y_end, axes.plot_rect, cam, axis_y_color),
    ) {
        canvas.draw_line(o, yp, 1.8, 0, false);
        draw_bitmap_text(
            canvas,
            yp.x as i32 + 6,
            yp.y as i32 + 2,
            axes.y_label.as_deref().unwrap_or("y"),
            axes.label_scale,
            axis_y_color,
        );
    }
    if let (Some(o), Some(zp)) = (
        project_3d(origin, axes.plot_rect, cam, axis_z_color),
        project_3d(z_end, axes.plot_rect, cam, axis_z_color),
    ) {
        canvas.draw_line(o, zp, 1.8, 0, false);
        draw_bitmap_text(
            canvas,
            zp.x as i32 + 6,
            zp.y as i32 + 2,
            axes.z_label.as_deref().unwrap_or("z"),
            axes.label_scale,
            axis_z_color,
        );
    }

    if let Some(title) = &axes.title {
        draw_text_centered(
            canvas,
            (axes.viewport.0 + axes.viewport.2 / 2) as i32,
            axes.viewport.1 as i32 + 6,
            title,
            axes.title_scale,
            text_color,
        );
    }
}

fn draw_3d_orientation_gizmo(canvas: &mut Canvas, axes: &AxesView) {
    let Some(cam) = axes.camera_3d.as_ref() else {
        return;
    };
    let forward = (cam.target - cam.position).normalize_or_zero();
    if forward.length_squared() < 1e-9 {
        return;
    }
    let world_up = cam.up.normalize_or_zero();
    let right = forward.cross(world_up).normalize_or_zero();
    if right.length_squared() < 1e-9 {
        return;
    }
    let up = right.cross(forward).normalize_or_zero();
    if up.length_squared() < 1e-9 {
        return;
    }

    #[derive(Clone, Copy)]
    struct AxisItem {
        label: &'static str,
        dir_world: Vec3,
        color: [u8; 4],
        z_sort: f32,
    }

    let mut axis_items = [
        AxisItem {
            label: "X",
            dir_world: Vec3::X,
            color: [235, 80, 80, 255],
            z_sort: 0.0,
        },
        AxisItem {
            label: "Y",
            dir_world: Vec3::Y,
            color: [90, 220, 120, 255],
            z_sort: 0.0,
        },
        AxisItem {
            label: "Z",
            dir_world: Vec3::Z,
            color: [90, 160, 255, 255],
            z_sort: 0.0,
        },
    ];

    for a in &mut axis_items {
        let x = a.dir_world.dot(right);
        let y = a.dir_world.dot(up);
        let z = a.dir_world.dot(-forward);
        a.z_sort = z;
        a.dir_world = Vec3::new(x, y, z);
    }
    axis_items.sort_by(|a, b| a.z_sort.total_cmp(&b.z_sort));

    let scale = ((axes.viewport.2.min(axes.viewport.3) as f32) / 720.0).clamp(0.8, 1.6);
    let gizmo_size =
        ((axes.viewport.2.min(axes.viewport.3) as f32) * 0.16).clamp(44.0, 110.0) * scale;
    let pad = (30.0 * scale).round() as i32;
    let origin = Vec2::new(
        (axes.viewport.0 as i32 + pad) as f32,
        ((axes.viewport.1 + axes.viewport.3) as i32 - pad) as f32,
    );
    canvas.draw_disc(
        origin,
        (2.0 * scale).max(1.0),
        [210, 214, 224, 255],
        0.0,
        false,
    );

    let axis_len = gizmo_size * 0.65;
    let head_len = (8.0 * scale).min(axis_len * 0.35);
    let head_w = 5.0 * scale;
    for a in &axis_items {
        let dir2 = Vec2::new(a.dir_world.x, -a.dir_world.y);
        let mag = dir2.length();
        if !mag.is_finite() || mag < 1e-4 {
            continue;
        }
        let d = dir2 / mag;
        let end = origin + d * axis_len;
        canvas.draw_line(
            ScreenVertex {
                x: origin.x,
                y: origin.y,
                z: 0.0,
                color: a.color,
            },
            ScreenVertex {
                x: end.x,
                y: end.y,
                z: 0.0,
                color: a.color,
            },
            (2.0 * scale).max(1.2),
            0,
            false,
        );

        let base = end - d * head_len;
        let perp = Vec2::new(-d.y, d.x);
        canvas.draw_line(
            ScreenVertex {
                x: end.x,
                y: end.y,
                z: 0.0,
                color: a.color,
            },
            ScreenVertex {
                x: (base + perp * head_w).x,
                y: (base + perp * head_w).y,
                z: 0.0,
                color: a.color,
            },
            (2.0 * scale).max(1.2),
            0,
            false,
        );
        canvas.draw_line(
            ScreenVertex {
                x: end.x,
                y: end.y,
                z: 0.0,
                color: a.color,
            },
            ScreenVertex {
                x: (base - perp * head_w).x,
                y: (base - perp * head_w).y,
                z: 0.0,
                color: a.color,
            },
            (2.0 * scale).max(1.2),
            0,
            false,
        );

        let label_pos = end + d * (10.0 * scale);
        draw_bitmap_text(
            canvas,
            label_pos.x as i32 - 3,
            label_pos.y as i32 - 3,
            a.label,
            1,
            a.color,
        );
    }
}

fn draw_legend_for_axes(canvas: &mut Canvas, figure: &Figure, axes: &AxesView) {
    if !figure.legend_enabled {
        return;
    }
    let entries = figure.legend_entries();
    if entries.is_empty() {
        return;
    }

    let max_entries = entries.len().min(8);
    let pad = 10i32;
    let row_h = 20i32;
    let legend_w = ((axes.viewport.2 as f32 * 0.30).clamp(92.0, 148.0)).round() as i32;
    let legend_h = row_h * max_entries as i32 + 10;
    let x = (axes.viewport.0 + axes.viewport.2) as i32 - legend_w - pad;
    let y = axes.viewport.1 as i32 + 12;

    canvas.fill_rect(x, y, legend_w, legend_h, [8, 14, 24, 220]);
    canvas.stroke_rect(x, y, legend_w, legend_h, [36, 52, 74, 245], 1.0);

    for (i, entry) in entries.into_iter().take(max_entries).enumerate() {
        let yy = y + 6 + i as i32 * row_h + row_h / 2;
        let swatch_x0 = x + 10;
        let swatch_x1 = swatch_x0 + 18;
        let swatch_color = to_u8_rgba(entry.color.to_array());
        canvas.draw_line(
            ScreenVertex {
                x: swatch_x0 as f32,
                y: yy as f32,
                z: 0.0,
                color: swatch_color,
            },
            ScreenVertex {
                x: swatch_x1 as f32,
                y: yy as f32,
                z: 0.0,
                color: swatch_color,
            },
            2.0,
            0,
            false,
        );

        let label = if entry.label.is_empty() {
            "Series".to_string()
        } else {
            entry.label
        };
        draw_bitmap_text(canvas, x + 34, yy - 4, &label, 1, [220, 228, 239, 255]);
    }
}

pub async fn render_figure_rgba_bytes(
    mut figure: Figure,
    width: u32,
    height: u32,
    theme: Option<PlotThemeConfig>,
    camera: Option<&Camera>,
    axes_cameras: Option<&[Camera]>,
    _textmark: Option<&str>,
) -> Result<Vec<u8>, String> {
    let width = width.max(1);
    let height = height.max(1);
    let bg = if is_default_figure_bg(figure.background_color) {
        theme
            .as_ref()
            .map(|cfg| cfg.build_theme().get_background_color())
            .unwrap_or_else(|| Vec4::new(1.0, 1.0, 1.0, 1.0))
    } else {
        figure.background_color
    };
    let mut canvas = Canvas::new(width, height, to_u8_rgba(bg.to_array()));

    let (rows, cols) = figure.axes_grid();
    let viewports = compute_tiled_viewports(width, height, rows.max(1), cols.max(1));
    let axes_count = rows.max(1) * cols.max(1);

    let has_3d_flags: Vec<bool> = (0..axes_count)
        .map(|axes_index| axes_has_3d_content(&figure, axes_index))
        .collect();
    let axes_sizes: Vec<(u32, u32)> = viewports
        .iter()
        .zip(has_3d_flags.iter())
        .map(|(vp, has_3d)| {
            let rect = compute_plot_rect(*vp, *has_3d);
            (rect.2.max(1), rect.3.max(1))
        })
        .collect();

    let render_items = figure.render_data_with_axes_with_viewport_and_gpu(
        Some((width, height)),
        Some(&axes_sizes),
        None,
        None,
    );

    let mut axes_views = Vec::with_capacity(axes_count);
    for axes_index in 0..axes_count {
        let has_3d = has_3d_flags[axes_index];
        let viewport = viewports[axes_index];
        let plot_rect = compute_plot_rect(viewport, has_3d);
        let bounds_2d = choose_axes_bounds(&figure, axes_index, &render_items);
        let (bmin, bmax) = choose_axes_bounds_3d(axes_index, &render_items, bounds_2d);
        let camera_3d = if has_3d {
            Some(if axes_count == 1 {
                camera.cloned().unwrap_or_else(|| {
                    choose_axes_camera(&figure, axes_index, axes_cameras, bmin, bmax)
                })
            } else {
                choose_axes_camera(&figure, axes_index, axes_cameras, bmin, bmax)
            })
        } else {
            None
        };

        let (title, x_label, y_label, z_label) = get_axes_title_and_labels(&figure, axes_index);
        let (title_scale, label_scale, tick_scale, show_grid, show_box) =
            get_axes_style_and_display_prefs(&figure, axes_index);

        axes_views.push(AxesView {
            viewport,
            plot_rect,
            bounds_2d,
            bounds_3d: (bmin, bmax),
            camera_3d,
            has_3d_content: has_3d,
            title,
            x_label,
            y_label,
            z_label,
            title_scale,
            label_scale,
            tick_scale,
            show_grid,
            show_box,
        });
    }

    for axes in &axes_views {
        if axes.has_3d_content {
            draw_3d_axes_decorations(&mut canvas, axes);
        } else {
            draw_2d_axes_decorations(&mut canvas, axes);
        }
    }

    for (axes_index, rd) in render_items.iter() {
        if rd.vertices.is_empty() {
            continue;
        }
        let Some(axes) = axes_views.get(*axes_index) else {
            continue;
        };
        draw_render_data(&mut canvas, rd, axes);
    }
    for axes in &axes_views {
        if !axes.has_3d_content {
            continue;
        }
        draw_3d_orientation_gizmo(&mut canvas, axes);
        if axes_views.len() == 1 {
            draw_legend_for_axes(&mut canvas, &figure, axes);
        }
    }

    Ok(canvas.rgba())
}

fn draw_render_data(canvas: &mut Canvas, render_data: &RenderData, axes: &AxesView) {
    let width_px = render_data.material.roughness.max(1.0);
    let style_code = render_data.material.metallic as i32;

    match render_data.pipeline_type {
        PipelineType::Lines => {
            for segment in render_data.vertices.chunks_exact(2) {
                let Some(a) = project_vertex(&segment[0], axes) else {
                    continue;
                };
                let Some(b) = project_vertex(&segment[1], axes) else {
                    continue;
                };
                canvas.draw_line(a, b, width_px, style_code, axes.has_3d_content);
            }
        }
        PipelineType::Points | PipelineType::Scatter3 => {
            for v in &render_data.vertices {
                let Some(p) = project_vertex(v, axes) else {
                    continue;
                };
                let marker_radius = (v.normal[2].max(1.0) * 0.5).max(1.0);
                canvas.draw_disc(
                    Vec2::new(p.x, p.y),
                    marker_radius,
                    p.color,
                    p.z,
                    axes.has_3d_content,
                );
            }
        }
        PipelineType::Triangles => {
            if axes.has_3d_content && render_data.indices.is_none() {
                return;
            }
            if let Some(indices) = &render_data.indices {
                for tri in indices.chunks_exact(3) {
                    let (Some(v0), Some(v1), Some(v2)) = (
                        render_data.vertices.get(tri[0] as usize),
                        render_data.vertices.get(tri[1] as usize),
                        render_data.vertices.get(tri[2] as usize),
                    ) else {
                        continue;
                    };
                    let (Some(p0), Some(p1), Some(p2)) = (
                        project_vertex(v0, axes),
                        project_vertex(v1, axes),
                        project_vertex(v2, axes),
                    ) else {
                        continue;
                    };
                    canvas.fill_triangle(p0, p1, p2, axes.has_3d_content);
                }
            } else {
                for tri in render_data.vertices.chunks_exact(3) {
                    let (Some(p0), Some(p1), Some(p2)) = (
                        project_vertex(&tri[0], axes),
                        project_vertex(&tri[1], axes),
                        project_vertex(&tri[2], axes),
                    ) else {
                        continue;
                    };
                    canvas.fill_triangle(p0, p1, p2, axes.has_3d_content);
                }
            }
        }
        PipelineType::Textured => {
            for tri in render_data.vertices.chunks_exact(3) {
                let (Some(p0), Some(p1), Some(p2)) = (
                    project_vertex(&tri[0], axes),
                    project_vertex(&tri[1], axes),
                    project_vertex(&tri[2], axes),
                ) else {
                    continue;
                };
                canvas.fill_triangle(p0, p1, p2, axes.has_3d_content);
            }
        }
    }
}

pub fn encode_png_bytes(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    use image::{ImageBuffer, ImageFormat, Rgba};

    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width.max(1), height.max(1), rgba.to_vec())
        .ok_or_else(|| "Failed to create image buffer for CPU PNG encoding".to_string())?;
    let mut out = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut out, ImageFormat::Png)
        .map_err(|err| format!("Failed to encode CPU PNG bytes: {err}"))?;
    Ok(out.into_inner())
}

pub async fn render_figure_png_bytes(
    figure: Figure,
    width: u32,
    height: u32,
    theme: Option<PlotThemeConfig>,
    camera: Option<&Camera>,
    axes_cameras: Option<&[Camera]>,
    textmark: Option<&str>,
) -> Result<Vec<u8>, String> {
    let rgba =
        render_figure_rgba_bytes(figure, width, height, theme, camera, axes_cameras, textmark)
            .await?;
    encode_png_bytes(width.max(1), height.max(1), &rgba)
}
