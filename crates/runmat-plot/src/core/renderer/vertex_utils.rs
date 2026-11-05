//! Utility functions for creating common vertex patterns.

use super::*;
use glam::Vec2;

/// Create vertices for a line from start to end point
pub fn create_line(start: Vec3, end: Vec3, color: Vec4) -> Vec<Vertex> {
    vec![Vertex::new(start, color), Vertex::new(end, color)]
}

/// CPU polyline extrusion for thick lines (butt caps, miter joins simplified)
/// Input: contiguous points. Output: triangle list vertices.
pub fn extrude_polyline(points: &[Vec3], color: Vec4, width: f32) -> Vec<Vertex> {
    let mut out: Vec<Vertex> = Vec::new();
    if points.len() < 2 {
        return out;
    }
    for i in 0..points.len() - 1 {
        let [a, b, c, d] = extruded_segment_corners(points[i], points[i + 1], width);
        // Two triangles: a-b-c and a-c-d
        out.push(Vertex::new(a, color));
        out.push(Vertex::new(b, color));
        out.push(Vertex::new(c, color));
        out.push(Vertex::new(a, color));
        out.push(Vertex::new(c, color));
        out.push(Vertex::new(d, color));
    }
    out
}

/// Shared segment geometry for tessellation and allocation-free extent preparation.
/// Width is already in data units: never clamp it to a pixel-sized minimum.
pub(crate) fn extruded_segment_corners(p0: Vec3, p1: Vec3, width: f32) -> [Vec3; 4] {
    let half_w = width.max(0.0) * 0.5;
    let dir = (p1 - p0).truncate();
    let len = (dir.x * dir.x + dir.y * dir.y).sqrt().max(1e-6);
    let offset = Vec3::new(-dir.y / len * half_w, dir.x / len * half_w, 0.0);
    [p0 - offset, p0 + offset, p1 + offset, p1 - offset]
}

fn line_intersection(p: Vec2, r: Vec2, q: Vec2, s: Vec2) -> Option<Vec2> {
    let rxs = r.perp_dot(s);
    if rxs.abs() < 1e-6 {
        return None;
    }
    let t = (q - p).perp_dot(s) / rxs;
    Some(p + r * t)
}

/// Longest miter, as a multiple of half the line width, before a join is beveled instead
/// (the SVG and canvas default). Without it, a sharp turn in noisy data throws a spike far
/// past the data: a 2,000-point noise trace drew peaks 9 dB above its maximum.
const MITER_LIMIT: f32 = 4.0;

/// Extrude polyline with join styles at internal vertices.
pub fn extrude_polyline_with_join(
    points: &[Vec3],
    color: Vec4,
    width: f32,
    join: crate::plots::line::LineJoin,
) -> Vec<Vertex> {
    let mut out: Vec<Vertex> = Vec::new();
    if points.len() < 2 {
        return out;
    }
    // See `extrude_polyline` for rationale: keep width in data-space units.
    let half_w = width.max(0.0) * 0.5;
    // Base quads
    out.extend(extrude_polyline(points, color, width));

    // Joins
    for i in 1..points.len() - 1 {
        let p_prev = points[i - 1];
        let p = points[i];
        let p_next = points[i + 1];
        let d0 = (p - p_prev).truncate();
        let d1 = (p_next - p).truncate();
        let l0 = d0.length().max(1e-6);
        let l1 = d1.length().max(1e-6);
        let n0 = Vec2::new(-d0.y / l0, d0.x / l0);
        let n1 = Vec2::new(-d1.y / l1, d1.x / l1);
        let turn = d0.perp_dot(d1); // >0 left turn, <0 right turn

        if turn > 1e-6 {
            // Left turn: outer side is left (use +n)
            let left0 = p.truncate() + n0 * half_w;
            let left1 = p.truncate() + n1 * half_w;
            match join {
                crate::plots::line::LineJoin::Bevel => {
                    // Triangle wedge (p, left0, left1)
                    out.push(Vertex::new(p, color));
                    out.push(Vertex::new(left0.extend(0.0), color));
                    out.push(Vertex::new(left1.extend(0.0), color));
                }
                crate::plots::line::LineJoin::Miter => {
                    let dir_edge0 = (p.truncate() - p_prev.truncate()).normalize_or_zero();
                    let dir_edge1 = (p_next.truncate() - p.truncate()).normalize_or_zero();
                    let l_edge = line_intersection(left0, dir_edge0, left1, dir_edge1)
                        .filter(|m| m.distance(p.truncate()) <= MITER_LIMIT * half_w);
                    if let Some(miter) = l_edge {
                        // fill wedge left0-miter-left1
                        out.push(Vertex::new(left0.extend(0.0), color));
                        out.push(Vertex::new(miter.extend(0.0), color));
                        out.push(Vertex::new(left1.extend(0.0), color));
                    } else {
                        // fallback to bevel
                        out.push(Vertex::new(p, color));
                        out.push(Vertex::new(left0.extend(0.0), color));
                        out.push(Vertex::new(left1.extend(0.0), color));
                    }
                }
                crate::plots::line::LineJoin::Round => {
                    // Arc fan from left0 -> left1 around p
                    let center = p.truncate();
                    let a0 = (left0 - center).to_array();
                    let a1 = (left1 - center).to_array();
                    let ang0 = a0[1].atan2(a0[0]);
                    let mut ang1 = a1[1].atan2(a1[0]);
                    // Ensure CCW sweep
                    if ang1 < ang0 {
                        ang1 += std::f32::consts::TAU;
                    }
                    let steps = 10usize;
                    let dtheta = (ang1 - ang0) / steps as f32;
                    let r = half_w;
                    for k in 0..steps {
                        let theta0 = ang0 + dtheta * k as f32;
                        let theta1 = ang0 + dtheta * (k + 1) as f32;
                        let v0 =
                            Vec2::new(center.x + theta0.cos() * r, center.y + theta0.sin() * r);
                        let v1 =
                            Vec2::new(center.x + theta1.cos() * r, center.y + theta1.sin() * r);
                        out.push(Vertex::new(p, color));
                        out.push(Vertex::new(v0.extend(0.0), color));
                        out.push(Vertex::new(v1.extend(0.0), color));
                    }
                }
            }
        } else if turn < -1e-6 {
            // Right turn: outer side is right (use -n)
            let right0 = p.truncate() - n0 * half_w;
            let right1 = p.truncate() - n1 * half_w;
            match join {
                crate::plots::line::LineJoin::Bevel => {
                    out.push(Vertex::new(p, color));
                    out.push(Vertex::new(right1.extend(0.0), color));
                    out.push(Vertex::new(right0.extend(0.0), color));
                }
                crate::plots::line::LineJoin::Miter => {
                    let dir_edge0 = (p.truncate() - p_prev.truncate()).normalize_or_zero();
                    let dir_edge1 = (p_next.truncate() - p.truncate()).normalize_or_zero();
                    let l_edge = line_intersection(right0, dir_edge0, right1, dir_edge1)
                        .filter(|m| m.distance(p.truncate()) <= MITER_LIMIT * half_w);
                    if let Some(miter) = l_edge {
                        out.push(Vertex::new(right1.extend(0.0), color));
                        out.push(Vertex::new(miter.extend(0.0), color));
                        out.push(Vertex::new(right0.extend(0.0), color));
                    } else {
                        out.push(Vertex::new(p, color));
                        out.push(Vertex::new(right1.extend(0.0), color));
                        out.push(Vertex::new(right0.extend(0.0), color));
                    }
                }
                crate::plots::line::LineJoin::Round => {
                    let center = p.truncate();
                    let a0 = (right0 - center).to_array();
                    let a1 = (right1 - center).to_array();
                    let mut ang0 = a0[1].atan2(a0[0]);
                    let mut ang1 = a1[1].atan2(a1[0]);
                    // Ensure CW sweep becomes CCW by swapping
                    if ang0 < ang1 {
                        std::mem::swap(&mut ang0, &mut ang1);
                    }
                    let steps = 10usize;
                    let dtheta = (ang0 - ang1) / steps as f32;
                    let r = half_w;
                    for k in 0..steps {
                        let theta0 = ang0 - dtheta * k as f32;
                        let theta1 = ang0 - dtheta * (k + 1) as f32;
                        let v0 =
                            Vec2::new(center.x + theta0.cos() * r, center.y + theta0.sin() * r);
                        let v1 =
                            Vec2::new(center.x + theta1.cos() * r, center.y + theta1.sin() * r);
                        out.push(Vertex::new(p, color));
                        out.push(Vertex::new(v0.extend(0.0), color));
                        out.push(Vertex::new(v1.extend(0.0), color));
                    }
                }
            }
        }
    }

    out
}

/// Create vertices for a triangle
pub fn create_triangle(p1: Vec3, p2: Vec3, p3: Vec3, color: Vec4) -> Vec<Vertex> {
    vec![
        Vertex::new(p1, color),
        Vertex::new(p2, color),
        Vertex::new(p3, color),
    ]
}

/// Create vertices for a point cloud
pub fn create_point_cloud(points: &[Vec3], colors: &[Vec4]) -> Vec<Vertex> {
    points
        .iter()
        .zip(colors.iter())
        .map(|(&pos, &color)| Vertex::new(pos, color))
        .collect()
}

/// Create vertices for a parametric line plot (1px line segments)
pub fn create_line_plot(x_data: &[f64], y_data: &[f64], color: Vec4) -> Vec<Vertex> {
    let mut vertices = Vec::new();

    for i in 1..x_data.len() {
        let start = Vec3::new(x_data[i - 1] as f32, y_data[i - 1] as f32, 0.0);
        let end = Vec3::new(x_data[i] as f32, y_data[i] as f32, 0.0);
        vertices.extend(create_line(start, end, color));
    }

    vertices
}

/// Create dashed/dotted line vertices by selectively including segments.
/// Approximation: pattern is applied per original segment index.
pub fn create_line_plot_dashed(
    x_data: &[f64],
    y_data: &[f64],
    color: Vec4,
    style: crate::plots::line::LineStyle,
) -> Vec<Vertex> {
    let mut vertices = Vec::new();
    for i in 1..x_data.len() {
        let include = match style {
            crate::plots::line::LineStyle::None => false,
            crate::plots::line::LineStyle::Solid => true,
            crate::plots::line::LineStyle::Dashed => (i % 4) < 2, // on,on,off,off
            crate::plots::line::LineStyle::Dotted => false,       // handled elsewhere as points
            crate::plots::line::LineStyle::DashDot => {
                let m = i % 6;
                m < 2 || m == 3 // on,on,off,on,off,off
            }
        };
        if include {
            let start = Vec3::new(x_data[i - 1] as f32, y_data[i - 1] as f32, 0.0);
            let end = Vec3::new(x_data[i] as f32, y_data[i] as f32, 0.0);
            vertices.extend(create_line(start, end, color));
        }
    }
    vertices
}

/// Create thick polyline as triangles (used when line width > 1)
pub fn create_thick_polyline(
    x_data: &[f64],
    y_data: &[f64],
    color: Vec4,
    width_px: f32,
) -> Vec<Vertex> {
    let mut pts: Vec<Vec3> = Vec::with_capacity(x_data.len());
    for i in 0..x_data.len() {
        pts.push(Vec3::new(x_data[i] as f32, y_data[i] as f32, 0.0));
    }
    extrude_polyline(&pts, color, width_px)
}

/// Thick polyline with join style
pub fn create_thick_polyline_with_join(
    x_data: &[f64],
    y_data: &[f64],
    color: Vec4,
    width_px: f32,
    join: crate::plots::line::LineJoin,
) -> Vec<Vertex> {
    let mut pts: Vec<Vec3> = Vec::with_capacity(x_data.len());
    for i in 0..x_data.len() {
        pts.push(Vec3::new(x_data[i] as f32, y_data[i] as f32, 0.0));
    }
    extrude_polyline_with_join(&pts, color, width_px, join)
}

/// Create dashed/dotted thick polyline by skipping segments in the extruder.
pub fn create_thick_polyline_dashed(
    x_data: &[f64],
    y_data: &[f64],
    color: Vec4,
    width_px: f32,
    style: crate::plots::line::LineStyle,
) -> Vec<Vertex> {
    let mut out: Vec<Vertex> = Vec::new();
    if x_data.len() < 2 {
        return out;
    }
    let pts: Vec<Vec3> = x_data
        .iter()
        .zip(y_data.iter())
        .map(|(&x, &y)| Vec3::new(x as f32, y as f32, 0.0))
        .collect();
    for i in 0..pts.len() - 1 {
        let include = match style {
            crate::plots::line::LineStyle::None => false,
            crate::plots::line::LineStyle::Solid => true,
            crate::plots::line::LineStyle::Dashed => (i % 4) < 2,
            crate::plots::line::LineStyle::Dotted => false,
            crate::plots::line::LineStyle::DashDot => {
                let m = i % 6;
                m < 2 || m == 3
            }
        };
        if include {
            let seg = [pts[i], pts[i + 1]];
            out.extend(extrude_polyline(&seg, color, width_px));
        }
    }
    out
}

/// Dashed, dotted and dash-dot strokes for a polyline in viewport pixels. The pattern
/// runs along the line's length, so dashes look the same however densely it is sampled.
pub fn create_thick_polyline_dashed_px(
    x_px: &[f64],
    y_px: &[f64],
    color: Vec4,
    width_px: f32,
    style: crate::plots::line::LineStyle,
) -> Vec<Vertex> {
    use crate::plots::line::LineStyle;
    let w = width_px.max(1.0);
    // (on, off) lengths in pixels, scaled with the line width.
    let pattern: Vec<f32> = match style {
        LineStyle::None => return Vec::new(),
        LineStyle::Solid => {
            return create_thick_polyline_with_join(
                x_px,
                y_px,
                color,
                width_px,
                crate::plots::line::LineJoin::Miter,
            )
        }
        LineStyle::Dashed => vec![6.0 + 2.0 * w, 4.0 + w],
        LineStyle::Dotted => vec![w, 2.0 + 1.5 * w],
        LineStyle::DashDot => vec![6.0 + 2.0 * w, 3.0 + w, w, 3.0 + w],
    };
    let mut out = Vec::new();
    let n = x_px.len().min(y_px.len());
    if n < 2 {
        return out;
    }
    let mut phase = 0usize;
    let mut remaining = pattern[0];
    let mut dash: Vec<Vec3> = vec![Vec3::new(x_px[0] as f32, y_px[0] as f32, 0.0)];
    let flush = |dash: &mut Vec<Vec3>, out: &mut Vec<Vertex>| {
        if dash.len() >= 2 {
            out.extend(extrude_polyline(dash, color, width_px));
        }
        dash.clear();
    };
    for i in 0..n - 1 {
        let a = Vec3::new(x_px[i] as f32, y_px[i] as f32, 0.0);
        let b = Vec3::new(x_px[i + 1] as f32, y_px[i + 1] as f32, 0.0);
        let length = (b - a).truncate().length();
        if !length.is_finite() || length <= 0.0 {
            continue;
        }
        let mut t = 0.0f32;
        while t < length {
            let step = remaining.min(length - t);
            t += step;
            remaining -= step;
            let p = a + (b - a) * (t / length);
            if phase.is_multiple_of(2) {
                dash.push(p);
            }
            if remaining <= 1.0e-4 {
                if phase.is_multiple_of(2) {
                    flush(&mut dash, &mut out);
                }
                phase = (phase + 1) % pattern.len();
                remaining = pattern[phase];
                if phase.is_multiple_of(2) {
                    dash.push(p);
                }
            }
        }
    }
    if phase.is_multiple_of(2) {
        flush(&mut dash, &mut out);
    }
    out
}

/// Square caps variant: extend endpoints by half width
pub fn create_thick_polyline_square_caps(
    x_data: &[f64],
    y_data: &[f64],
    color: Vec4,
    width_px: f32,
) -> Vec<Vertex> {
    if x_data.len() < 2 {
        return Vec::new();
    }
    let mut pts: Vec<Vec3> = Vec::with_capacity(x_data.len());
    for i in 0..x_data.len() {
        pts.push(Vec3::new(x_data[i] as f32, y_data[i] as f32, 0.0));
    }
    // extend start
    let dir0 = (pts[1] - pts[0]).truncate();
    let len0 = (dir0.x * dir0.x + dir0.y * dir0.y).sqrt().max(1e-6);
    let ext0 = Vec3::new(
        -(dir0.x / len0) * (width_px * 0.5),
        -(dir0.y / len0) * (width_px * 0.5),
        0.0,
    );
    pts[0] += ext0;
    // extend end
    let n = pts.len();
    let dir1 = (pts[n - 1] - pts[n - 2]).truncate();
    let len1 = (dir1.x * dir1.x + dir1.y * dir1.y).sqrt().max(1e-6);
    let ext1 = Vec3::new(
        (dir1.x / len1) * (width_px * 0.5),
        (dir1.y / len1) * (width_px * 0.5),
        0.0,
    );
    pts[n - 1] += ext1;
    extrude_polyline(&pts, color, width_px)
}

/// Round caps variant: square caps plus approximated semicircle fan at ends
pub fn create_thick_polyline_round_caps(
    x_data: &[f64],
    y_data: &[f64],
    color: Vec4,
    width_px: f32,
    segments: usize,
) -> Vec<Vertex> {
    let mut base = create_thick_polyline_square_caps(x_data, y_data, color, width_px);
    if x_data.len() < 2 {
        return base;
    }
    let r = width_px * 0.5;
    // start fan
    let p0 = Vec3::new(x_data[0] as f32, y_data[0] as f32, 0.0);
    let p1 = Vec3::new(x_data[1] as f32, y_data[1] as f32, 0.0);
    let dir0 = (p1 - p0).truncate();
    let theta0 = dir0.y.atan2(dir0.x) + std::f32::consts::PI; // facing backward
    for i in 0..segments {
        let a0 = theta0 - std::f32::consts::PI * (i as f32 / segments as f32);
        let a1 = theta0 - std::f32::consts::PI * ((i + 1) as f32 / segments as f32);
        let v0 = Vec3::new(p0.x + a0.cos() * r, p0.y + a0.sin() * r, 0.0);
        let v1 = Vec3::new(p0.x + a1.cos() * r, p0.y + a1.sin() * r, 0.0);
        base.push(Vertex::new(p0, color));
        base.push(Vertex::new(v0, color));
        base.push(Vertex::new(v1, color));
    }
    // end fan
    let n = x_data.len();
    let q0 = Vec3::new(x_data[n - 2] as f32, y_data[n - 2] as f32, 0.0);
    let q1 = Vec3::new(x_data[n - 1] as f32, y_data[n - 1] as f32, 0.0);
    let dir1 = (q1 - q0).truncate();
    let theta1 = dir1.y.atan2(dir1.x);
    let center = q1;
    for i in 0..segments {
        let a0 = theta1 - std::f32::consts::PI * (i as f32 / segments as f32);
        let a1 = theta1 - std::f32::consts::PI * ((i + 1) as f32 / segments as f32);
        let v0 = Vec3::new(center.x + a0.cos() * r, center.y + a0.sin() * r, 0.0);
        let v1 = Vec3::new(center.x + a1.cos() * r, center.y + a1.sin() * r, 0.0);
        base.push(Vertex::new(center, color));
        base.push(Vertex::new(v0, color));
        base.push(Vertex::new(v1, color));
    }
    base
}

/// Create vertices for a scatter plot
pub fn create_scatter_plot(x_data: &[f64], y_data: &[f64], color: Vec4) -> Vec<Vertex> {
    x_data
        .iter()
        .zip(y_data.iter())
        .map(|(&x, &y)| Vertex::new(Vec3::new(x as f32, y as f32, 0.0), color))
        .collect()
}
