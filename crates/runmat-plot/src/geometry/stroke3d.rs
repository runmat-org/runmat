use crate::core::Vertex;
use crate::plots::line::LineStyle;
use glam::{Vec3, Vec4};

const EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeCap3D {
    Butt,
    Square,
}

#[derive(Debug, Clone, Copy)]
pub struct StrokeStyle3D {
    pub half_width_data: f32,
    pub line_style: LineStyle,
    pub cap: StrokeCap3D,
}

impl StrokeStyle3D {
    pub fn new(half_width_data: f32, line_style: LineStyle, cap: StrokeCap3D) -> Self {
        Self {
            half_width_data,
            line_style,
            cap,
        }
    }
}

#[inline]
pub fn line_style_includes_segment(segment: usize, style: LineStyle) -> bool {
    match style {
        LineStyle::Solid => true,
        LineStyle::Dashed => (segment % 4) < 2,
        LineStyle::Dotted => segment.is_multiple_of(4),
        LineStyle::DashDot => {
            let m = segment % 6;
            m < 2 || m == 3
        }
    }
}

pub fn create_line_vertices_dashed(points: &[Vec3], color: Vec4, style: LineStyle) -> Vec<Vertex> {
    let mut out = Vec::new();
    if points.len() < 2 {
        return out;
    }
    for i in 0..points.len() - 1 {
        if !line_style_includes_segment(i, style) {
            continue;
        }
        let a = points[i];
        let b = points[i + 1];
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        if (b - a).length_squared() <= EPS {
            continue;
        }
        out.push(Vertex::new(a, color));
        out.push(Vertex::new(b, color));
    }
    out
}

pub fn tessellate_polyline(points: &[Vec3], color: Vec4, style: StrokeStyle3D) -> Vec<Vertex> {
    let mut out = Vec::new();
    if points.len() < 2 {
        return out;
    }

    let half_width = style.half_width_data.max(1e-6);
    let sides: Vec<Vec3> = (0..points.len()).map(|i| side_at(points, i)).collect();

    for i in 0..points.len() - 1 {
        if !line_style_includes_segment(i, style.line_style) {
            continue;
        }
        let start = points[i];
        let end = points[i + 1];
        if !start.is_finite() || !end.is_finite() {
            continue;
        }
        let dir = end - start;
        let len = dir.length();
        if len <= EPS {
            continue;
        }
        let dir_n = dir / len;
        let mut a = start;
        let mut b = end;
        if style.cap == StrokeCap3D::Square {
            if i == 0 {
                a -= dir_n * half_width;
            }
            if i == points.len() - 2 {
                b += dir_n * half_width;
            }
        }

        let sa = sides[i] * half_width;
        let sb = sides[i + 1] * half_width;
        let v0 = a + sa;
        let v1 = b + sb;
        let v2 = b - sb;
        let v3 = a - sa;

        out.push(Vertex::new(v0, color));
        out.push(Vertex::new(v1, color));
        out.push(Vertex::new(v2, color));
        out.push(Vertex::new(v0, color));
        out.push(Vertex::new(v2, color));
        out.push(Vertex::new(v3, color));
    }

    out
}

pub fn tessellate_polyline_tube(
    points: &[Vec3],
    color: Vec4,
    style: StrokeStyle3D,
    radial_segments: usize,
) -> Vec<Vertex> {
    let mut out = Vec::new();
    if points.len() < 2 {
        return out;
    }

    let half_width = style.half_width_data.max(1e-6);
    let radial_segments = radial_segments.max(3);
    let mut seg = 0usize;
    while seg + 1 < points.len() {
        if !line_style_includes_segment(seg, style.line_style)
            || !is_valid_segment(points[seg], points[seg + 1])
        {
            seg += 1;
            continue;
        }

        let run_start = seg;
        let mut run_end = seg;
        while run_end + 1 < points.len() - 1
            && line_style_includes_segment(run_end + 1, style.line_style)
            && is_valid_segment(points[run_end + 1], points[run_end + 2])
        {
            run_end += 1;
        }

        let run = &points[run_start..=run_end + 1];
        out.extend(tessellate_tube_run(
            run,
            color,
            half_width,
            style.cap,
            radial_segments,
        ));
        seg = run_end + 1;
    }

    out
}

fn side_at(points: &[Vec3], idx: usize) -> Vec3 {
    let prev = prev_dir(points, idx);
    let next = next_dir(points, idx);
    let tangent = match (prev, next) {
        (Some(a), Some(b)) => {
            let s = a + b;
            if s.length_squared() > EPS {
                s.normalize()
            } else {
                b
            }
        }
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => Vec3::X,
    };

    let mut ref_axis = if tangent.z.abs() < 0.95 {
        Vec3::Z
    } else {
        Vec3::X
    };
    let mut side = tangent.cross(ref_axis);
    if side.length_squared() <= EPS {
        ref_axis = Vec3::Y;
        side = tangent.cross(ref_axis);
    }
    if side.length_squared() <= EPS {
        Vec3::Y
    } else {
        side.normalize()
    }
}

fn prev_dir(points: &[Vec3], idx: usize) -> Option<Vec3> {
    if idx == 0 {
        return None;
    }
    let mut j = idx;
    while j > 0 {
        let a = points[j - 1];
        let b = points[j];
        if a.is_finite() && b.is_finite() {
            let d = b - a;
            let len2 = d.length_squared();
            if len2 > EPS {
                return Some(d / len2.sqrt());
            }
        }
        j -= 1;
    }
    None
}

fn next_dir(points: &[Vec3], idx: usize) -> Option<Vec3> {
    if idx + 1 >= points.len() {
        return None;
    }
    let mut j = idx;
    while j + 1 < points.len() {
        let a = points[j];
        let b = points[j + 1];
        if a.is_finite() && b.is_finite() {
            let d = b - a;
            let len2 = d.length_squared();
            if len2 > EPS {
                return Some(d / len2.sqrt());
            }
        }
        j += 1;
    }
    None
}

fn is_valid_segment(a: Vec3, b: Vec3) -> bool {
    a.is_finite() && b.is_finite() && (b - a).length_squared() > EPS
}

fn tessellate_tube_run(
    run: &[Vec3],
    color: Vec4,
    radius: f32,
    cap: StrokeCap3D,
    radial_segments: usize,
) -> Vec<Vertex> {
    let mut out = Vec::new();
    if run.len() < 2 {
        return out;
    }

    let tangents = run_tangents(run);
    if tangents.is_empty() {
        return out;
    }
    let (normals, binormals) = parallel_transport_frames(&tangents);
    let mut centers = run.to_vec();
    if cap == StrokeCap3D::Square {
        centers[0] -= tangents[0] * radius;
        let last = centers.len() - 1;
        centers[last] += tangents[last] * radius;
    }

    let mut rings: Vec<Vec<Vec3>> = Vec::with_capacity(centers.len());
    for i in 0..centers.len() {
        let center = centers[i];
        let n = normals[i];
        let b = binormals[i];
        let mut ring = Vec::with_capacity(radial_segments);
        for s in 0..radial_segments {
            let theta = std::f32::consts::TAU * (s as f32) / (radial_segments as f32);
            let offset = n * theta.cos() * radius + b * theta.sin() * radius;
            ring.push(center + offset);
        }
        rings.push(ring);
    }

    for i in 0..rings.len() - 1 {
        let a = &rings[i];
        let b = &rings[i + 1];
        for s in 0..radial_segments {
            let n = (s + 1) % radial_segments;
            let v00 = a[s];
            let v01 = a[n];
            let v10 = b[s];
            let v11 = b[n];
            out.push(Vertex::new(v00, color));
            out.push(Vertex::new(v10, color));
            out.push(Vertex::new(v11, color));
            out.push(Vertex::new(v00, color));
            out.push(Vertex::new(v11, color));
            out.push(Vertex::new(v01, color));
        }
    }

    let start_center = centers[0];
    let start_ring = &rings[0];
    for s in 0..radial_segments {
        let n = (s + 1) % radial_segments;
        out.push(Vertex::new(start_center, color));
        out.push(Vertex::new(start_ring[n], color));
        out.push(Vertex::new(start_ring[s], color));
    }
    let last = rings.len() - 1;
    let end_center = centers[last];
    let end_ring = &rings[last];
    for s in 0..radial_segments {
        let n = (s + 1) % radial_segments;
        out.push(Vertex::new(end_center, color));
        out.push(Vertex::new(end_ring[s], color));
        out.push(Vertex::new(end_ring[n], color));
    }

    out
}

fn run_tangents(run: &[Vec3]) -> Vec<Vec3> {
    let mut seg_dirs = Vec::with_capacity(run.len().saturating_sub(1));
    for i in 0..run.len() - 1 {
        let d = run[i + 1] - run[i];
        let len2 = d.length_squared();
        if len2 <= EPS {
            return Vec::new();
        }
        seg_dirs.push(d / len2.sqrt());
    }
    let mut tangents = Vec::with_capacity(run.len());
    tangents.push(seg_dirs[0]);
    for i in 1..run.len() - 1 {
        let s = seg_dirs[i - 1] + seg_dirs[i];
        if s.length_squared() > EPS {
            tangents.push(s.normalize());
        } else {
            tangents.push(seg_dirs[i]);
        }
    }
    tangents.push(*seg_dirs.last().unwrap_or(&Vec3::X));
    tangents
}

fn parallel_transport_frames(tangents: &[Vec3]) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut normals = Vec::with_capacity(tangents.len());
    let mut binormals = Vec::with_capacity(tangents.len());

    let t0 = tangents[0];
    let mut n0 = orthogonal_unit(t0);
    let mut b0 = t0.cross(n0);
    if b0.length_squared() <= EPS {
        n0 = Vec3::Y;
        b0 = t0.cross(n0);
    }
    b0 = b0.normalize_or_zero();
    n0 = b0.cross(t0).normalize_or_zero();
    normals.push(n0);
    binormals.push(b0);

    for i in 1..tangents.len() {
        let prev_t = tangents[i - 1];
        let t = tangents[i];
        let mut n = normals[i - 1];

        let axis = prev_t.cross(t);
        let axis_len = axis.length();
        if axis_len > EPS {
            let axis_u = axis / axis_len;
            let cos_theta = prev_t.dot(t).clamp(-1.0, 1.0);
            let sin_theta = axis_len.clamp(0.0, 1.0);
            n = rotate_about_axis(n, axis_u, cos_theta, sin_theta);
        }

        n = (n - t * n.dot(t)).normalize_or_zero();
        if n.length_squared() <= EPS {
            n = orthogonal_unit(t);
        }
        let mut b = t.cross(n).normalize_or_zero();
        if b.length_squared() <= EPS {
            b = orthogonal_unit(t.cross(Vec3::X));
        }
        n = b.cross(t).normalize_or_zero();
        normals.push(n);
        binormals.push(b);
    }

    (normals, binormals)
}

fn orthogonal_unit(t: Vec3) -> Vec3 {
    let ref_axis = if t.z.abs() < 0.95 { Vec3::Z } else { Vec3::X };
    let mut n = ref_axis - t * ref_axis.dot(t);
    if n.length_squared() <= EPS {
        n = Vec3::Y - t * Vec3::Y.dot(t);
    }
    if n.length_squared() <= EPS {
        Vec3::X
    } else {
        n.normalize()
    }
}

fn rotate_about_axis(v: Vec3, axis_u: Vec3, cos_theta: f32, sin_theta: f32) -> Vec3 {
    v * cos_theta + axis_u.cross(v) * sin_theta + axis_u * axis_u.dot(v) * (1.0 - cos_theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tessellated_polyline_has_continuous_shared_joint_side() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let tris = tessellate_polyline(
            &points,
            Vec4::ONE,
            StrokeStyle3D::new(0.1, LineStyle::Solid, StrokeCap3D::Butt),
        );
        assert_eq!(tris.len(), 12);
        let shared_a = Vec3::from_array(tris[1].position);
        let shared_b = Vec3::from_array(tris[6].position);
        assert!((shared_a - shared_b).length() < 1e-5);
    }

    #[test]
    fn dotted_style_uses_sparse_single_segments() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(4.0, 0.0, 0.0),
        ];
        let verts = create_line_vertices_dashed(&points, Vec4::ONE, LineStyle::Dotted);
        // Segments 0 and 4 are included for the first five points (0..3 exist here -> only 0).
        assert_eq!(verts.len(), 2);
    }

    #[test]
    fn tube_tessellation_produces_dense_triangles_for_visible_3d_width() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.5, 0.25),
            Vec3::new(2.0, 0.0, 0.5),
        ];
        let verts = tessellate_polyline_tube(
            &points,
            Vec4::ONE,
            StrokeStyle3D::new(0.1, LineStyle::Solid, StrokeCap3D::Square),
            8,
        );
        assert!(!verts.is_empty());
        assert!(verts.len() > 48);
    }
}
