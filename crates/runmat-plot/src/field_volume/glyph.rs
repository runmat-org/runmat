//! Arrow glyphs: a vector field drawn as shaded arrows at lattice points of a plane. Each arrow
//! points along the field evaluated exactly at its centre (element value or basis); its color
//! carries the exact value and its length a bounded proportion of it, so short arrows stay
//! readable.

use super::{add_scaled, cross, norm, normalize, sub, FieldTriangles, Reduction, SolvedField, TetrahedronLocator};

#[derive(Debug, Clone, PartialEq)]
pub struct GlyphOptions {
    pub origin: [f64; 3],
    pub normal: [f64; 3],
    /// Lattice spacing in metres; also the longest arrow.
    pub spacing: f64,
    /// Arrows below this fraction of the largest sampled magnitude are left out.
    pub min_relative_magnitude: f64,
    /// Facets around each arrow.
    pub segments: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Glyphs {
    pub triangles: FieldTriangles,
    pub arrows: usize,
    /// Largest curved-element deviation from the straight-sided tetrahedra used to locate points.
    pub geometric_error_m: f64,
}

/// Arrows of the vector `field` (the instantaneous field at `phase` for a phasor) on the
/// lattice of `options`, colored by `color` of the field at each arrow's centre.
pub fn arrow_glyphs(
    locator: &TetrahedronLocator<'_>,
    field: &SolvedField,
    color: Reduction,
    phase_radians: f64,
    options: &GlyphOptions,
) -> Glyphs {
    let mesh = locator.mesh();
    let mut glyphs = Glyphs::default();
    if field.components != 3 {
        return glyphs;
    }
    let (s, c) = phase_radians.sin_cos();
    let normal = normalize(options.normal);
    let helper = if normal[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = normalize(cross(normal, helper));
    let v = cross(normal, u);
    let (min, max) = mesh.bounds();
    let spacing = options.spacing.max(f64::MIN_POSITIVE);
    let steps = ((norm(sub(max, min)) / spacing).ceil() as i64).min(400);
    let mut samples = Vec::new();
    for i in -steps..=steps {
        for j in -steps..=steps {
            let point = add_scaled(add_scaled(options.origin, u, i as f64 * spacing), v, j as f64 * spacing);
            let Some(located) = locator.locate(point) else { continue };
            let value = field.at(mesh, located.element, located.weights);
            let instantaneous = [0, 1, 2].map(|k| value.real[k] * c - value.imaginary[k] * s);
            glyphs.geometric_error_m = glyphs.geometric_error_m.max(mesh.curvature_deviation(located.element));
            samples.push((point, instantaneous, color.apply(&value)));
        }
    }
    let largest = samples.iter().map(|(_, vector, _)| norm(*vector)).fold(0.0, f64::max);
    if largest <= 0.0 {
        return glyphs;
    }
    for (point, vector, value) in samples {
        let magnitude = norm(vector);
        if magnitude < largest * options.min_relative_magnitude {
            continue;
        }
        let length = spacing * 0.9 * (0.35 + 0.65 * magnitude / largest);
        push_arrow(&mut glyphs.triangles, point, normalize(vector), length, value, options.segments.max(3));
        glyphs.arrows += 1;
    }
    glyphs
}

/// An arrow centred on `center`: a shaft and a cone head, every vertex carrying `value`.
fn push_arrow(out: &mut FieldTriangles, center: [f64; 3], direction: [f64; 3], length: f64, value: f64, segments: usize) {
    let helper = if direction[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let side = normalize(cross(direction, helper));
    let up = cross(direction, side);
    let shaft_radius = length * 0.06;
    let head_radius = length * 0.16;
    let head_length = length * 0.35;
    let tail = add_scaled(center, direction, -0.5 * length);
    let neck = add_scaled(tail, direction, length - head_length);
    let tip = add_scaled(tail, direction, length);
    let ring = |origin: [f64; 3], radius: f64, k: usize| -> ([f64; 3], [f64; 3]) {
        let angle = k as f64 / segments as f64 * std::f64::consts::TAU;
        let (sin, cos) = angle.sin_cos();
        let radial = [0, 1, 2].map(|a| side[a] * cos + up[a] * sin);
        (add_scaled(origin, radial, radius), radial)
    };
    for k in 0..segments {
        let (a0, n0) = ring(tail, shaft_radius, k);
        let (a1, n1) = ring(tail, shaft_radius, k + 1);
        let (b0, _) = ring(neck, shaft_radius, k);
        let (b1, _) = ring(neck, shaft_radius, k + 1);
        let shaft = [
            out.push_vertex(a0, n0, value),
            out.push_vertex(a1, n1, value),
            out.push_vertex(b1, n1, value),
            out.push_vertex(b0, n0, value),
        ];
        out.indices.extend_from_slice(&[shaft[0], shaft[1], shaft[2], shaft[0], shaft[2], shaft[3]]);
        let (h0, m0) = ring(neck, head_radius, k);
        let (h1, m1) = ring(neck, head_radius, k + 1);
        // Cone normals lean forward by the head's half angle.
        let lean = |radial: [f64; 3]| normalize(add_scaled(radial, direction, head_radius / head_length));
        let head = [
            out.push_vertex(h0, lean(m0), value),
            out.push_vertex(h1, lean(m1), value),
            out.push_vertex(tip, direction, value),
        ];
        out.indices.extend_from_slice(&head);
        let back = direction.map(|x| -x);
        let base = [out.push_vertex(neck, back, value), out.push_vertex(h1, back, value), out.push_vertex(h0, back, value)];
        out.indices.extend_from_slice(&base);
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::TetOrder;
    use super::*;

    #[test]
    fn arrows_point_along_the_field_at_their_centres_and_carry_its_value() {
        let mesh = unit_cube(TetOrder::Quadratic);
        let field = nodal(&mesh, |p| vec![p[2], 0.0, 1.0 + p[0]]);
        let locator = TetrahedronLocator::new(&mesh);
        let options = GlyphOptions { origin: [0.5, 0.5, 0.5], normal: [0.0, 1.0, 0.0], spacing: 0.2, min_relative_magnitude: 0.0, segments: 6 };
        let glyphs = arrow_glyphs(&locator, &field, Reduction::Magnitude, 0.0, &options);
        assert!(glyphs.arrows >= 16, "{}", glyphs.arrows);
        assert_eq!(glyphs.triangles.triangle_count(), glyphs.arrows * 6 * 4);
        // Each arrow's vertices share one value; the arrow points along the field at its centre,
        // and the value is the field's magnitude there. Per facet: 4 shaft, 3 head (tip last),
        // 3 base vertices.
        let per_arrow = glyphs.triangles.positions.len() / glyphs.arrows;
        assert_eq!(per_arrow, 6 * 10);
        let at = |k: usize| glyphs.triangles.positions[k].map(f64::from);
        for arrow in 0..glyphs.arrows {
            let first = arrow * per_arrow;
            let value = glyphs.triangles.values[first];
            assert!(glyphs.triangles.values[first..first + per_arrow].iter().all(|v| *v == value));
            let tip = at(first + 6);
            let neck = [0, 1, 2].map(|a| (0..6).map(|k| at(first + 10 * k + 4)[a]).sum::<f64>() / 6.0);
            let axis = sub(tip, neck);
            let center = add_scaled(neck, axis, -0.15 / 0.35);
            let exact = [center[2], 0.0, 1.0 + center[0]];
            assert!((value - norm(exact)).abs() < 1e-5, "{value} vs {}", norm(exact));
            let alignment = normalize(axis).iter().zip(normalize(exact)).map(|(a, b)| a * b).sum::<f64>();
            assert!(alignment > 1.0 - 1e-6, "{alignment}");
        }
    }
}
