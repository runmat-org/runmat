//! Section caps: the cut of the mesh by a plane, colored by the field where it is cut.
//!
//! Each element the plane crosses contributes its own planar polygon (a triangle or a quad
//! between the crossings of its corner edges). Points on it are known by their barycentric
//! coordinates in that element, so the field is evaluated there with the element's own value
//! (element fields) or basis (nodal fields). Nodal fields are evaluated on a refined grid of each
//! polygon triangle; the colors drawn between grid points are linear, and the cap reports the
//! largest difference between the field and that linear color at sub-edge midpoints.

use super::{dot, normalize, sub, cross, FieldSupport, FieldTriangles, Reduction, SolvedField, TetMesh};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectionOptions {
    pub origin: [f64; 3],
    /// The cap faces against this normal (toward the removed side, `dot(x - origin, normal) < 0`).
    pub normal: [f64; 3],
    /// Grid divisions per polygon-triangle edge for nodal fields (element fields are constant
    /// on each element and use one).
    pub subdivisions: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SectionCap {
    pub triangles: FieldTriangles,
    /// Elements the plane crosses.
    pub cut_elements: usize,
    /// Largest |field - linear color interpolation| at sub-edge midpoints, in field units.
    pub display_error: f64,
    /// Largest distance between a cut curved element and the straight-sided tetrahedron the cut
    /// uses (zero for Tet4 and straight Tet10 elements), in metres.
    pub geometric_error_m: f64,
}

/// The cap of `mesh` cut by the plane of `options`, colored by `reduction` of `field`.
pub fn section_cap(mesh: &TetMesh, field: &SolvedField, reduction: Reduction, options: &SectionOptions) -> SectionCap {
    let normal = normalize(options.normal);
    let facing = normal.map(|v| -v);
    let divisions = match field.support {
        FieldSupport::Element => 1,
        FieldSupport::Node => options.subdivisions.max(1),
    };
    let mut cap = SectionCap::default();
    for element in 0..mesh.element_count() {
        let corners = mesh.corners(element);
        let distance = corners.map(|p| dot(sub(p, options.origin), normal));
        let above: Vec<usize> = (0..4).filter(|&k| distance[k] >= 0.0).collect();
        let below: Vec<usize> = (0..4).filter(|&k| distance[k] < 0.0).collect();
        let edges: Vec<(usize, usize)> = match (above.len(), below.len()) {
            (1, 3) => below.iter().map(|&b| (above[0], b)).collect(),
            (3, 1) => above.iter().map(|&a| (a, below[0])).collect(),
            // Cyclic order around the quad: consecutive crossings share a corner.
            (2, 2) => vec![(above[0], below[0]), (above[0], below[1]), (above[1], below[1]), (above[1], below[0])],
            _ => continue,
        };
        cap.cut_elements += 1;
        cap.geometric_error_m = cap.geometric_error_m.max(mesh.curvature_deviation(element));
        let polygon: Vec<[f64; 4]> = edges
            .iter()
            .map(|&(a, b)| {
                let t = distance[a] / (distance[a] - distance[b]);
                let mut weights = [0.0; 4];
                weights[a] = 1.0 - t;
                weights[b] = t;
                weights
            })
            .collect();
        let evaluate = |weights: [f64; 4]| reduction.apply(&field.at(mesh, element, weights));
        for k in 1..polygon.len() - 1 {
            let mut triangle = [polygon[0], polygon[k], polygon[k + 1]];
            let positions = triangle.map(|w| mesh.point(element, w));
            if dot(cross(sub(positions[1], positions[0]), sub(positions[2], positions[0])), facing) < 0.0 {
                triangle.swap(1, 2);
            }
            emit_refined(&mut cap, mesh, element, triangle, divisions, facing, &evaluate);
        }
    }
    cap
}

/// Emits `triangle` (barycentric corners in `element`) as a `divisions`² grid of triangles.
fn emit_refined(
    cap: &mut SectionCap,
    mesh: &TetMesh,
    element: usize,
    triangle: [[f64; 4]; 3],
    divisions: usize,
    facing: [f64; 3],
    evaluate: &dyn Fn([f64; 4]) -> f64,
) {
    let n = divisions;
    let at = |i: usize, j: usize| -> [f64; 4] {
        let (u, v) = (i as f64 / n as f64, j as f64 / n as f64);
        [0, 1, 2, 3].map(|k| triangle[0][k] * (1.0 - u - v) + triangle[1][k] * u + triangle[2][k] * v)
    };
    // Row-major ids of the grid points (i along corner 1, j along corner 2, i + j <= n).
    let mut ids = Vec::with_capacity((n + 1) * (n + 2) / 2);
    let mut grid = Vec::with_capacity(ids.capacity());
    for i in 0..=n {
        for j in 0..=n - i {
            let weights = at(i, j);
            let value = evaluate(weights);
            ids.push(cap.triangles.push_vertex(mesh.point(element, weights), facing, value));
            grid.push((weights, value));
        }
    }
    // Grid point (i, j) follows rows 0..i, row r holding n + 1 - r points.
    let offset = |i: usize, j: usize| -> usize { (0..i).map(|r| n + 1 - r).sum::<usize>() + j };
    let mut sub_triangles = Vec::new();
    for i in 0..n {
        for j in 0..n - i {
            sub_triangles.push([offset(i, j), offset(i + 1, j), offset(i, j + 1)]);
            if i + j + 1 < n {
                sub_triangles.push([offset(i + 1, j), offset(i + 1, j + 1), offset(i, j + 1)]);
            }
        }
    }
    for corners in sub_triangles {
        cap.triangles.indices.extend(corners.map(|k| ids[k]));
        if evaluate_is_constant(&grid) {
            continue;
        }
        for (a, b) in [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])] {
            let (wa, va) = grid[a];
            let (wb, vb) = grid[b];
            let middle = [0, 1, 2, 3].map(|k| 0.5 * (wa[k] + wb[k]));
            let error = (evaluate(middle) - 0.5 * (va + vb)).abs();
            cap.display_error = cap.display_error.max(error);
        }
    }
}

/// A grid holding one value (an element field) is drawn exactly by linear colors.
fn evaluate_is_constant(grid: &[([f64; 4], f64)]) -> bool {
    grid.windows(2).all(|pair| pair[0].1 == pair[1].1)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::{TetOrder, FieldSupport};
    use super::*;

    fn area(t: &FieldTriangles) -> f64 {
        t.indices
            .chunks_exact(3)
            .map(|tri| {
                let p = [0, 1, 2].map(|k| t.positions[tri[k] as usize].map(f64::from));
                0.5 * super::super::norm(cross(sub(p[1], p[0]), sub(p[2], p[0])))
            })
            .sum()
    }

    fn options(origin: [f64; 3], normal: [f64; 3], subdivisions: usize) -> SectionOptions {
        SectionOptions { origin, normal, subdivisions }
    }

    #[test]
    fn a_cap_covers_the_section_faces_the_removed_side_and_draws_a_linear_field_exactly() {
        let mesh = unit_cube(TetOrder::Linear);
        let field = nodal(&mesh, |p| vec![2.0 * p[1] - p[2]]);
        let cap = section_cap(&mesh, &field, Reduction::Magnitude, &options([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 1));
        assert!((area(&cap.triangles) - 1.0).abs() < 1e-6, "area {}", area(&cap.triangles));
        assert!(cap.cut_elements > 0);
        for tri in cap.triangles.indices.chunks_exact(3) {
            let p = [0, 1, 2].map(|k| cap.triangles.positions[tri[k] as usize].map(f64::from));
            assert!(cross(sub(p[1], p[0]), sub(p[2], p[0]))[0] < 0.0, "cap faces -x");
        }
        for (p, v) in cap.triangles.positions.iter().zip(&cap.triangles.values) {
            assert!((p[0] - 0.5).abs() < 1e-6);
            let exact = (2.0 * p[1] as f64 - p[2] as f64).abs();
            assert!((v - exact).abs() < 1e-6);
        }
        assert_eq!(cap.geometric_error_m, 0.0);
    }

    #[test]
    fn quadratic_nodal_fields_use_the_element_basis_and_report_the_display_error() {
        let mesh = unit_cube(TetOrder::Quadratic);
        let f = |p: [f64; 3]| p[1] * p[1] + p[2];
        let field = nodal(&mesh, |p| vec![f(p)]);
        let coarse = section_cap(&mesh, &field, Reduction::InstantaneousAt { phase_radians: 0.0 }, &options([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 1));
        let fine = section_cap(&mesh, &field, Reduction::InstantaneousAt { phase_radians: 0.0 }, &options([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 4));
        // Every drawn point carries the exact quadratic value, never a corner-linear one.
        for (p, v) in fine.triangles.positions.iter().zip(&fine.triangles.values) {
            assert!((v - f(p.map(f64::from))).abs() < 1e-6);
        }
        // y^2 departs from its chord by h^2/4 at the chord's middle: refinement shrinks it.
        assert!(coarse.display_error > 0.01, "{}", coarse.display_error);
        assert!(fine.display_error < coarse.display_error / 8.0, "{} vs {}", fine.display_error, coarse.display_error);
    }

    #[test]
    fn element_fields_stay_constant_per_element_on_the_cap() {
        let mesh = unit_cube(TetOrder::Linear);
        let field = SolvedField { support: FieldSupport::Element, components: 1, real: (0..6).map(f64::from).collect(), imaginary: None };
        let cap = section_cap(&mesh, &field, Reduction::InstantaneousAt { phase_radians: 0.0 }, &options([0.5, 0.5, 0.5], [0.0, 0.0, 1.0], 4));
        assert_eq!(cap.display_error, 0.0);
        // Each triangle's three vertices hold one element's value: nothing is averaged.
        for tri in cap.triangles.indices.chunks_exact(3) {
            let values = [0, 1, 2].map(|k| cap.triangles.values[tri[k] as usize]);
            assert!(values[0] == values[1] && values[1] == values[2]);
            assert_eq!(values[0].fract(), 0.0);
        }
        let distinct: std::collections::BTreeSet<i64> = cap.triangles.values.iter().map(|v| *v as i64).collect();
        assert_eq!(distinct.len(), 6, "the mid-height cut crosses all six tetrahedra");
    }

    #[test]
    fn phasor_parts_on_a_cap() {
        let mesh = unit_cube(TetOrder::Linear);
        let mut field = nodal(&mesh, |p| vec![p[0]]);
        field.imaginary = Some(positions(&mesh).iter().map(|p| p[1]).collect());
        let plane = options([0.5, 0.5, 0.5], [0.0, 0.0, 1.0], 2);
        let amplitude = section_cap(&mesh, &field, Reduction::Magnitude, &plane);
        let quarter = section_cap(&mesh, &field, Reduction::InstantaneousAt { phase_radians: std::f64::consts::FRAC_PI_2 }, &plane);
        for ((p, a), q) in amplitude.triangles.positions.iter().zip(&amplitude.triangles.values).zip(&quarter.triangles.values) {
            let (x, y) = (p[0] as f64, p[1] as f64);
            assert!((a - x.hypot(y)).abs() < 1e-6);
            assert!((q + y).abs() < 1e-6, "Re((x + iy) i) = -y");
        }
        assert!(amplitude.display_error > 0.0, "|z| is not linear between grid points");
    }

    #[test]
    fn a_plane_outside_the_mesh_cuts_nothing() {
        let mesh = unit_cube(TetOrder::Linear);
        let field = nodal(&mesh, |p| vec![p[0]]);
        let cap = section_cap(&mesh, &field, Reduction::Magnitude, &options([2.0, 0.0, 0.0], [1.0, 0.0, 0.0], 2));
        assert!(cap.triangles.is_empty());
        assert_eq!(cap.cut_elements, 0);
    }
}
