use super::*;
use crate::physics::cfd::canonical::qualification::fixture::{case, ReferenceCase};

/// Cells within `depth` face steps of `root`.
fn ring(t: &FiniteVolumeTopology, root: usize, depth: usize) -> BTreeSet<usize> {
    let mut visited = BTreeSet::from([root]);
    let mut front = vec![root];
    for _ in 0..depth {
        front = next_ring(t, &front, &mut visited);
    }
    visited
}

fn reproduces_affine(stencils: &PressureGradientStencils, t: &FiniteVolumeTopology, c: usize) {
    let g = [0.3, -1.7, 2.9];
    let value = |x: [f64; 3]| dot(g, x);
    let mut gradient = [0.0; 3];
    for (sample, weight) in stencils.cell(c) {
        let Sample::Cell(n) = *sample else {
            panic!("closed box has no outlet samples")
        };
        gradient = add(
            gradient,
            scale(
                *weight,
                value(t.cells[n].centroid_m) - value(t.cells[c].centroid_m),
            ),
        );
    }
    assert!(
        norm(sub(gradient, g)) < 1e-9 * norm(g),
        "{gradient:?} vs {g:?}"
    );
}

// A cell in a corner strip (one face on an inlet, one on a wall) of the compliant-tube mesh
// sees only its own layer within three face steps: eight such cells had every depth-3 centroid
// in one plane, so the pressure stencil was singular in the plane normal. Reproduced here by
// moving every centroid within three face steps of a cell into that cell's z-plane.
#[test]
#[ignore = "fluid-18 WIP fixture: on the transplanted Couette mesh, moving the ring centroids into the root plane makes another cell (element 50) coincide with a moved centroid, so its stencil has a zero displacement before widening applies; owned by the fluid lane"]
fn coplanar_depth_three_ring_widens_to_the_next_face_ring() {
    let mut t = case(4, ReferenceCase::Couette, false).topology;
    let b = vec![FaceBoundary::Interior; t.faces.len()];
    let root = t.cells.len() / 2;
    let flat = ring(&t, root, DEPTH);
    assert!(ring(&t, root, DEPTH + 1).len() > flat.len());
    let z = t.cells[root].centroid_m[2];
    for &c in &flat {
        t.cells[c].centroid_m[2] = z;
    }
    let stencils = derive(&t, &b).unwrap();
    assert!(
        stencils.cell(root).len() > flat.len() - 1,
        "the ring grew past depth 3"
    );
    reproduces_affine(&stencils, &t, root);
    // Spanning rings keep the depth-3 neighborhood.
    let other = *ring(&t, root, DEPTH + 2)
        .difference(&ring(&t, root, DEPTH + 1))
        .next()
        .unwrap();
    assert_eq!(stencils.cell(other).len(), ring(&t, other, DEPTH).len() - 1);
    reproduces_affine(&stencils, &t, other);
}

#[test]
fn spanning_test_separates_coplanar_from_three_dimensional_samples() {
    use super::super::reconstruction::GradientMetric;
    let plane = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-2.0, 0.5, 0.0],
    ];
    assert!(!GradientMetric::derive(&plane)
        .unwrap()
        .spans_three_dimensions());
    let tilted = plane.map(|d| [d[0], d[1], 1e-5 * d[0]]);
    assert!(!GradientMetric::derive(&tilted)
        .unwrap()
        .spans_three_dimensions());
    let spread = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -1.0, 0.2],
    ];
    assert!(GradientMetric::derive(&spread)
        .unwrap()
        .spans_three_dimensions());
}
