use super::*;
use crate::cdt::{
    build_delaunay_volume_topology, DelaunayInsertionOptions, DelaunayTopologyOptions,
};
use runmat_meshing_core::NeverCancelled;

// Independent full lifted determinant. Unit-sphere integer coordinates and
// binary perturbations make every product and sum exactly representable here.
fn determinant(matrix: [[f64; 5]; 5]) -> f64 {
    fn sum(matrix: &[[f64; 5]; 5], row: usize, used: u8, parity: bool) -> f64 {
        if row == 5 {
            return if parity { -1.0 } else { 1.0 };
        }
        (0..5)
            .filter(|column| used & (1 << column) == 0)
            .map(|column| {
                let inversions = !(used >> (column + 1)).count_ones().is_multiple_of(2);
                matrix[row][column]
                    * sum(matrix, row + 1, used | (1 << column), parity ^ inversions)
            })
            .sum()
    }
    sum(&matrix, 0, 0, false)
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Conflict-cavity inclusion and legalization share the exact symbolic in-sphere tie
//   policy. An independent full lifted determinant on a unit-sphere inventory exercises inclusion
//   and exclusion without changing physical coordinates. Separate forced on-face traversal,
//   whole-topology geometric admission, cancellation and original native recovery gates remain
//   required; this oracle does not establish global constraint preservation or performance.
#[test]
fn cavity_ties_match_the_independent_lifted_oracle_and_legalization_policy() {
    let points = [
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
        [0., 1., 0.],
        [1., 0., 0.],
    ];
    let node = |index: usize| {
        DelaunayVolumeNode::explicit(
            StableDigest::from_bytes([(index + 1) as u8; 32]),
            points[index],
        )
    };
    let mut excluded = 0;
    let mut included = 0;
    for query in 0..5 {
        let nodes = (0..5).filter(|index| *index != query).map(node).collect();
        let Ok(topology) = build_delaunay_volume_topology(
            nodes,
            vec![[0, 1, 2, 3]],
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        ) else {
            continue;
        };
        let vertices = topology.tetrahedra[0].vertex_indices;
        let mut rows = vertices
            .map(|index| {
                topology.nodes[index as usize]
                    .coordinates_m
                    .canonical_value()
            })
            .to_vec();
        rows.push(points[query]);
        let lifted = |perturbed: bool| {
            std::array::from_fn(|row| {
                let [x, y, z] = rows[row];
                let rank = points.iter().position(|point| *point == rows[row]).unwrap();
                let shift = if perturbed {
                    2.0_f64.powi(-4 * (rank as i32 + 1))
                } else {
                    0.0
                };
                [x, y, z, x * x + y * y + z * z - shift, 1.0]
            })
        };
        assert_eq!(determinant(lifted(false)), 0.0);
        let mut inside = lifted(false);
        inside[4] = [0., 0., 0., 0., 1.];
        let oracle = if determinant(lifted(true)) * determinant(inside) > 0.0 {
            PredicateSign::Positive
        } else {
            PredicateSign::Negative
        };
        let mut work = InsertionWork::new(DelaunayInsertionOptions::default(), &NeverCancelled);
        let actual = circumsphere_relation(&topology, vertices, node(query), &mut work).unwrap();
        assert_eq!(actual, oracle);
        assert_eq!(
            super::super::in_circumsphere(&topology, vertices, node(query), &mut work).unwrap(),
            oracle == PredicateSign::Positive
        );
        if oracle == PredicateSign::Positive {
            included += 1
        } else {
            excluded += 1
        }
    }
    assert!(
        included > 0 && excluded > 0,
        "both sides of a physical tie must be exercised"
    );
}
