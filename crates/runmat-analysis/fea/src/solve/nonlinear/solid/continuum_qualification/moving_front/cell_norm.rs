//! Exact front-split L2 norms of actual cell-average fields on straight tetrahedra.
//! Does not invent a continuous constitutive history between integration points.
use super::*;
use std::collections::BTreeMap;

mod clipping;

#[derive(Debug, Clone, Copy)]
pub(super) struct CellNorm {
    pub stress: f64,
    pub alpha: f64,
    pub stress_mean_bias: f64,
    pub alpha_mean_bias: f64,
    pub stress_projection: f64,
    pub alpha_projection: f64,
}

fn volume(tet: [[f64; 3]; 4]) -> f64 {
    let [a, b, c] =
        [1, 2, 3].map(|i| std::array::from_fn::<_, 3, _>(|axis| tet[i][axis] - tet[0][axis]));
    (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]))
        .abs()
        / 6.0
}

fn affine_square(tet: [[f64; 3]; 4], values: [f64; 4]) -> f64 {
    let sum = values.iter().sum::<f64>();
    volume(tet) * (sum * sum + values.iter().map(|v| v * v).sum::<f64>()) / 20.0
}

fn pieces(tet: [[f64; 3]; 4], factor: f64) -> Vec<[[f64; 3]; 4]> {
    let r = front(factor);
    let mut pieces = clipping::clip(tet, r, true);
    pieces.extend(clipping::clip(tet, r, false));
    let actual = pieces.iter().map(|tet| volume(*tet)).sum::<f64>();
    assert!((actual - volume(tet)).abs() < 1e-12 * volume(tet));
    pieces
}

pub(super) fn measure(
    mesh: &SolverMeshArtifact,
    evaluation: &NonlinearSolidEvaluation,
    factor: f64,
) -> CellNorm {
    let nodes = mesh
        .topology
        .nodes
        .iter()
        .map(|n| (n.node_id, n.coordinates_m.canonical_value()))
        .collect::<BTreeMap<_, _>>();
    let mut means = BTreeMap::<_, (f64, f64)>::new();
    assert_eq!(
        evaluation.point_evidence.len(),
        evaluation.trial_state.points.len()
    );
    for (point, state) in evaluation
        .point_evidence
        .iter()
        .zip(&evaluation.trial_state.points)
    {
        let mean = means.entry(point.identity.element_identity).or_default();
        mean.0 += point.volume_weight_m3 * state.equivalent_plastic_strain;
        mean.1 += point.volume_weight_m3;
    }
    let fields = evaluation
        .element_evidence
        .iter()
        .map(|e| (e.element_identity, e))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(fields.len(), mesh.topology.volume_elements.len());
    let mut stress_error = 0.0;
    let mut stress_norm = 0.0;
    let mut alpha_error = 0.0;
    let mut alpha_norm = 0.0;
    let mut stress_bias = 0.0;
    let mut alpha_bias = 0.0;
    let mut stress_projection = 0.0;
    let mut alpha_projection = 0.0;
    for element in &mesh.topology.volume_elements {
        let tet = std::array::from_fn(|i| nodes[&element.node_ids[i]]);
        let mean = &fields[&element.stable_identity];
        assert!((mean.volume_m3 / volume(tet) - 1.0).abs() < 1e-12);
        let (alpha_sum, weight) = means[&element.stable_identity];
        let actual_alpha = alpha_sum / weight;
        let split = pieces(tet, factor);
        let mut reference_mean_stress = [0.0; 6];
        let mut reference_mean_alpha = 0.0;
        for piece in &split {
            let values = piece.map(|v| reference(v[0], factor));
            for value in values {
                for (mean, value) in reference_mean_stress.iter_mut().zip(value.stress) {
                    *mean += volume(*piece) * value / (4.0 * volume(tet));
                }
                reference_mean_alpha += volume(*piece) * value.alpha / (4.0 * volume(tet));
            }
        }
        alpha_bias += volume(tet) * (actual_alpha - reference_mean_alpha).powi(2);
        for (actual, expected) in mean.mean_stress_pa.iter().zip(reference_mean_stress) {
            stress_bias += volume(tet) * (actual - expected).powi(2);
        }
        for piece in split {
            let expected = piece.map(|v| reference(v[0], factor));
            for (axis, reference_mean) in reference_mean_stress.iter().enumerate() {
                let values = expected.map(|v| v.stress[axis]);
                stress_error += affine_square(piece, values.map(|v| mean.mean_stress_pa[axis] - v));
                stress_norm += affine_square(piece, values);
                stress_projection += affine_square(piece, values.map(|v| reference_mean - v));
            }
            alpha_error += affine_square(piece, expected.map(|v| actual_alpha - v.alpha));
            alpha_projection +=
                affine_square(piece, expected.map(|v| reference_mean_alpha - v.alpha));
        }
        // Keep history normalization defined before yield, as in the registered point norm.
        alpha_norm += affine_square(tet, tet.map(|v| reference(v[0], 1.0).alpha));
    }
    // Orthogonal P0 projection identity: output error = mean bias + reference
    // variation within cells. Check rather than attributing all improvement to physics.
    assert!((stress_error - stress_bias - stress_projection).abs() < 1e-12 * stress_norm);
    assert!((alpha_error - alpha_bias - alpha_projection).abs() < 1e-12 * alpha_norm);
    CellNorm {
        stress: (stress_error / stress_norm).sqrt(),
        alpha: (alpha_error / alpha_norm).sqrt(),
        stress_mean_bias: (stress_bias / stress_norm).sqrt(),
        alpha_mean_bias: (alpha_bias / alpha_norm).sqrt(),
        stress_projection: (stress_projection / stress_norm).sqrt(),
        alpha_projection: (alpha_projection / alpha_norm).sqrt(),
    }
}

#[test]
fn split_tetrahedral_norms_equal_independent_piecewise_cube_integrals() {
    for cells in [1, 2, 4] {
        let mesh = slab_mesh(cells);
        let mut factors = vec![0.125, 0.25, 0.375, 0.5, 1.0];
        factors.extend(
            [1.0, 0.75, 0.5, 0.25, 0.0]
                .map(|r| elastic_tangent() * YIELD / (2.0 * mu() * stress(r)[0])),
        );
        for factor in factors {
            let mut squared = [0.0; 2];
            let mut total_volume = 0.0;
            for element in &mesh.topology.volume_elements {
                let tet = std::array::from_fn(|i| {
                    mesh.topology.nodes[element.node_ids[i] as usize - 1]
                        .coordinates_m
                        .canonical_value()
                });
                for piece in pieces(tet, factor) {
                    total_volume += volume(piece);
                    let values = piece.map(|v| reference(v[0], factor));
                    for axis in 0..6 {
                        squared[0] += affine_square(piece, values.map(|v| v.stress[axis]));
                    }
                    squared[1] += affine_square(piece, values.map(|v| v.alpha));
                }
            }
            assert!((total_volume - 1.0).abs() < 1e-12);
            // Simpson is exact separately on each branch for the quadratic squared
            // reference. It uses no mesh geometry or production quadrature.
            let mut independent = [0.0; 2];
            let r = front(factor);
            for (a, b) in [(0.0, r), (r, 1.0)] {
                for (x, weight) in [(a, 1.0), ((a + b) / 2.0, 4.0), (b, 1.0)] {
                    let value = reference(x, factor);
                    independent[0] +=
                        (b - a) / 6.0 * weight * value.stress.iter().map(|v| v * v).sum::<f64>();
                    independent[1] += (b - a) / 6.0 * weight * value.alpha.powi(2);
                }
            }
            for (actual, expected) in squared.iter().zip(independent) {
                assert!(
                    (actual - expected).abs() < 1e-12 * expected.max(1.0),
                    "cells={cells} factor={factor} actual={actual} expected={expected}"
                );
            }
        }
    }
}
