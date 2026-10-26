//! Closed-form affine Tet10 elasticity, independent of production gradients/quadrature.
//! This verifies the original discrete stiffness, not the continuum convergence gate.
use super::*;
use crate::assembly::{assemble_structural_system_borrowed, StructuralFormulationRequirement};
use crate::excitation::{realize_step_excitations, ExcitationSamplePoint};

type Gradient = [[f64; 3]; 4];

fn difference(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| a[axis] - b[axis])
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn affine_gradients(nodes: [[f64; 3]; 10]) -> ([Gradient; 10], f64) {
    let e1 = difference(nodes[1], nodes[0]);
    let e2 = difference(nodes[2], nodes[0]);
    let e3 = difference(nodes[3], nodes[0]);
    let determinant = e1
        .iter()
        .zip(cross(e2, e3))
        .map(|(a, b)| a * b)
        .sum::<f64>();
    assert!(determinant.is_finite() && determinant > 0.0);
    let mut g = [[0.0; 3]; 4];
    g[1] = cross(e2, e3).map(|v| v / determinant);
    g[2] = cross(e3, e1).map(|v| v / determinant);
    g[3] = cross(e1, e2).map(|v| v / determinant);
    g[0] = std::array::from_fn(|axis| -g[1][axis] - g[2][axis] - g[3][axis]);
    let coefficients = std::array::from_fn(|node| {
        if node < 4 {
            // grad(lambda_i*(2*lambda_i-1)); sum(lambda)=1.
            std::array::from_fn(|point| {
                g[node].map(|v| (if node == point { 3.0 } else { -1.0 }) * v)
            })
        } else {
            // Identify the physical edge by coordinates, not production local-edge tables.
            let mut edge = None;
            for i in 0..4 {
                for j in i + 1..4 {
                    if (0..3).all(|axis| {
                        (nodes[node][axis] - 0.5 * (nodes[i][axis] + nodes[j][axis])).abs()
                            < 1.0e-14
                    }) {
                        assert!(
                            edge.replace((i, j)).is_none(),
                            "midside must identify exactly one edge"
                        );
                    }
                }
            }
            let (i, j) = edge.expect("original Tet10 geometry must be affine");
            // grad(4*lambda_i*lambda_j).
            std::array::from_fn(|point| {
                std::array::from_fn(|axis| {
                    4.0 * ((if point == i { g[j][axis] } else { 0.0 })
                        + (if point == j { g[i][axis] } else { 0.0 }))
                })
            })
        }
    });
    (coefficients, determinant / 6.0)
}

fn product(a: &Gradient, b: &Gradient, i: usize, j: usize, volume: f64) -> f64 {
    // Integral(lambda_r*lambda_s) = V*(1+delta_rs)/20.
    let sum_a = a.iter().map(|v| v[i]).sum::<f64>();
    let sum_b = b.iter().map(|v| v[j]).sum::<f64>();
    volume / 20.0 * (sum_a * sum_b + a.iter().zip(b).map(|(a, b)| a[i] * b[j]).sum::<f64>())
}

#[test]
fn original_cantilever_stiffness_matches_closed_form_barycentric_moments() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let loads = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    // Original independent physical coefficients: E=200GPa, nu=.3.
    let shear = 200.0e9 / (2.0 * 1.3);
    let lambda = 200.0e9 * 0.3 / (1.3 * 0.4);
    for divisions in [1, 2, 3] {
        let mesh = structured_cantilever(divisions, ElementOrder::Tet10);
        let summary = assemble_structural_system_borrowed(
            &model,
            &loads,
            Some(&mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        let n = mesh.topology.nodes.len() * 3;
        assert_eq!(summary.dof_count, n);
        let mut exact = vec![0.0; n * n];
        for element in &mesh.topology.volume_elements {
            assert_eq!(element.node_ids.len(), 10);
            let indices: [usize; 10] = std::array::from_fn(|local| {
                let i = element.node_ids[local] as usize - 1;
                assert_eq!(mesh.topology.nodes[i].node_id, element.node_ids[local]);
                i
            });
            let nodes = indices.map(|i| mesh.topology.nodes[i].coordinates_m.canonical_value());
            let (gradients, volume) = affine_gradients(nodes);
            for (i, a) in gradients.iter().enumerate() {
                for (j, b) in gradients.iter().enumerate() {
                    let trace = (0..3)
                        .map(|axis| product(a, b, axis, axis, volume))
                        .sum::<f64>();
                    for alpha in 0..3 {
                        for beta in 0..3 {
                            let value = lambda * product(a, b, alpha, beta, volume)
                                + shear
                                    * (product(a, b, beta, alpha, volume)
                                        + if alpha == beta { trace } else { 0.0 });
                            exact[(indices[i] * 3 + alpha) * n + indices[j] * 3 + beta] += value;
                        }
                    }
                }
            }
        }
        let actual = summary.operator.operators().stiffness();
        let mut difference_squared = 0.0;
        let mut norm_squared = 0.0;
        let mut max_difference = 0.0_f64;
        let mut max_coefficient = 0.0_f64;
        for (i, row) in exact.chunks_exact(n).enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let difference = actual.get(i, j) - value;
                difference_squared += difference * difference;
                norm_squared += value * value;
                max_difference = max_difference.max(difference.abs());
                max_coefficient = max_coefficient.max(value.abs());
            }
        }
        let relative = (difference_squared / norm_squared).sqrt();
        let max_relative = max_difference / max_coefficient;
        assert!(
            relative < 1.0e-13 && max_relative < 1.0e-13,
            "divisions={divisions} norm_difference={relative:e} max_difference={max_relative:e}"
        );
        eprintln!("original stiffness divisions={divisions} entries={} relative={relative:e} maximum_relative={max_relative:e}",n*n);
    }
}
