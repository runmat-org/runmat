//! Original-mesh discrete-equilibrium discriminator, not continuum qualification.
//! Test-only dense factorization shares no sparse solver, ordering or preconditioner.
use super::*;
use crate::assembly::{assemble_structural_system_borrowed, StructuralFormulationRequirement};
use crate::excitation::{realize_step_excitations, ExcitationSamplePoint};

fn dense_cholesky(matrix: &[Vec<f64>], load: &[f64]) -> Vec<f64> {
    let n = load.len();
    assert_eq!(matrix.len(), n);
    let scale = (0..n).map(|i| matrix[i][i]).fold(0.0_f64, f64::max);
    assert!(scale.is_finite() && scale > 0.0);
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        assert_eq!(matrix[i].len(), n);
        for j in 0..=i {
            let mut value = matrix[i][j] / scale;
            for k in 0..j {
                value -= lower[i][k] * lower[j][k];
            }
            lower[i][j] = if i == j {
                assert!(
                    value.is_finite() && value > 0.0,
                    "pivot={i} value={value:e}"
                );
                value.sqrt()
            } else {
                value / lower[j][j]
            };
        }
    }
    let mut x = load.iter().map(|v| v / scale).collect::<Vec<_>>();
    for i in 0..n {
        for j in 0..i {
            x[i] -= lower[i][j] * x[j];
        }
        x[i] /= lower[i][i];
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] -= lower[j][i] * x[j];
        }
        x[i] /= lower[i][i];
    }
    x
}

#[test]
fn original_cantilever_equilibrium_matches_separate_dense_factorization() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let loads = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let mut peaks = Vec::new();
    for divisions in [1, 2, 3] {
        let mesh = structured_cantilever(divisions, ElementOrder::Tet10);
        let summary = assemble_structural_system_borrowed(
            &model,
            &loads,
            Some(&mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        let operator = &summary.operator;
        let free = operator
            .constrained_dofs()
            .iter()
            .enumerate()
            .filter_map(|(i, fixed)| (!fixed).then_some(i))
            .collect::<Vec<_>>();
        // The original clamp is homogeneous. No transformed or fitted reference load.
        let stiffness = operator.operators().stiffness();
        let matrix = free
            .iter()
            .map(|&i| {
                free.iter()
                    .map(|&j| stiffness.get(i, j))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let force = free
            .iter()
            .map(|&i| summary.physical_load[i])
            .collect::<Vec<_>>();
        let dense = dense_cholesky(&matrix, &force);
        let result = crate::run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                solver_mesh: Some(mesh.clone()),
                require_solver_mesh_for_solid: true,
                ..LinearStaticSolveOptions::default()
            },
        )
        .unwrap();
        let actual = result
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap();
        let norm = dense.iter().map(|v| v * v).sum::<f64>().sqrt();
        let error = free
            .iter()
            .zip(&dense)
            .map(|(&i, v)| (actual[i] - v).powi(2))
            .sum::<f64>()
            .sqrt()
            / norm;
        let balance = matrix
            .iter()
            .zip(&force)
            .map(|(row, f)| (row.iter().zip(&dense).map(|(k, u)| k * u).sum::<f64>() - f).powi(2))
            .sum::<f64>()
            .sqrt()
            / force.iter().map(|f| f * f).sum::<f64>().sqrt();
        assert!(
            error < 1.0e-8,
            "divisions={divisions} displacement_difference={error:e}"
        );
        assert!(
            balance < 1.0e-11,
            "divisions={divisions} dense_residual={balance:e}"
        );
        for (i, fixed) in operator.constrained_dofs().iter().enumerate() {
            if *fixed {
                assert_eq!(actual[i], 0.0);
            }
        }
        let mut resultant = [0.0; 3];
        let mut moment = [0.0; 3];
        for (node, f) in mesh
            .topology
            .nodes
            .iter()
            .zip(summary.physical_load.chunks_exact(3))
        {
            let [x, y, z] = node.coordinates_m.canonical_value();
            for axis in 0..3 {
                resultant[axis] += f[axis];
            }
            moment[0] += y * f[2] - z * f[1];
            moment[1] += z * f[0] - x * f[2];
            moment[2] += x * f[1] - y * f[0];
        }
        for (v, exact) in resultant
            .into_iter()
            .chain(moment)
            .zip([0.0, -1000.0, 0.0, 500.0, 0.0, -1000.0])
        {
            assert!(
                (v - exact).abs() < 1.0e-8,
                "divisions={divisions} resultant={v} exact={exact}"
            );
        }
        let work = force.iter().zip(&dense).map(|(f, u)| f * u).sum::<f64>();
        let energy = result
            .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
            .unwrap()
            .as_host_f64()
            .unwrap()[0];
        assert!((2.0 * energy - work).abs() / work.abs() < 1.0e-8);
        peaks.push(dense.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
        eprintln!("original cantilever divisions={divisions} free_dofs={} dense_residual={balance:e} sparse_difference={error:e} dense_peak={:e}", free.len(), peaks.last().unwrap());
    }
    eprintln!("original dense peaks={peaks:?} refinement_change_ratio={:e}; original 0.25 qualification gate remains separate", (peaks[2] - peaks[1]).abs() / (peaks[1] - peaks[0]).abs());
}

#[test]
fn separate_factorization_preserves_known_solution_across_unit_scales() {
    // Direct multiplication of this fixed matrix by [1,-2,3] gives [5,-5,7].
    for scale in [1.0e-100, 1.0, 1.0e100] {
        let matrix = [[4.0, 1.0, 1.0], [1.0, 3.0, 0.0], [1.0, 0.0, 2.0]]
            .map(|row| row.map(|v| v * scale).to_vec());
        let load = [5.0, -5.0, 7.0].map(|v| v * scale);
        let actual = dense_cholesky(&matrix, &load);
        for (v, expected) in actual.into_iter().zip([1.0, -2.0, 3.0]) {
            assert!((v - expected).abs() < 2.0e-14);
        }
    }
}
