//! Locate the original continuum discrepancy before taking any time step.
//! This is a discriminator, not a replacement for the mandatory history gates.
use super::*;
use crate::{
    algebra::spmv,
    assembly::{assemble_structural_system_borrowed, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
};

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[test]
fn original_sine_initial_acceleration_separates_spatial_and_time_error() {
    for quadratic in [false, true] {
        for cells in if quadratic { [1, 2, 4] } else { [2, 4, 8] } {
            let mesh = mesh(quadratic, cells);
            let model = model(&mesh, false);
            let loads = realize_step_excitations(
                &model,
                &model.sequence.steps[0],
                ExcitationSamplePoint::time(0.0),
            )
            .unwrap();
            let summary = assemble_structural_system_borrowed(
                &model,
                &loads,
                Some(&mesh),
                StructuralFormulationRequirement::PhysicalDiscretization,
            )
            .unwrap();
            let coarse_time = run(&mesh, false, 2);
            let finer_time = run(&mesh, false, 4);
            // The initial equilibrium is independent of the future time partition.
            // Do not substitute a projected eigenmode or fit an assembled frequency.
            assert_eq!(
                coarse_time.acceleration_snapshots[0],
                finer_time.acceleration_snapshots[0]
            );
            let displacement = coarse_time.displacement_snapshots[0].as_host_f64().unwrap();
            let acceleration = coarse_time.acceleration_snapshots[0].as_host_f64().unwrap();
            let exact = reference::state(0.0, false);
            let mut continuum_acceleration = vec![0.0; summary.dof_count];
            for (node, values) in mesh.topology.nodes.iter().zip(displacement.chunks_exact(3)) {
                assert_eq!(
                    values,
                    &[
                        reference::shape(node.coordinates_m.canonical_value()[0]) * AMPLITUDE,
                        0.0,
                        0.0
                    ]
                );
            }
            for (node, values) in mesh
                .topology
                .nodes
                .iter()
                .zip(continuum_acceleration.chunks_exact_mut(3))
            {
                values[0] = reference::shape(node.coordinates_m.canonical_value()[0]) * exact[2];
            }
            let mass = summary.operator.operators().mass();
            let elastic = spmv(summary.operator.operators().stiffness(), displacement).unwrap();
            let discrete_inertia = spmv(mass, acceleration).unwrap();
            let continuum_inertia = spmv(mass, &continuum_acceleration).unwrap();
            let free = |values: Vec<f64>| {
                values
                    .into_iter()
                    .zip(summary.operator.constrained_dofs())
                    .filter_map(|(value, constrained)| (!constrained).then_some(value))
                    .collect::<Vec<_>>()
            };
            let equilibrium = free(
                discrete_inertia
                    .iter()
                    .zip(&elastic)
                    .zip(&summary.physical_load)
                    .map(|((inertia, elastic), load)| inertia + elastic - load)
                    .collect(),
            );
            let continuum_defect = free(
                continuum_inertia
                    .iter()
                    .zip(&elastic)
                    .zip(&summary.physical_load)
                    .map(|((inertia, elastic), load)| inertia + elastic - load)
                    .collect(),
            );
            let scale = norm(&free(elastic));
            let residual = norm(&equilibrium) / scale;
            let defect = norm(&continuum_defect) / scale;
            let error = acceleration
                .iter()
                .zip(&continuum_acceleration)
                .map(|(actual, exact)| (actual - exact).powi(2))
                .sum::<f64>();
            let acceleration_error =
                (error / mesh.topology.nodes.len() as f64).sqrt() / exact[2].abs();
            assert!(
                residual <= 1.0e-10,
                "quadratic={quadratic} cells={cells} residual={residual:e}"
            );
            assert!(
                defect.is_finite() && defect > 256.0 * f64::EPSILON,
                "the sampled continuum acceleration is not a discrete equilibrium: {defect:e}"
            );
            eprintln!("initial sine quadratic={quadratic} cells={cells} continuum_balance_defect={defect:e} equilibrium_residual={residual:e} initial_acceleration_error={acceleration_error:e}");
        }
    }
}
