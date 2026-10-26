//! Independent continuum references, not a lumped-body cooling surrogate.
use super::*;
use crate::execution::ThermalWorkload;
use runmat_analysis_core::*;
use runmat_meshing_core::SolverMeshArtifact;

mod initial;
mod retention;
mod temporal;

use crate::physics::thermal::qualification::{left_temperature, mesh, workload, AMBIENT, TARGET};

fn assert_acceptance(result: &FeaThermalRunResult, solved: usize) {
    assert_eq!(result.nonlinear_evidence.len(), solved);
    assert_eq!(result.residual_norms.len(), solved);
    assert_eq!(result.energy_balances.len(), solved);
    for evidence in &result.nonlinear_evidence {
        assert!((1..=MAX_CORRECTIONS).contains(&evidence.iteration_count));
        assert!(evidence.backtracking_count <= MAX_CORRECTIONS * MAX_TRIALS);
        assert!(evidence.relative_equilibrium_residual <= TARGET);
        assert!(evidence.relative_increment <= TARGET);
        assert!(
            evidence.power_scale_w.canonical_value().is_finite()
                && evidence.power_scale_w.canonical_value() >= 0.0
        );
    }
    assert!(result
        .residual_norms
        .iter()
        .all(|value| value.is_finite() && *value <= TARGET));
    assert!(result
        .energy_balances
        .iter()
        .all(|balance| balance.relative_residual <= TARGET));
    result.validate_complete().unwrap();
}

#[test]
fn fixed_hot_slab_matches_independent_gray_surface_scalar_equilibrium() {
    // Exact continuum T is linear. Independent monotone bisection solves the surface law,
    // not the RunMat discrete operator or Newton routine, on a unit-length/unit-area slab.
    let (surface, power) =
        crate::physics::thermal::qualification::reference::steady_surface_and_power();
    for quadratic in [false, true] {
        let mesh = mesh(quadratic);
        let mut input = workload();
        left_temperature(
            &mut input,
            PrescribedValue::Constant(Temperature::new(400.0)),
        );
        let result = input.execute(&mesh).unwrap();
        assert_acceptance(&result, 1);
        retention::verify(&input, &mesh, &result);
        for (node, &temperature) in mesh
            .topology
            .nodes
            .iter()
            .zip(result.temperature_snapshots[0].as_host_f64().unwrap())
        {
            let expected = 400.0 + (surface - 400.0) * node.coordinates_m.canonical_value()[0];
            assert!(
                (temperature - expected).abs() / 400.0 < 1.0e-8,
                "quadratic={quadratic} x={} actual={temperature} expected={expected}",
                node.coordinates_m.canonical_value()[0]
            );
        }
        let balance = result.energy_balances[0];
        assert!((balance.boundary_loss_w.canonical_value() / power - 1.0).abs() < 1.0e-8);
        assert!(
            (balance.prescribed_boundary_power_w.canonical_value() / power + 1.0).abs() < 1.0e-8
        );
        assert_eq!(balance.applied_power_w.canonical_value(), 0.0);
        assert_eq!(balance.storage_rate_w.canonical_value(), 0.0);
        for gradient in result.temperature_gradient_snapshots[0]
            .as_host_f64()
            .unwrap()
            .chunks_exact(3)
        {
            assert!(
                norm(
                    gradient
                        .iter()
                        .zip([surface - 400.0, 0.0, 0.0])
                        .map(|(actual, expected)| actual - expected)
                ) / (400.0 - surface)
                    < 1.0e-8
            );
        }
        for flux in result.heat_flux_snapshots[0]
            .as_host_f64()
            .unwrap()
            .chunks_exact(3)
        {
            assert!(
                norm(
                    flux.iter()
                        .zip([power, 0.0, 0.0])
                        .map(|(actual, expected)| actual - expected)
                ) / power
                    < 1.0e-8
            );
        }
    }
}

#[test]
fn cold_initial_guess_uses_real_backtracking_and_rejected_runs_publish_nothing() {
    let mesh = mesh(false);
    let mut input = workload();
    input.model.thermal.as_mut().unwrap().reference_temperature = Temperature::new(1.0);
    left_temperature(
        &mut input,
        PrescribedValue::Constant(Temperature::new(400.0)),
    );
    let result = input.execute(&mesh).unwrap();
    assert_acceptance(&result, 1);
    assert!(
        result.nonlinear_evidence[0].backtracking_count > 0,
        "the cold start must exercise damping, not only full Newton steps"
    );
    assert!(result.temperature_snapshots[0]
        .as_host_f64()
        .unwrap()
        .iter()
        .all(|temperature| *temperature >= AMBIENT && *temperature <= 400.0));

    let mut invalid = input.clone();
    invalid
        .model
        .thermal
        .as_mut()
        .unwrap()
        .reference_temperature = Temperature::new(-1.0);
    assert!(invalid.execute(&mesh).is_err());
    let mut gpu = input.clone();
    gpu.backend = ComputeBackend::Gpu;
    assert!(
        gpu.execute(&mesh).is_err(),
        "radiation must reject GPU before any CPU fallback"
    );
    let mut inadmissible = input;
    inadmissible.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = 1;
    assert!(inadmissible.execute(&mesh).is_err());
}

fn manufactured(intervals: usize, mesh: &SolverMeshArtifact) -> ThermalWorkload {
    crate::physics::thermal::qualification::cooling::manufactured(
        intervals,
        initial::solved_linear_profile(mesh),
    )
}
