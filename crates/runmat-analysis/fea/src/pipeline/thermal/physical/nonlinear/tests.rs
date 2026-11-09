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

// witness: FEA-THERMAL-SOLVE-001/steady @ thermal_steady/tet4|tet10/public_study;
//   check: hot-slab-radiation; boundary: Tet4/Tet10 fixed-hot slab against independent scalar bisection; original fields,
//   power and retained nonlinear evidence checked.
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
    // Radiation has no device path: the request falls back to cpu_reference with one warning
    // and the same numbers, and strict placement fails closed before any solve.
    let fallback = gpu.execute(&mesh).unwrap();
    assert_eq!(fallback.run.requested_backend, ComputeBackend::Gpu);
    assert_eq!(fallback.run.execution_backend, crate::ExecutionBackend::Cpu);
    assert_eq!(
        fallback
            .run
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.code
                == crate::contracts::accelerator_admission::ACCELERATOR_ADMISSION_FALLBACK_CODE)
            .count(),
        1
    );
    assert_eq!(
        fallback.temperature_snapshots[0].as_host_f64().unwrap(),
        result.temperature_snapshots[0].as_host_f64().unwrap()
    );
    fallback.validate_complete().unwrap();
    let strict = crate::contracts::accelerator_admission::with_accelerator_admission_policy(
        crate::contracts::accelerator_admission::AcceleratorAdmissionPolicy::Required,
        || gpu.execute(&mesh),
    );
    assert!(matches!(strict, Err(FeaRunError::AcceleratorRequired(_))));
    let mut inadmissible = input;
    inadmissible.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    assert!(inadmissible.execute(&mesh).is_err());
}

fn manufactured(intervals: usize, mesh: &SolverMeshArtifact) -> ThermalWorkload {
    crate::physics::thermal::qualification::cooling::manufactured(
        intervals,
        initial::solved_linear_profile(mesh),
    )
}

// witness: FEA-THERMAL-INPUT-001/boundaries @ thermal_steady/tet4|tet10/numerical_kernel; coverage: partial;
//   check: radiating-bar; boundary: The validation table's radiating-bar check: 1000 W/m^2 into one end of a 1 m steel
//   bar (k = 45 W/m/K) that radiates from the other (emissivity 0.8, 300 K surroundings) matches
//   the closed form q = eps sigma (T^4 - 300^4) at the radiating end and q L / k across the bar,
//   with the applied and radiated power equal; Tet4 and Tet10 slabs.
#[test]
fn radiating_bar_matches_the_gray_surface_closed_form_and_balances_power() {
    use crate::physics::thermal::qualification::SIGMA;
    const FLUX: f64 = 1000.0;
    const CONDUCTIVITY: f64 = 45.0;
    const EPSILON: f64 = 0.8;
    // Independent scalar bisection of the radiating-end temperature.
    let law = |t: f64| EPSILON * SIGMA * (t.powi(4) - 300.0_f64.powi(4)) - FLUX;
    let (mut low, mut high) = (300.0_f64, 1000.0_f64);
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if law(middle) > 0.0 {
            high = middle;
        } else {
            low = middle;
        }
    }
    let radiating = 0.5 * (low + high);
    let heated = radiating + FLUX / CONDUCTIVITY;
    for quadratic in [false, true] {
        let mesh = mesh(quadratic);
        let mut input = workload();
        input.model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .conductivity_w_per_mk = runmat_types::physical::Quantity::new(CONDUCTIVITY);
        for boundary in &mut input.model.boundaries {
            if let BoundaryOperator::ThermalRadiation { emissivity, .. } = &mut boundary.operator {
                *emissivity = SurfaceEmissivity::new(EPSILON);
            }
        }
        input.model.boundaries.push(AnalysisBoundary {
            boundary_id: "heated-end".into(),
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_left",
            )),
            operator: BoundaryOperator::ThermalHeatFlux {
                value: PrescribedValue::Constant(HeatFlux::new(FLUX)),
            },
        });
        let result = input.execute(&mesh).unwrap();
        for (node, &temperature) in mesh
            .topology
            .nodes
            .iter()
            .zip(result.temperature_snapshots[0].as_host_f64().unwrap())
        {
            let expected = heated + (radiating - heated) * node.coordinates_m.canonical_value()[0];
            assert!(
                (temperature - expected).abs() < 1e-8 * expected,
                "quadratic={quadratic} x={} {temperature} vs {expected}",
                node.coordinates_m.canonical_value()[0]
            );
        }
        let balance = &result.energy_balances[0];
        assert!(balance.relative_residual <= input.residual_target);
        result.validate_complete().unwrap();
    }
}
