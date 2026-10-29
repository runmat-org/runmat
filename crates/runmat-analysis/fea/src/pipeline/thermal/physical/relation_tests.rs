//! Periodic and temperature-jump relations against independent closed forms.
use runmat_analysis_core::*;

use crate::physics::thermal::qualification::{mesh, workload};

fn periodic(jump_k: Option<f64>) -> ThermalConstraint {
    let periodic = PeriodicConstraint {
        pair: PairedConstraint {
            primary: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_left",
            )),
            secondary: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_right",
            )),
            pairing: PairingPolicy::GeometricTolerance(Length::new(1e-9)),
        },
        transform: PeriodicTransform::Translation(Translation::new([1.0, 0.0, 0.0])),
    };
    ThermalConstraint {
        constraint_id: "left-right".into(),
        kind: match jump_k {
            Some(jump) => ThermalConstraintKind::AffineJump {
                periodic,
                temperature_jump: PrescribedValue::Constant(TemperatureDifference::new(jump)),
            },
            None => ThermalConstraintKind::Periodic(periodic),
        },
    }
}

fn convective_left(input: &mut crate::execution::ThermalWorkload, ambient_k: f64) {
    input
        .model
        .boundaries
        .retain(|b| b.boundary_id == "insulated-sides");
    input.model.boundaries.push(AnalysisBoundary {
        boundary_id: "left-film".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        operator: BoundaryOperator::ThermalConvection {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(ambient_k)),
            coefficient: HeatTransferCoefficient::new(25.0),
        },
    });
}

// witness: FEA-THERMAL-DOMAIN-001/constraints @ thermal_steady/tet4|tet10/numerical_kernel;
//   coverage: partial;
//   boundary: A periodic cell with a 20 K temperature jump and a convective left face reproduces T
//   = 300 + 20 x exactly (within 1e-8 relative) on linear and quadratic meshes, with energy-balance
//   residuals within target; tied and affine relations beyond one periodic pair are not covered.
#[test]
fn temperature_jump_carries_a_linear_profile_through_a_periodic_cell() {
    // The right face sits 20 K above the left one, which exchanges heat with a 300 K film.
    // The heat that crosses the cell returns through the jump, so the film carries no net
    // heat: the left face sits at the ambient and T = 300 + 20 x exactly.
    for quadratic in [false, true] {
        let mesh = mesh(quadratic);
        let mut input = workload();
        convective_left(&mut input, 300.0);
        input
            .model
            .constraints
            .push(AnalysisConstraint::Thermal(periodic(Some(20.0))));
        let result = input.execute(&mesh).unwrap();
        for (node, &temperature) in mesh
            .topology
            .nodes
            .iter()
            .zip(result.temperature_snapshots[0].as_host_f64().unwrap())
        {
            let expected = 300.0 + 20.0 * node.coordinates_m.canonical_value()[0];
            assert!(
                (temperature - expected).abs() < 1e-8 * 300.0,
                "quadratic={quadratic} x={} actual={temperature} expected={expected}",
                node.coordinates_m.canonical_value()[0]
            );
        }
        for balance in &result.energy_balances {
            assert!(balance.relative_residual <= input.residual_target);
        }
        result.validate_complete().unwrap();
    }
}

// witness: FEA-THERMAL-DOMAIN-001/constraints @ thermal_steady/tet4/numerical_kernel;
//   coverage: partial;
//   boundary: A source-free periodic cell settles at the film ambient temperature.
#[test]
fn periodic_cell_without_sources_settles_at_the_film_ambient() {
    // No source acts and the right face follows the left one, so the left film holds the
    // whole cell at its 330 K ambient.
    let mesh = mesh(false);
    let mut input = workload();
    convective_left(&mut input, 330.0);
    input
        .model
        .constraints
        .push(AnalysisConstraint::Thermal(periodic(None)));
    let result = input.execute(&mesh).unwrap();
    for &temperature in result.temperature_snapshots[0].as_host_f64().unwrap() {
        assert!((temperature - 330.0).abs() < 1e-8 * 330.0, "{temperature}");
    }
}

// witness: FEA-THERMAL-DOMAIN-001/constraints @ thermal_steady/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Radiation Newton with a periodic jump matches the independent scalar balance of film
//   and gray-surface radiation and the exact linear profile on Tet4 and Tet10; steady only.
#[test]
fn radiating_periodic_cell_balances_film_and_radiation_through_the_jump() {
    // The right face radiates and sits 20 K above the left one, which exchanges heat with a
    // 300 K film. No source acts, so the film gain equals the radiated loss:
    // h (T_L - 300) + eps sigma ((T_L + 20)^4 - 300^4) = 0, with T = T_L + 20 x exactly.
    use crate::physics::thermal::qualification::{AMBIENT, EMISSIVITY, SIGMA};
    let h = 25.0;
    let balance =
        |t: f64| h * (t - 300.0) + EMISSIVITY * SIGMA * ((t + 20.0).powi(4) - AMBIENT.powi(4));
    let (mut low, mut high) = (200.0_f64, 300.0_f64);
    assert!(balance(low) < 0.0 && balance(high) > 0.0);
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if balance(middle) > 0.0 {
            high = middle;
        } else {
            low = middle;
        }
    }
    let left = 0.5 * (low + high);
    for quadratic in [false, true] {
        let mesh = mesh(quadratic);
        let mut input = workload();
        input.model.boundaries.push(AnalysisBoundary {
            boundary_id: "left-film".into(),
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_left",
            )),
            operator: BoundaryOperator::ThermalConvection {
                ambient_temperature: PrescribedValue::Constant(Temperature::new(300.0)),
                coefficient: HeatTransferCoefficient::new(h),
            },
        });
        input
            .model
            .constraints
            .push(AnalysisConstraint::Thermal(periodic(Some(20.0))));
        let result = input.execute(&mesh).unwrap();
        for (node, &temperature) in mesh
            .topology
            .nodes
            .iter()
            .zip(result.temperature_snapshots[0].as_host_f64().unwrap())
        {
            let expected = left + 20.0 * node.coordinates_m.canonical_value()[0];
            assert!(
                (temperature - expected).abs() < 1e-7 * 300.0,
                "quadratic={quadratic} x={} actual={temperature} expected={expected}",
                node.coordinates_m.canonical_value()[0]
            );
        }
        assert_eq!(result.nonlinear_evidence.len(), 1);
        for balance in &result.energy_balances {
            assert!(balance.relative_residual <= input.residual_target);
        }
        result.validate_complete().unwrap();
    }
}
