//! Original pressure survives gauge shifts, fresh solves and cold recovery.
//! This diagnostic does not replace the authored force-balance qualification gate.
use super::*;
use runmat_analysis_core::{
    BodyForceDensity, BoundaryOperator, FluidConstraintKind, PhysicalSignal, PrescribedValue,
    Pressure, SignalRepresentation, SignalValue, Velocity,
};

// witness: FEA-FLUID-EQUATIONS-001/pressure_state @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel; coverage: partial;
//   boundary: A uniform translation is balanced at both pressure references.
#[test]
fn independent_uniform_translation_is_balanced_at_both_pressure_references() {
    for reference in [0.0, 101325.0] {
        let mut f = case(2, ReferenceCase::Couette, false);
        let exact = [0.001, 0.0, 0.0];
        for constraint in &mut f.constraints {
            let FluidConstraintKind::PrescribedVelocity { value, .. } = &mut constraint.kind else {
                panic!("original velocity authority");
            };
            *value = PrescribedValue::Constant(exact.map(Velocity::new));
        }
        for boundary in &mut f.operators {
            let BoundaryOperator::FluidOutletPressure { value } = &mut boundary.operator else {
                panic!("original outlet authority");
            };
            *value = PrescribedValue::Constant(Pressure::new(reference));
        }
        let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
        // Independent continuum solution: constant U/p, zero gradient/stress;
        // oriented volume flux is the exact constant vector dotted with face area.
        let velocity = vec![exact; f.topology.cells.len()];
        let pressure = vec![0.0; velocity.len()];
        let flux = f
            .topology
            .faces
            .iter()
            .map(|face| geometry_ops::dot(exact, face.owner_area_vector_m2))
            .collect::<Vec<_>>();
        let residual = recovery::evaluate(&p, &velocity, &pressure, &flux).unwrap();
        eprintln!(
            "independent translation gauge={reference}: worst_cell={:e}, momentum={:e}, flux={:e}",
            residual
                .cell_momentum_relative
                .iter()
                .copied()
                .fold(0.0, f64::max),
            residual.momentum_relative,
            residual.flux_constitutive_relative
        );
        assert!(residual.accepted(1e-8));
    }
}

// witness: FEA-FLUID-EQUATIONS-001/pressure_state @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Fourier duct scaling and two gauges retain bitwise original velocity/fluctuation/flux
//   with and without acceleration; cold JSON replay only.
#[test]
fn original_fluctuation_survives_absolute_rounding_fresh_solve_and_cold_recovery() {
    for acceleration in [
        None,
        Some(crate::solve::fixed_point::AndersonOnePolicy::default()),
    ] {
        verify_original_pressure(acceleration);
    }
}

fn verify_original_pressure(acceleration: Option<crate::solve::fixed_point::AndersonOnePolicy>) {
    let mut f = case(2, ReferenceCase::Duct, false);
    // Scaling rho, mu and the original force together preserves velocity and Re.
    // The Fourier boundary solution is independent of this dimensional scaling.
    f.policy.fixed_point_acceleration = acceleration;
    let scale = 1e-5;
    f.material.density =
        runmat_types::physical::Quantity::new(f.material.density.canonical_value() * (scale));
    f.material.dynamic_viscosity = runmat_types::physical::Quantity::new(
        f.material.dynamic_viscosity.canonical_value() * (scale),
    );
    let PhysicalSignal::BodyForceDensity(signal) = &mut f.signals[0] else {
        panic!("original duct force authority");
    };
    signal.representation = SignalRepresentation::Constant(SignalValue::Vector3([
        BodyForceDensity::new(scale),
        BodyForceDensity::new(0.0),
        BodyForceDensity::new(0.0),
    ]));
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let original = p.solve().unwrap();
    let baseline = p.recover(&original).unwrap();
    assert!(baseline.accepted(f.policy.physical_relative_tolerance));
    drop(p);

    let reference = 101325.0;
    for boundary in &mut f.operators {
        let BoundaryOperator::FluidOutletPressure { value } = &mut boundary.operator else {
            panic!("original duct outlet authority");
        };
        *value = PrescribedValue::Constant(Pressure::new(reference));
    }
    let shifted = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let pressure = original
        .pressure_pa
        .canonical_values()
        .iter()
        .map(|v| reference + v)
        .collect::<Vec<_>>();
    let lost = original
        .pressure_pa
        .canonical_values()
        .iter()
        .zip(&pressure)
        .map(|(a, b)| (a - (b - reference)).abs())
        .fold(0.0, f64::max);
    let residual = recovery::evaluate(
        &shifted,
        original.velocity_m_per_s.canonical_values(),
        original.pressure_fluctuation_pa.canonical_values(),
        original.face_volume_flux_m3_per_s.canonical_values(),
    )
    .unwrap();
    let worst =
        |r: &CanonicalFluidResidual| r.cell_momentum_relative.iter().copied().fold(0.0, f64::max);
    eprintln!("frozen pressure gauge: rounding_pa={lost:e}, baseline_cell={:e}, shifted_cell={:e}, baseline_flux={:e}, shifted_flux={:e}",
        worst(&baseline), worst(&residual), baseline.flux_constitutive_relative,
        residual.flux_constitutive_relative);
    assert!(lost > 0.0);
    assert_eq!(
        residual.cell_momentum_n.canonical_values(),
        baseline.cell_momentum_n.canonical_values()
    );
    assert_eq!(
        residual.cell_momentum_relative,
        baseline.cell_momentum_relative
    );
    assert_eq!(
        residual.flux_constitutive_relative,
        baseline.flux_constitutive_relative
    );
    assert!(residual.accepted(f.policy.physical_relative_tolerance));

    let fresh = shifted.solve().unwrap();
    assert_eq!(
        fresh.velocity_m_per_s.canonical_values(),
        original.velocity_m_per_s.canonical_values()
    );
    assert_eq!(
        fresh.face_volume_flux_m3_per_s.canonical_values(),
        original.face_volume_flux_m3_per_s.canonical_values()
    );
    assert_eq!(
        fresh.pressure_fluctuation_pa.canonical_values(),
        original.pressure_fluctuation_pa.canonical_values()
    );
    assert_eq!(fresh.outer_iterations, original.outer_iterations);
    assert_eq!(fresh.linear_iterations, original.linear_iterations);
    assert_eq!(fresh.pressure_pa.canonical_values().as_slice(), pressure);
    let encoded = serde_json::to_vec(&fresh).unwrap();
    let cold: CanonicalFluidResult = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(cold, fresh);
    assert_eq!(shifted.recover(&cold).unwrap(), residual);
    assert!(shifted.recover(&original).is_err());

    // Rounding of the public absolute projection must never erase authority.
    let mut reconstructed = cold.clone();
    reconstructed.pressure_fluctuation_pa = runmat_types::physical::Quantity::new(
        pressure.iter().map(|v| v - reference).collect::<Vec<_>>(),
    );
    assert!(shifted.recover(&reconstructed).is_err());
    let mut substituted = cold.clone();
    substituted.pressure_pa = substituted
        .pressure_pa
        .try_map_canonical_storage(|mut values| {
            values[0] += 1.0;
            values
        })
        .unwrap();
    assert!(shifted.recover(&substituted).is_err());
    let mut missing = serde_json::to_value(&cold).unwrap();
    missing
        .as_object_mut()
        .unwrap()
        .remove("pressure_fluctuation_pa");
    assert!(serde_json::from_value::<CanonicalFluidResult>(missing).is_err());
}
