use super::*;
use runmat_analysis_core::{
    BoundaryOperator, FluidConstraint, FluidConstraintKind, PrescribedValue, Pressure, Velocity,
};

pub(super) fn closed(n: usize, velocity: [f64; 3], reference: f64) -> fixture::Fixture {
    let mut f = case(n, ReferenceCase::Couette, false);
    for c in &mut f.constraints {
        let FluidConstraintKind::PrescribedVelocity { value, .. } = &mut c.kind else {
            panic!("original velocity")
        };
        *value = PrescribedValue::Constant(velocity.map(Velocity::new));
    }
    for boundary in f.operators.drain(..) {
        f.constraints.push(FluidConstraint {
            constraint_id: format!("closed-{}", boundary.boundary_id),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: boundary.target,
                value: PrescribedValue::Constant(velocity.map(Velocity::new)),
            },
        });
    }
    let FluidConstraintKind::PrescribedVelocity { target, .. } = &f.constraints[0].kind else {
        unreachable!()
    };
    f.constraints.push(FluidConstraint {
        constraint_id: "original-pressure-gauge".into(),
        kind: FluidConstraintKind::PressureReference {
            target: target.clone(),
            value: PrescribedValue::Constant(Pressure::new(reference)),
        },
    });
    f
}

#[test]
fn closed_original_uniform_translation_and_mean_trace_survive_cold_recovery() {
    for n in [2, 3] {
        for reference in [0.0, 101325.0] {
            let velocity = [0.001, 0.002, -0.0005];
            let f = closed(n, velocity, reference);
            let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
            let original = p.solve().unwrap();
            for u in original.velocity_m_per_s.canonical_values() {
                assert!(norm(sub(*u, velocity)) < 1e-8);
            }
            assert!(original
                .pressure_fluctuation_pa
                .canonical_values()
                .iter()
                .all(|v| v.abs() < 1e-7));
            let cold: CanonicalFluidResult =
                serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
            assert_eq!(p.recover(&cold).unwrap(), original.residual);
            assert!(original
                .residual
                .boundary_balances
                .iter()
                .all(|b| b.authority_id != "original-pressure-gauge"));
            let mut changed = cold;
            changed.pressure_fluctuation_pa = changed
                .pressure_fluctuation_pa
                .try_map_canonical_storage(|values| {
                    values
                        .into_iter()
                        .map(|value| value + 1.0)
                        .collect::<Vec<_>>()
                })
                .unwrap();
            changed.pressure_pa = changed
                .pressure_pa
                .try_map_canonical_storage(|_| {
                    changed
                        .pressure_fluctuation_pa
                        .canonical_values()
                        .iter()
                        .map(|value| reference + value)
                        .collect::<Vec<_>>()
                })
                .unwrap();
            assert!(p.recover(&changed).is_err());
        }
    }
}

// witness: FEA-FLUID-EQUATIONS-001/coupling @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Closed gauge/reference and net flux rejection; not full pressure-velocity envelope
//   proof.
#[test]
fn closed_gauge_admission_rejects_missing_duplicate_outlet_and_incompatible_flux() {
    let mut empty = closed(2, [0.0; 3], 0.0);
    empty.constraints.last_mut().unwrap().constraint_id.clear();
    assert!(CanonicalFluidProblem::prepare(empty.input()).is_err());
    let mut f = closed(1, [0.0; 3], 0.0);
    f.constraints.pop();
    assert!(CanonicalFluidProblem::prepare(f.input()).is_err());
    let mut f = closed(1, [0.0; 3], 0.0);
    let mut duplicate = f.constraints.last().unwrap().clone();
    duplicate.constraint_id = "second-gauge".into();
    f.constraints.push(duplicate);
    assert!(CanonicalFluidProblem::prepare(f.input()).is_err());
    let mut f = closed(1, [0.0; 3], 0.0);
    let c = f.constraints.remove(0);
    let FluidConstraintKind::PrescribedVelocity { target, .. } = c.kind else {
        unreachable!()
    };
    f.operators.push(runmat_analysis_core::AnalysisBoundary {
        boundary_id: "physical-outlet".into(),
        target,
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(0.0)),
        },
    });
    assert!(CanonicalFluidProblem::prepare(f.input()).is_err());
    let mut f = closed(1, [0.0; 3], 0.0);
    let FluidConstraintKind::PrescribedVelocity { value, .. } = &mut f.constraints[0].kind else {
        unreachable!()
    };
    *value = PrescribedValue::Constant([1.0, 2.0, 3.0].map(Velocity::new));
    assert!(CanonicalFluidProblem::prepare(f.input()).is_err());
}

#[test]
fn cancelled_closed_solve_returns_no_physical_result() {
    let f = closed(2, [0.0; 3], 0.0);
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let cancellation = crate::progress::FeaCancellationToken::new();
    cancellation.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation,
            ..Default::default()
        });
    assert!(matches!(p.solve(), Err(CanonicalFluidError::Cancelled)));
}

#[test]
fn closed_coplanar_cell_pressure_inventory_rejects_without_fake_boundary_samples() {
    let f = closed(1, [0.0; 3], 0.0);
    assert!(matches!(
        CanonicalFluidProblem::prepare(f.input()),
        Err(CanonicalFluidError::InvalidCell {
            reason: "rank-deficient physical gradient stencil",
            ..
        })
    ));
}

#[test]
fn closed_rotational_reference_converges_from_original_affine_velocity_data() {
    let omega = 0.01_f64;
    let mut errors = Vec::new();
    let mut pressure_errors = Vec::new();
    for n in [2, 3] {
        let mut f = closed(n, [0.0; 3], 0.0);
        for constraint in &mut f.constraints {
            if let FluidConstraintKind::PrescribedVelocity { target, value } = &mut constraint.kind
            {
                let face = f
                    .source
                    .boundary_faces
                    .iter()
                    .find(|face| {
                        face.provenance
                            .iter()
                            .any(|e| target.matches_persistent_entity(e))
                    })
                    .unwrap();
                *value = PrescribedValue::Constant(
                    [-omega * face.centroid_m[1], omega * face.centroid_m[0], 0.0]
                        .map(Velocity::new),
                );
            }
        }
        let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
        let result = p.solve().unwrap();
        p.recover(&result).unwrap();
        let gradients =
            geometry_ops::pressure_gradient(&p, result.pressure_fluctuation_pa.canonical_values())
                .unwrap();
        let mut ue = 0.0;
        let mut us = 0.0;
        let mut pe = 0.0;
        let mut ps = 0.0;
        for ((cell, u), g) in f
            .topology
            .cells
            .iter()
            .zip(result.velocity_m_per_s.canonical_values())
            .zip(gradients)
        {
            let exact = [-omega * cell.centroid_m[1], omega * cell.centroid_m[0], 0.0];
            let gp = [
                omega * omega * cell.centroid_m[0],
                omega * omega * cell.centroid_m[1],
                0.0,
            ];
            ue += cell.volume_m3 * norm(sub(*u, exact)).powi(2);
            us += cell.volume_m3 * norm(exact).powi(2);
            pe += cell.volume_m3 * norm(sub(g, gp)).powi(2);
            ps += cell.volume_m3 * norm(gp).powi(2);
        }
        let error = (ue / us).sqrt();
        let pressure_error = (pe / ps).sqrt();
        eprintln!("closed rotating reference n={n} velocity={error:e} pressure_gradient={pressure_error:e}");
        errors.push(error);
        pressure_errors.push(pressure_error);
    }
    assert!(
        errors[1] < errors[0] && errors[1] < 0.05,
        "independent rotating velocity {errors:?}"
    );
    assert!(
        pressure_errors[1] < pressure_errors[0] && pressure_errors[1] < 0.35,
        "independent centrifugal pressure gradient {pressure_errors:?}"
    );
}
