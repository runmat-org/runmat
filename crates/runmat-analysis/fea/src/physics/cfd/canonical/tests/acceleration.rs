use super::*;
use crate::solve::fixed_point::AndersonOnePolicy;

#[test]
fn physical_merit_and_outlet_admissibility_reject_speculative_images() {
    // Deliberately supplied history defects isolate the retention safeguard;
    // they are not claimed to be an actual SIMPLE trajectory or physics truth.
    for backflow in [false, true] {
        let mut f = case(1, ReferenceCase::Couette, false);
        f.policy.fixed_point_acceleration = Some(AndersonOnePolicy::default());
        let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
        let n = p.topology.cells.len();
        let exact_velocity = p
            .topology
            .cells
            .iter()
            .map(|cell| reference::velocity(f.reference, cell.centroid_m, false))
            .collect::<Vec<_>>();
        let original_flux = pressure::h_flux(&p, &exact_velocity, &vec![[0.0; 3]; n]).unwrap();
        let mut history = None;
        for (offset, defect) in [(0.1, 2.0), (0.001, 1.0)] {
            let mut velocity = exact_velocity.clone();
            velocity[0][0] += offset;
            let mut pressure = vec![0.0; n];
            let mut flux = original_flux.clone();
            if backflow && defect == 2.0 {
                let outlet = p
                    .boundaries
                    .iter()
                    .position(|boundary| {
                        matches!(boundary, boundary::FaceBoundary::PressureOutlet { .. })
                    })
                    .unwrap();
                flux[outlet] += 1.0;
            }
            let expected = (velocity.clone(), pressure.clone(), flux.clone());
            let ordinary = recovery::evaluate(&p, &velocity, &pressure, &flux).unwrap();
            assert!(!ordinary.accepted(f.policy.physical_relative_tolerance));
            let retained = super::super::acceleration::consider(
                &p,
                super::super::acceleration::Fields {
                    velocity: &mut velocity,
                    pressure: &mut pressure,
                    flux: &mut flux,
                },
                vec![defect; n],
                &mut history,
                ordinary.clone(),
                None,
            )
            .unwrap();
            assert_eq!(retained, ordinary, "backflow={backflow}, defect={defect}");
            assert_eq!((velocity, pressure, flux), expected);
        }
    }
}

#[test]
fn speculative_coefficient_overflow_keeps_validated_ordinary_fields() {
    let mut f = case(1, ReferenceCase::Couette, false);
    f.policy.fixed_point_acceleration = Some(AndersonOnePolicy::default());
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let n = p.topology.cells.len();
    let mut velocity = p
        .topology
        .cells
        .iter()
        .map(|cell| reference::velocity(f.reference, cell.centroid_m, false))
        .collect::<Vec<_>>();
    let mut pressure = vec![0.0; n];
    let mut flux = pressure::h_flux(&p, &velocity, &vec![[0.0; 3]; n]).unwrap();
    velocity[0][0] += 1e-4;
    let expected = (velocity.clone(), pressure.clone(), flux.clone());
    let original = recovery::evaluate(&p, &velocity, &pressure, &flux).unwrap();
    assert!(!original.accepted(f.policy.physical_relative_tolerance));
    let mut history = None;
    for defect in [1e308, -1e308] {
        let retained = super::super::acceleration::consider(
            &p,
            super::super::acceleration::Fields {
                velocity: &mut velocity,
                pressure: &mut pressure,
                flux: &mut flux,
            },
            vec![defect; n],
            &mut history,
            original.clone(),
            None,
        )
        .unwrap();
        assert_eq!(retained, original);
        assert_eq!(
            (&velocity, &pressure, &flux),
            (&expected.0, &expected.1, &expected.2)
        );
    }
    assert!(history.is_none());
}

#[test]
fn acceleration_policy_is_explicit_bound_and_admitted_before_work() {
    let mut f = case(1, ReferenceCase::Couette, false);
    let ordinary = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let original_identity = ordinary.numerical_input_identity;
    f.policy.fixed_point_acceleration = Some(AndersonOnePolicy::default());
    let accelerated = CanonicalFluidProblem::prepare(f.input()).unwrap();
    assert_ne!(original_identity, accelerated.numerical_input_identity);
    let encoded = serde_json::to_value(f.policy).unwrap();
    assert_eq!(
        serde_json::from_value::<CanonicalFluidPolicy>(encoded.clone()).unwrap(),
        f.policy
    );
    let mut missing = encoded;
    missing
        .as_object_mut()
        .unwrap()
        .remove("fixed_point_acceleration");
    assert!(serde_json::from_value::<CanonicalFluidPolicy>(missing).is_err());
    for coefficient in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        f.policy.fixed_point_acceleration = Some(AndersonOnePolicy {
            maximum_absolute_coefficient: coefficient,
        });
        assert!(matches!(
            CanonicalFluidProblem::prepare(f.input()),
            Err(CanonicalFluidError::Invalid(_))
        ));
    }
}

#[test]
fn additional_acceleration_storage_and_work_cannot_escape_ceilings() {
    let f = case(1, ReferenceCase::Couette, false);
    let geometry = crate::topology::FiniteVolumeTopology::reserve(
        &f.source,
        f.topology.policy,
        &f.topology.selection,
    )
    .unwrap();
    let faces = f.topology.faces.len();
    let baseline = CanonicalFluidReservation::for_counts(&geometry, faces, f.policy).unwrap();
    let mut policy = f.policy;
    policy.fixed_point_acceleration = Some(AndersonOnePolicy::default());
    let accelerated = CanonicalFluidReservation::for_counts(&geometry, faces, policy).unwrap();
    let n = geometry.cell_count as u64;
    let m = faces as u64;
    assert_eq!(
        accelerated.resident_bytes_upper_bound - baseline.resident_bytes_upper_bound,
        16 * (26 * n + 15 * m) + 4096
    );
    assert_eq!(
        accelerated.operator_work_upper_bound - baseline.operator_work_upper_bound,
        policy.maximum_outer_iterations as u64 * (10000 * m + 100100 * n)
    );
    assert_eq!(
        accelerated.linear_iterations_upper_bound,
        baseline.linear_iterations_upper_bound
    );
    policy.maximum_resident_bytes = Some(accelerated.resident_bytes_upper_bound - 1);
    assert!(matches!(
        CanonicalFluidReservation::for_counts(&geometry, faces, policy),
        Err(CanonicalFluidError::Resource("resident bytes"))
    ));
    policy.maximum_resident_bytes = Some(accelerated.resident_bytes_upper_bound);
    policy.maximum_operator_work = Some(accelerated.operator_work_upper_bound - 1);
    assert!(matches!(
        CanonicalFluidReservation::for_counts(&geometry, faces, policy),
        Err(CanonicalFluidError::Resource("operator work"))
    ));
    policy.maximum_operator_work = Some(accelerated.operator_work_upper_bound);
    assert_eq!(
        CanonicalFluidReservation::for_counts(&geometry, faces, policy).unwrap(),
        accelerated
    );
}

#[test]
fn accelerated_affine_reference_retains_original_equations_and_cold_binding() {
    for rotated in [false, true] {
        let mut f = case(2, ReferenceCase::Couette, rotated);
        f.policy.fixed_point_acceleration = Some(AndersonOnePolicy::default());
        let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
        let result = p.solve().unwrap();
        assert!(result
            .residual
            .accepted(f.policy.physical_relative_tolerance));
        assert!(l2(&f, &result) < 1e-6);
        assert_eq!(p.recover(&result).unwrap(), result.residual);
        let cold: CanonicalFluidResult =
            serde_json::from_slice(&serde_json::to_vec(&result).unwrap()).unwrap();
        assert_eq!(p.recover(&cold).unwrap(), result.residual);
        f.policy.fixed_point_acceleration = None;
        let ordinary = CanonicalFluidProblem::prepare(f.input()).unwrap();
        assert!(ordinary.recover(&cold).is_err());
    }
}
