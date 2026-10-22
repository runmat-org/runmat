use super::*;
#[test]
fn requested_global_plan_and_trial_payloads_cover_actual_allocations_and_precancel_rejects_without_heap(
) {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = Fixture::new(order, true);
        let (operator, retained, construction_peak) =
            crate::topology::measure_test_allocations(|| f.operator(true));
        let resources = operator.resources().unwrap();
        assert_eq!(retained as u64, resources.prepared_resident_bytes);
        assert!(construction_peak as u64 <= resources.construction_peak_additional_bytes);
        let initial = operator.initial_configuration(&f.positions).unwrap();
        let current = f.moved(0.06, 0.02);
        let (trial, retained, peak) = crate::topology::measure_test_allocations(|| {
            operator.assemble_trial(&initial, 0, &current).unwrap()
        });
        assert_eq!(retained as u64, resources.trial_resident_bytes);
        assert!(peak as u64 <= resources.trial_peak_additional_bytes);
        let token = crate::progress::FeaCancellationToken::new();
        token.cancel();
        let _guard =
            crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
                cancellation: token,
                ..Default::default()
            });
        let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
            operator.assemble_trial(&trial.configuration, 1, &current)
        });
        assert!(matches!(result, Err(ContactOperatorError::Cancelled)));
        assert_eq!((retained, peak), (0, 0));
    }
}
// witness: FEA-CONTACT-OPERATOR-001/complementarity_penetration @
//   penalty_coulomb_contact/tet10/numerical_kernel; coverage: partial;
//   boundary: Explicit operator controls and initial-penetration limits are rejected instead of
//   defaulted on a Tet10 fixture; complementarity residuals are not asserted.
#[test]
fn explicit_operator_controls_and_initial_penetration_limits_reject_instead_of_defaulting() {
    let f = Fixture::new(ElementOrder::Tet10, false);
    let mut control = Fixture::control(false);
    control.maximum_initial_penetration_m = f64::NAN;
    assert!(CanonicalContactOperator::prepare(
        &f.projection,
        &f.interfaces,
        &f.rows,
        3 * f.rows.len(),
        &[control],
        Fixture::budget()
    )
    .is_err());
    let operator = f.operator(false);
    assert!(matches!(
        operator.initial_configuration(&f.moved(0.06, 0.)),
        Err(ContactOperatorError::InitialPenetration)
    ));
    let mut budget = Fixture::budget();
    budget.maximum_reference_points = 6;
    assert!(CanonicalContactOperator::prepare(
        &f.projection,
        &f.interfaces,
        &f.rows,
        3 * f.rows.len(),
        &[Fixture::control(false)],
        budget
    )
    .is_err());
    budget = Fixture::budget();
    budget.maximum_refinement_evaluations_per_trial = 1;
    assert!(CanonicalContactOperator::prepare(
        &f.projection,
        &f.interfaces,
        &f.rows,
        3 * f.rows.len(),
        &[Fixture::control(false)],
        budget
    )
    .is_err());
}

#[test]
fn resource_caps_bind_execution_without_changing_physical_configuration_scope() {
    let f = Fixture::new(ElementOrder::Tet10, true);
    let first = f.operator(true);
    let state = first.initial_configuration(&f.positions).unwrap();
    let mut policy = Fixture::budget();
    policy.maximum_coordinates -= 1;
    let second = CanonicalContactOperator::prepare(
        &f.projection,
        &f.interfaces,
        &f.rows,
        3 * f.rows.len(),
        &[Fixture::control(true)],
        policy,
    )
    .unwrap();
    assert_eq!(first.identity(), second.identity());
    assert_ne!(first.execution_identity(), second.execution_identity());
    second.validate_configuration(&state, 0).unwrap();
}

// witness: FEA-CONTACT-OPERATOR-001/surface_search_gap @
//   penalty_coulomb_contact/tet10/numerical_kernel; coverage: partial;
//   boundary: Duplicate secondary laws and conflicting surface roles are rejected on a Tet10
//   fixture; gap evaluation itself is covered in the penalty tests.
#[test]
fn global_duplicate_secondary_laws_and_conflicting_surface_roles_reject() {
    let f = Fixture::new(ElementOrder::Tet10, true);
    let mut duplicate = f.interfaces[0].clone();
    duplicate.interface_id = "second-contact".into();
    let interfaces = vec![f.interfaces[0].clone(), duplicate];
    let controls = [Fixture::control(true); 2];
    assert!(matches!(
        CanonicalContactOperator::prepare(
            &f.projection,
            &interfaces,
            &f.rows,
            3 * f.rows.len(),
            &controls,
            Fixture::budget()
        ),
        Err(ContactOperatorError::Selection(
            ContactSurfaceError::SurfaceConflict
        ))
    ));
    let mut conflicting = interfaces;
    let interface = &mut conflicting[1];
    std::mem::swap(&mut interface.primary, &mut interface.secondary);
    assert!(matches!(
        CanonicalContactOperator::prepare(
            &f.projection,
            &conflicting,
            &f.rows,
            3 * f.rows.len(),
            &controls,
            Fixture::budget()
        ),
        Err(ContactOperatorError::Selection(
            ContactSurfaceError::SurfaceConflict
        ))
    ));
}

#[test]
fn face_inventory_bounds_and_precancelled_preparation_reject_before_allocation() {
    let f = Fixture::new(ElementOrder::Tet10, true);
    let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
        CanonicalContactSurfaces::derive_bounded(&f.projection, &f.interfaces[0], 0, 1)
    });
    assert!(matches!(result, Err(ContactSurfaceError::Budget)));
    assert_eq!((retained, peak), (0, 0));
    let operator = f.operator(true);
    let interface = &operator.interfaces[0];
    let point = &interface.points[0];
    let (binding, retained, peak) =
        crate::topology::measure_test_allocations(|| point.bind(&interface.inventory).unwrap());
    assert_eq!(binding.scope_identity(), point.scope_identity());
    assert_eq!((retained, peak), (0, 0));
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
        CanonicalContactOperator::prepare(
            &f.projection,
            &f.interfaces,
            &f.rows,
            3 * f.rows.len(),
            &[Fixture::control(true)],
            Fixture::budget(),
        )
    });
    assert!(matches!(result, Err(ContactOperatorError::Cancelled)));
    assert_eq!((retained, peak), (0, 0));
}

#[test]
fn serialized_contact_controls_reject_unknown_nested_controls() {
    for field in ["pairing", "stationary"] {
        let mut value = serde_json::to_value(Fixture::control(true)).unwrap();
        value[field]
            .as_object_mut()
            .unwrap()
            .insert("undeclared_control".into(), serde_json::json!(1));
        assert!(serde_json::from_value::<ContactInterfacePolicy>(value).is_err());
    }
    let mut value = serde_json::to_value(Fixture::control(true)).unwrap();
    value["stationary"]["differential"]
        .as_object_mut()
        .unwrap()
        .insert("minimum_metric_ration".into(), serde_json::json!(0.1));
    assert!(serde_json::from_value::<ContactInterfacePolicy>(value).is_err());
    let mut value = serde_json::to_value(Fixture::control(true)).unwrap();
    value["pairing"]["narrow"]
        .as_object_mut()
        .unwrap()
        .insert("unknown_depth".into(), serde_json::json!(4));
    assert!(serde_json::from_value::<ContactInterfacePolicy>(value).is_err());
}

#[test]
fn source_counted_preparation_admission_covers_original_allocations_and_exact_boundaries() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = Fixture::new(order, true);
        let controls = [Fixture::control(true)];
        let policy = Fixture::budget();
        let (demand, retained, count_peak) = crate::topology::measure_test_allocations(|| {
            ContactPreparationDemand::derive(
                &f.projection,
                &f.interfaces,
                &f.rows,
                3 * f.rows.len(),
                &controls,
                policy,
            )
            .unwrap()
        });
        assert_eq!(retained, 0);
        assert!(
            count_peak as u64 <= demand.counting_peak_bytes,
            "count peak {count_peak}, demand {} for {order:?}",
            demand.counting_peak_bytes
        );
        let mut exact = policy;
        exact.maximum_preparation_resident_bytes = demand.construction_peak_bytes;
        exact.maximum_preparation_scalar_operations = demand.scalar_operations_upper_bound;
        let (operator, _, peak) = crate::topology::measure_test_allocations(|| {
            CanonicalContactOperator::prepare(
                &f.projection,
                &f.interfaces,
                &f.rows,
                3 * f.rows.len(),
                &controls,
                exact,
            )
            .unwrap()
        });
        assert!(
            peak as u64 <= demand.construction_peak_bytes,
            "order {order:?}: {peak} vs {}",
            demand.construction_peak_bytes
        );
        assert_eq!(operator.point_count(), demand.reference_points);
        let surfaces = CanonicalContactSurfaces::derive(&f.projection, &f.interfaces[0]).unwrap();
        let support = ContactTangentSupport::derive(
            &surfaces,
            &f.rows,
            3 * f.rows.len(),
            ContactSupportBudget {
                maximum_face_pairs: usize::MAX,
                maximum_coordinates: usize::MAX,
            },
        )
        .unwrap();
        assert_eq!(support.face_pair_count(), demand.face_pairs);
        assert_eq!(support.coordinate_count(), demand.coordinates);
        for resident in [true, false] {
            let mut short = exact;
            if resident {
                short.maximum_preparation_resident_bytes -= 1;
            } else {
                short.maximum_preparation_scalar_operations -= 1;
            }
            assert!(matches!(
                CanonicalContactOperator::prepare(
                    &f.projection,
                    &f.interfaces,
                    &f.rows,
                    3 * f.rows.len(),
                    &controls,
                    short
                ),
                Err(ContactOperatorError::Budget)
            ));
        }
        for resident in [true, false] {
            let mut none = exact;
            if resident {
                none.maximum_preparation_resident_bytes = 0;
            } else {
                none.maximum_preparation_scalar_operations = 0;
            }
            let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
                CanonicalContactOperator::prepare(
                    &f.projection,
                    &f.interfaces,
                    &f.rows,
                    3 * f.rows.len(),
                    &controls,
                    none,
                )
            });
            assert!(matches!(result, Err(ContactOperatorError::Budget)));
            assert_eq!((retained, peak), (0, 0));
        }
    }
}
#[test]
fn preparation_counts_all_original_interfaces_and_rejects_invalid_primary_layout() {
    let f = Fixture::new(ElementOrder::Tet10, false);
    let mut second = f.interfaces[0].clone();
    second.interface_id = "other-law".into();
    let single = ContactPreparationDemand::derive(
        &f.projection,
        &f.interfaces,
        &f.rows,
        3 * f.rows.len(),
        &[Fixture::control(false)],
        Fixture::budget(),
    )
    .unwrap();
    let duplicate = [f.interfaces[0].clone(), second];
    let both = ContactPreparationDemand::derive(
        &f.projection,
        &duplicate,
        &f.rows,
        3 * f.rows.len(),
        &[Fixture::control(false); 2],
        Fixture::budget(),
    )
    .unwrap();
    assert_eq!(both.reference_points, 2 * single.reference_points);
    assert_eq!(both.coordinates, 2 * single.coordinates);
    // A demand is a count claim, not source validation or authorization of duplicate physical laws.
    assert!(CanonicalContactOperator::prepare(
        &f.projection,
        &duplicate,
        &f.rows,
        3 * f.rows.len(),
        &[Fixture::control(false); 2],
        Fixture::budget()
    )
    .is_err());
    let mut projection = f.projection.clone();
    let selected = projection
        .boundary_faces
        .iter_mut()
        .find(|face| super::super::super::surfaces::matches(face, &f.interfaces[0].primary))
        .unwrap();
    selected.node_indices[0] = projection.nodes.len();
    assert!(matches!(
        CanonicalContactOperator::prepare(
            &projection,
            &f.interfaces,
            &f.rows,
            3 * f.rows.len(),
            &[Fixture::control(false)],
            Fixture::budget()
        ),
        Err(ContactOperatorError::Invalid)
    ));
}

#[test]
fn preparation_cancellation_and_missing_explicit_controls_reject() {
    let f = Fixture::new(ElementOrder::Tet10, false);
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
        CanonicalContactOperator::prepare(
            &f.projection,
            &f.interfaces,
            &f.rows,
            3 * f.rows.len(),
            &[Fixture::control(false)],
            Fixture::budget(),
        )
    });
    assert!(matches!(result, Err(ContactOperatorError::Cancelled)));
    assert_eq!((retained, peak), (0, 0));
    for field in [
        "maximum_preparation_resident_bytes",
        "maximum_preparation_scalar_operations",
    ] {
        let mut value = serde_json::to_value(Fixture::budget()).unwrap();
        value.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<ContactOperatorPolicy>(value).is_err());
    }
}
