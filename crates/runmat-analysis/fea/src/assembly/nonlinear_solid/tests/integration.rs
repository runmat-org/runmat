use super::*;

#[test]
fn explicit_integration_preserves_affine_physics_and_rejects_foreign_history() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, initial) = fixture(order, false);
        let original = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
        let original_state = original.zero_state();
        let strain = [0.01, -0.002, 0.003, 0.004, -0.006, 0.008];
        let stress = [14.88, 3.36, 8.16, 1.92, -2.88, 3.84];
        let displacement = affine_displacement(&mesh, &initial, strain);
        let expected = constant_stress_force(&mesh, &initial, stress);
        for rule in [
            CanonicalSolidIntegrationRule::Keast15,
            CanonicalSolidIntegrationRule::Duffy5,
            CanonicalSolidIntegrationRule::Duffy8,
            CanonicalSolidIntegrationRule::Duffy12,
            CanonicalSolidIntegrationRule::Duffy16,
        ] {
            let plan = CanonicalSolidIntegrationPlan::derive(&mesh, rule).unwrap();
            let basis = CanonicalNonlinearSolidBasis::prepare_with_integration(
                &model, &mesh, &initial, &plan,
            )
            .unwrap();
            assert_eq!(basis.point_count(), plan.total_point_count());
            let state = basis.zero_state();
            assert_eq!(state.integration_identity, plan.identity());
            if rule == CanonicalSolidIntegrationRule::Keast15 {
                assert_eq!(basis.identity(), original.identity());
                assert_eq!(state, original_state);
            } else {
                assert_ne!(basis.identity(), original.identity());
                assert!(basis.validate_state(&original_state).is_err());
                assert!(original.validate_state(&state).is_err());
            }
            let evaluation = basis.evaluate(&displacement, &state).unwrap();
            assert_eq!(evaluation.trial_state, state);
            for (actual, expected) in evaluation.internal_force.iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-9, "{order:?}/{rule:?}");
            }
            for point in &evaluation.point_evidence {
                for (actual, expected) in point.stress_pa.iter().zip(stress) {
                    assert!((actual - expected).abs() < 1e-9);
                }
            }
            let mut foreign = state.clone();
            foreign.integration_identity = Digest::sha256(b"foreign-rule");
            assert!(basis.validate_state(&foreign).is_err());
        }
    }
}

#[test]
fn cancelled_original_basis_preparation_preserves_typed_cancellation() {
    let (model, mesh, summary) = fixture(ElementOrder::Tet4, true);
    let context = crate::progress::FeaProgressContext::default();
    context.cancellation.cancel();
    let guard = crate::progress::replace_fea_progress_context(context);
    assert!(matches!(
        CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary),
        Err(NonlinearSolidError::Cancelled)
    ));
    assert!(matches!(
        NonlinearSolidError::from(crate::topology::AnalysisProjectionError::Cancelled),
        NonlinearSolidError::Cancelled
    ));
    drop(guard);
    assert!(CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary).is_ok());
}
