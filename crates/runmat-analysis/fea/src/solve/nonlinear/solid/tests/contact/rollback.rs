use super::*;
// witness: FEA-NONLINEAR-STATE-001/rollback @
//   small_strain_j2_penalty_coulomb_contact/tet4|tet10/prepared_controller;
//   boundary: Rejected/cancelled coupled trials preserve complete original history.
#[test]
fn rejected_coupled_newton_and_cancellation_preserve_complete_original_histories() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = CoupledFixture::new(order, true);
        let operator = f.operator(true);
        let basis =
            CanonicalNonlinearSolidBasis::prepare(&f.model, &f.surface.mesh, &f.summary).unwrap();
        let mut limited = policy();
        limited.max_iterations = 1;
        limited.max_cutbacks = 0;
        let problem = SolidNewtonProblem::prepare_with_contact(
            &basis,
            &f.summary,
            limited,
            Some(CoupledFixture::coupling(&operator)),
        )
        .unwrap();
        let initial = problem.initial_state().unwrap();
        let original = serde_json::to_vec(&initial).unwrap();
        assert!(matches!(
            problem.advance_next(&initial, 1.),
            Err(SolidNewtonError::NotConverged { .. })
        ));
        assert_eq!(serde_json::to_vec(&initial).unwrap(), original);
        let token = crate::progress::FeaCancellationToken::new();
        token.cancel();
        let guard =
            crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
                handler: None,
                cancellation: token,
                attempt_id: None,
            });
        assert!(matches!(
            problem.advance_next(&initial, 1.),
            Err(SolidNewtonError::Cancelled)
        ));
        drop(guard);
        assert_eq!(serde_json::to_vec(&initial).unwrap(), original);
    }
}
