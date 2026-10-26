use super::*;
// witness: FEA-NONLINEAR-STATE-001/cold_validation @
//   small_strain_j2_penalty_coulomb_contact/tet10/prepared_controller;
//   boundary: Combined scope/material/contact history and GENERAL admission substitution reject.
#[test]
fn combined_scope_history_and_general_admission_reject_substitution() {
    let f = CoupledFixture::new(ElementOrder::Tet10, true);
    let operator = f.operator(true);
    let basis =
        CanonicalNonlinearSolidBasis::prepare(&f.model, &f.surface.mesh, &f.summary).unwrap();
    let problem = SolidNewtonProblem::prepare_with_contact(
        &basis,
        &f.summary,
        policy(),
        Some(CoupledFixture::coupling(&operator)),
    )
    .unwrap();
    let initial = problem.initial_state().unwrap();
    let state = problem.advance_next(&initial, 1.).unwrap().state;
    let mut missing = state.clone();
    missing.contact = None;
    assert!(problem.validate_accepted(&missing).is_err());
    let plain = SolidNewtonProblem::prepare(&basis, &f.summary, policy()).unwrap();
    assert!(plain.validate_accepted(&state).is_err());
    let mut mismatched = state.clone();
    mismatched.displacement[problem.reduction.free_dof_indices[0]] += 1e-4;
    assert!(problem.validate_accepted(&mismatched).is_err());
    let mut encoded = serde_json::to_value(&state).unwrap();
    encoded["contact"]["points"][0]["point"]["frictional_work_j_per_m2"] = (-1.).into();
    let corrupt: AcceptedSolidState = serde_json::from_value(encoded).unwrap();
    assert!(problem.validate_accepted(&corrupt).is_err());
    let mut coupling = CoupledFixture::coupling(&operator);
    coupling.solve_admission.resident_bytes = 0;
    assert!(matches!(
        SolidNewtonProblem::prepare_with_contact(&basis, &f.summary, policy(), Some(coupling)),
        Err(SolidNewtonError::General(
            crate::solve::general::GeneralSolveError::Admission
        ))
    ));
    for budget in 0..3 {
        let mut coupling = CoupledFixture::coupling(&operator);
        if budget == 0 {
            coupling.maximum_geometry_trials_per_advance = 0;
        } else if budget == 1 {
            coupling.maximum_symbolic_scalar_operations = 0;
        } else {
            coupling.maximum_additional_resident_bytes = 0;
        }
        assert!(matches!(
            SolidNewtonProblem::prepare_with_contact(&basis, &f.summary, policy(), Some(coupling)),
            Err(SolidNewtonError::Contact(
                crate::physics::structural::contact::ContactOperatorError::Budget
            ))
        ));
    }
    assert_eq!(initial.contact_configuration().unwrap().ordinal(), 0);
}

#[test]
fn zero_load_open_contact_does_not_accept_unanchored_body_tangent() {
    let mut f = CoupledFixture::new(ElementOrder::Tet4, false);
    let n = f.summary.dof_count;
    f.summary.affine_constraint_reduction = Some(AffineConstraintReduction {
        full_dof_count: n,
        reduced_dof_count: n,
        free_dof_indices: (0..n).collect(),
        pivot_dof_indices: Vec::new(),
        recovery_row_offsets: (0..=n).collect(),
        recovery_column_indices: (0..n).collect(),
        recovery_coefficients: vec![1.; n],
        offset: vec![0.; n],
    });
    let operator = f.operator(false);
    let basis =
        CanonicalNonlinearSolidBasis::prepare(&f.model, &f.surface.mesh, &f.summary).unwrap();
    let mut coupling = CoupledFixture::coupling(&operator);
    let support = basis.contact_pattern().union(operator.pattern()).unwrap();
    // This separate nullspace reference has many more free coordinates than
    // the affine patch. Admit its actual full source inventory explicitly.
    coupling.solve_admission = GeneralSolveEnvelope::for_counts(
        support.rows(),
        support.column_indices().len(),
        coupling.solve_policy,
    )
    .unwrap();
    let problem =
        SolidNewtonProblem::prepare_with_contact(&basis, &f.summary, policy(), Some(coupling))
            .unwrap();
    assert!(matches!(
        problem.initial_state(),
        Err(SolidNewtonError::General(
            crate::solve::general::GeneralSolveError::NonsingularityNotEstablished(_)
        ))
    ));
}
