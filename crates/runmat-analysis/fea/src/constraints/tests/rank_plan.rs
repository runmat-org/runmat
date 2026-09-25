use super::*;

#[test]
fn scaled_equation_conflicts_are_reported_at_the_canonical_plan_boundary() {
    let (mut model, mesh) = model_and_mesh();
    let first = equation_constraint("first", 1, 2, 0.01);
    let mut scaled = equation_constraint("scaled", 1, 2, 0.01);
    let AnalysisConstraint::Structural(StructuralConstraint {
        kind: StructuralConstraintKind::Equation(equation),
        ..
    }) = &mut scaled
    else {
        unreachable!()
    };
    for term in &mut equation.terms {
        term.coefficient *= 1.0e-12;
    }
    equation.right_hand_side = Displacement::new(0.01e-12);
    model.constraints = vec![first.clone(), scaled.clone()];
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    assert_eq!(plan.evidence.active_row_count, 1);
    assert_eq!(plan.evidence.redundant_row_count, 1);

    let AnalysisConstraint::Structural(StructuralConstraint {
        kind: StructuralConstraintKind::Equation(equation),
        ..
    }) = &mut scaled
    else {
        unreachable!()
    };
    equation.right_hand_side = Displacement::new(0.02e-12);
    model.constraints = vec![first, scaled];
    assert_eq!(
        CanonicalStructuralConstraintPlan::compile(&model, &mesh),
        Err(ConstraintPlanError::ConflictingRow {
            first_constraint_id: "first".into(),
            second_constraint_id: "scaled".into(),
            conflicting_dof: None,
        })
    );
}

#[test]
fn large_prescribed_values_do_not_mask_plan_conflicts() {
    let (mut model, mesh) = model_and_mesh();
    model.constraints = vec![
        equation_constraint("first", 1, 2, 1.0e12),
        equation_constraint("second", 1, 2, 1.0e12 + 1.0),
    ];
    assert_eq!(
        CanonicalStructuralConstraintPlan::compile(&model, &mesh),
        Err(ConstraintPlanError::ConflictingRow {
            first_constraint_id: "first".into(),
            second_constraint_id: "second".into(),
            conflicting_dof: None,
        })
    );
}
