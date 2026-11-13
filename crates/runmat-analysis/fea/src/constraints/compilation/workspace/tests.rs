use super::*;
use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, RegionTarget, StructuralConstraintEquation,
    StructuralConstraintEquationTerm, StructuralConstraintKind, StructuralTranslationDof,
};
use runmat_meshing_core::ElementOrder;

fn equation() -> StructuralConstraint {
    StructuralConstraint {
        constraint_id: "equation/Δ".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Equation(StructuralConstraintEquation {
            terms: [1.0, -0.5]
                .into_iter()
                .map(|coefficient| StructuralConstraintEquationTerm {
                    target: RegionTarget::point("node:1"),
                    dof: StructuralTranslationDof::Ux,
                    coefficient,
                })
                .collect(),
            right_hand_side: Displacement::new(0.0),
        }),
    }
}

#[test]
fn source_and_group_workspace_retain_original_equation_borrowing_and_exact_slots() {
    let constraint = equation();
    let source = plan::constraint_source_workspace_profile(&constraint).unwrap();
    assert_eq!(
        source.equation_reference_bytes,
        (2 * std::mem::size_of::<&StructuralConstraintEquationTerm>()) as u64
    );
    assert_eq!(
        source.equation_ordering_bytes,
        (2 * std::mem::size_of::<usize>()) as u64
    );
    for coupling in [false, true] {
        let profile =
            constraint_group_construction_workspace_profile(&constraint, 7, coupling).unwrap();
        assert_eq!(profile.id_copy_bytes, constraint.constraint_id.len() as u64);
        assert_eq!(
            profile.target_ordering_bytes,
            (7 * std::mem::size_of::<usize>()) as u64
        );
        assert_eq!(
            profile.reference_identity_bytes,
            if coupling {
                std::mem::size_of::<runmat_execution::Digest>() as u64
            } else {
                0
            }
        );
        assert_eq!(profile.source_workspace_bytes, source.total_requested_bytes);
    }
}

#[test]
fn incremental_retention_bounds_combined_original_plan_without_copying_or_rescanning_it() {
    let mesh = crate::constraints::compilation_tests::mesh(ElementOrder::Tet10);
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model
        .constraints
        .push(AnalysisConstraint::Structural(equation()));
    let original = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let before = original.canonical_bytes().unwrap();
    let mut workspace = AccumulationWorkspace::new(original.groups.len()).unwrap();
    let mut groups = Vec::new();
    let mut rows = Vec::new();
    for group in &original.groups {
        let incoming: Vec<_> = original
            .rows
            .iter()
            .filter(|row| row.constraint_id == group.constraint_id)
            .cloned()
            .collect();
        let required = workspace
            .advance(
                group,
                &incoming,
                incoming.capacity(),
                rows.len(),
                rows.capacity(),
            )
            .unwrap();
        groups.push(group);
        rows.extend(incoming);
        let full = plan::canonical_value_workspace_bytes(&(&groups, &rows)).unwrap();
        assert!(workspace.retained_bytes >= full);
        assert!(required >= workspace.retained_bytes);
    }
    assert_eq!(original.canonical_bytes().unwrap(), before);
}

#[test]
fn group_and_accumulation_profiles_reject_overflow_bad_capacities_and_cancellation() {
    let constraint = equation();
    assert!(
        constraint_group_construction_workspace_profile(&constraint, usize::MAX, true).is_err()
    );
    assert!(AccumulationWorkspace::new(usize::MAX).is_err());
    let mesh = crate::constraints::compilation_tests::mesh(ElementOrder::Tet4);
    let model = crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let mut workspace = AccumulationWorkspace::new(1).unwrap();
    let original = workspace.retained_bytes;
    assert!(workspace
        .advance(&plan.groups[0], &plan.rows, 0, 1, 0)
        .is_err());
    assert_eq!(workspace.retained_bytes, original);
    workspace.retained_bytes = u64::MAX;
    assert!(workspace
        .advance(&plan.groups[0], &plan.rows, plan.rows.capacity(), 0, 0)
        .is_err());
    assert_eq!(workspace.retained_bytes, u64::MAX);
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        constraint_group_construction_workspace_profile(&constraint, 1, false).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        plan::constraint_source_workspace_profile(&constraint).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert!(matches!(
        AccumulationWorkspace::new(1),
        Err(ConstraintPlanError::Cancelled)
    ));
}
