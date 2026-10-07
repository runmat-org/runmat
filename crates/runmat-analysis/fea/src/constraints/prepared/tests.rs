use super::*;
use runmat_analysis_core::{
    CoordinateFrame, Displacement, StructuralConstraintEquation, StructuralConstraintEquationTerm,
    StructuralTranslationDof,
};
use runmat_meshing_core::ElementOrder;

fn equation() -> StructuralConstraint {
    StructuralConstraint {
        constraint_id: "duplicate-target-occurrences".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Equation(StructuralConstraintEquation {
            terms: [2.0, -1.0]
                .into_iter()
                .map(|coefficient| StructuralConstraintEquationTerm {
                    target: RegionTarget::point("node:1"),
                    dof: StructuralTranslationDof::Ux,
                    coefficient,
                })
                .collect(),
            right_hand_side: Displacement::new(0.125),
        }),
    }
}

#[test]
fn prepared_occurrences_move_original_selection_after_source_mesh_is_dropped() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = crate::constraints::compilation_tests::mesh(order);
        let constraint = equation();
        let mut prepared = PreparedTargets::derive(&constraint, Some(&mesh), None).unwrap();
        assert!(!prepared.requires_mixed());
        assert_eq!(prepared.remaining_node_count().unwrap(), 2);
        let pointers = prepared
            .selections
            .as_slice()
            .iter()
            .map(|(_, selection)| selection.resolved.matched_node_identities.as_ptr())
            .collect::<Vec<_>>();
        let expected_identity = mesh.topology.nodes[0].stable_identity;
        drop(mesh);
        let StructuralConstraintKind::Equation(equation) = &constraint.kind else {
            unreachable!()
        };
        for (ordinal, term) in equation.terms.iter().enumerate() {
            let (resolved, nodes) = prepared.take(&term.target).unwrap();
            assert_eq!(resolved.matched_node_identities, [expected_identity]);
            assert_eq!(resolved.matched_node_identities.as_ptr(), pointers[ordinal]);
            assert!(nodes.is_empty());
            assert_eq!(prepared.remaining_node_count().unwrap(), 1 - ordinal);
        }
        prepared.finish().unwrap();
    }
}

#[test]
fn prepared_sequences_reject_missing_unused_and_reordered_occurrences() {
    let mesh = crate::constraints::compilation_tests::mesh(ElementOrder::Tet4);
    let constraint = equation();
    let StructuralConstraintKind::Equation(equation) = &constraint.kind else {
        unreachable!()
    };
    assert!(PreparedTargets::derive(&constraint, Some(&mesh), None)
        .unwrap()
        .finish()
        .is_err());
    let mut prepared = PreparedTargets::derive(&constraint, Some(&mesh), None).unwrap();
    assert!(prepared.take(&equation.terms[1].target).is_err());
    let mut prepared = PreparedTargets::derive(&constraint, Some(&mesh), None).unwrap();
    for term in &equation.terms {
        prepared.take(&term.target).unwrap();
    }
    assert!(prepared.take(&equation.terms[0].target).is_err());
}

#[test]
fn empty_explicit_equation_rejects_without_dispatching_to_missing_solid_mesh() {
    let mut model = crate::fixtures::fixture_model(
        crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
    );
    model.constraints.clear();
    let mut constraint = equation();
    let StructuralConstraintKind::Equation(equation) = &mut constraint.kind else {
        unreachable!()
    };
    equation.terms.clear();
    model
        .constraints
        .push(runmat_analysis_core::AnalysisConstraint::Structural(
            constraint,
        ));
    assert!(
        crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(&model, None)
            .is_err()
    );
}
