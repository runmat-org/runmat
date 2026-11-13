//! Compare borrowed source ordering with the original owned/stable canonical projection.
use super::*;
use runmat_analysis_core::{
    CoordinateFrame, Displacement, RegionTarget, StructuralConstraint,
    StructuralConstraintEquation, StructuralConstraintEquationTerm, StructuralTranslationDof,
};
use runmat_canonical_codec::encode_contract;
use runmat_meshing_core::{PersistentEntityId, PersistentEntityKind};
#[test]
fn borrowed_equation_source_preserves_stable_ties_and_owned_reference_digest() {
    let terms: Vec<_> = (0..128)
        .map(|i| {
            let mut target = RegionTarget::point(format!("node:{}", i % 5));
            target
                .bind_resolved_entities(vec![PersistentEntityId::new(
                    PersistentEntityKind::Vertex,
                    format!("vertex-{i}"),
                    vec!["root".repeat(32)],
                )])
                .unwrap();
            StructuralConstraintEquationTerm {
                target,
                dof: StructuralTranslationDof::Ux,
                coefficient: 1.0,
            }
        })
        .collect();
    let constraint = StructuralConstraint {
        constraint_id: "equation".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Equation(StructuralConstraintEquation {
            terms,
            right_hand_side: Displacement::new(0.3),
        }),
    };
    let StructuralConstraintKind::Equation(equation) = &constraint.kind else {
        unreachable!()
    };
    let mut owned = equation.terms.clone();
    owned.sort_by(|a, b| {
        authored_target_key(&a.target)
            .cmp(&authored_target_key(&b.target))
            .then_with(|| a.dof.cmp(&b.dof))
            .then_with(|| a.coefficient.total_cmp(&b.coefficient))
    });
    let expected = canonical_digest(
        SOURCE_DOMAIN,
        &(
            &constraint.constraint_id,
            &constraint.frame,
            "equation",
            &owned,
            equation.right_hand_side,
        ),
    )
    .unwrap();
    let bytes = encode_contract(
        PLAN_PREFIX,
        SOURCE_DOMAIN,
        &(
            &constraint.constraint_id,
            &constraint.frame,
            "equation",
            &owned,
            equation.right_hand_side,
        ),
        PLAN_LIMITS,
    )
    .unwrap();
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    model.constraints = vec![runmat_analysis_core::AnalysisConstraint::Structural(
        constraint.clone(),
    )];
    let profile = constraint_authored_workspace_profile(&model).unwrap();
    assert_eq!(profile.constraint_count, 1);
    assert_eq!(
        profile.equation_reference_bytes,
        (equation.terms.len() * std::mem::size_of::<&StructuralConstraintEquationTerm>()) as u64
    );
    assert!(
        profile.canonical_materialization_bytes
            >= runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(
                &bytes
            )
            .unwrap()
    );
    assert_eq!(source_digest(&constraint).unwrap(), expected);
}
