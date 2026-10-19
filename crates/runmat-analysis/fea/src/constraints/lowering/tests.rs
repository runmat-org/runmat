use super::*;
use crate::{
    constraints::{CanonicalStructuralConstraintPlan, CanonicalStructuralDof},
    fixtures::{fixture_model, FixtureId},
};
use runmat_meshing_core::ElementOrder;

#[test]
fn expansion_inventory_bounds_original_solid_and_shell_row_allocations() {
    let mesh = crate::constraints::compilation_tests::mesh(ElementOrder::Tet10);
    for fixture in [
        FixtureId::CantileverLinearStatic,
        FixtureId::StructuralShellPlateMomentReference,
    ] {
        let model = fixture_model(fixture);
        let plan = CanonicalStructuralConstraintPlan::compile_for_model(
            &model,
            if model.structural.is_some() {
                None
            } else {
                Some(&mesh)
            },
        )
        .unwrap();
        let id = &plan.groups[0].constraint_id;
        let profile = constraint_row_expansion_profile(plan.rows.len(), 1, id).unwrap();
        let retained = plan.rows.len() * std::mem::size_of::<CanonicalStructuralConstraintRow>()
            + plan
                .rows
                .iter()
                .map(|r| {
                    r.terms.capacity() * std::mem::size_of::<CanonicalStructuralConstraintTerm>()
                        + r.constraint_id.capacity()
                })
                .sum::<usize>();
        assert!(profile.total_requested_bytes >= retained as u64);
        let expected = if model.structural.is_some() {
            6
        } else {
            plan.groups[0].targets[0].matched_node_identities.len() * 3
        };
        assert_eq!(plan.rows.len(), expected);
        for row in &plan.rows {
            assert_eq!(row.terms.len(), 1);
            assert_eq!(row.terms[0].coefficient, 1.0);
            assert_eq!(row.right_hand_side, 0.0);
        }
        if model.structural.is_some() {
            assert_eq!(
                plan.rows
                    .iter()
                    .filter(|row| matches!(row.terms[0].dof, CanonicalStructuralDof::Rotation(_)))
                    .count(),
                3
            );
        }
        assert_eq!(
            CanonicalStructuralConstraintPlan::from_canonical_bytes(
                &plan.canonical_bytes().unwrap()
            )
            .unwrap(),
            plan
        );
    }
}

#[test]
fn expansion_storage_rejects_overflow_and_cancellation_before_buffers() {
    assert!(constraint_row_expansion_profile(usize::MAX, 3, "id").is_err());
    assert!(constraint_row_expansion_profile(usize::MAX / 2, 1, "long-id").is_err());
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        constraint_row_expansion_profile(1, 1, "id").unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        rows(1, 1, "id").unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(owned_id("id").unwrap_err(), ConstraintPlanError::Cancelled);
    assert_eq!(
        super::super::row::new_direction(
            "id",
            runmat_meshing_core::StableDigest::ZERO,
            [1.0, 0.0, 0.0],
            0.0
        )
        .unwrap_err(),
        ConstraintPlanError::Cancelled
    );
}

#[test]
fn construction_inventory_bounds_original_owner_dof_and_identity_materialization_shapes() {
    use super::super::{row::constraint_row_workspace_profile, CanonicalStructuralDofOwner};
    use runmat_analysis_core::StructuralRotationDof;
    use runmat_execution::Digest;
    use runmat_meshing_core::StableDigest;
    for count in [1, 3, 17, 1024] {
        let profile =
            constraint_row_construction_profile(1, count, count, "original/Δ/row").unwrap();
        for owner in [
            CanonicalStructuralDofOwner::StructuralNode(StableDigest::from_bytes([255; 32])),
            CanonicalStructuralDofOwner::ReferencePoint(Digest::from_bytes([255; 32])),
        ] {
            for dof in [
                CanonicalStructuralDof::TRANSLATIONS[0],
                CanonicalStructuralDof::Rotation(StructuralRotationDof::Rz),
            ] {
                let terms: Vec<_> = (0..count)
                    .map(|ordinal| CanonicalStructuralConstraintTerm {
                        owner,
                        dof,
                        coefficient: if ordinal % 2 == 0 {
                            f64::MIN_POSITIVE
                        } else {
                            -f64::MAX
                        },
                    })
                    .collect();
                let actual =
                    constraint_row_workspace_profile("original/Δ/row", &terms, -f64::MAX).unwrap();
                assert!(
                    profile.maximum_row_workspace.total_requested_bytes
                        >= actual.total_requested_bytes
                );
                assert!(
                    profile.maximum_row_workspace.identity_materialization_bytes
                        >= actual.identity_materialization_bytes
                );
                assert_eq!(profile.retained.term_slots_upper_bound, count);
                assert_eq!(
                    profile.total_requested_bytes,
                    profile.retained.total_requested_bytes
                        + profile.maximum_row_workspace.total_requested_bytes
                );
            }
        }
    }
}

#[test]
fn variable_construction_inventory_counts_mixed_width_rows_and_empty_work() {
    let profile = constraint_row_construction_profile(3, 1 + 4 + 7, 7, "varied").unwrap();
    assert_eq!(profile.retained.row_slots, 3);
    assert_eq!(profile.retained.term_slots_upper_bound, 12);
    assert_eq!(
        profile.maximum_row_workspace.compaction_bytes,
        (7 * std::mem::size_of::<CanonicalStructuralConstraintTerm>()) as u64
    );
    let empty = constraint_row_construction_profile(0, 0, 0, "empty").unwrap();
    assert_eq!(empty.total_requested_bytes, 0);
    for (rows, terms, width) in [
        (0, 1, 1),
        (1, 1, 2),
        (2, 7, 3),
        (usize::MAX, usize::MAX, usize::MAX),
    ] {
        assert!(constraint_row_construction_profile(rows, terms, width, "invalid").is_err());
    }
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        constraint_row_construction_profile(1, 1, 1, "cancelled").unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        variable_rows(1, 1, 1, "cancelled").unwrap_err(),
        ConstraintPlanError::Cancelled
    );
}
