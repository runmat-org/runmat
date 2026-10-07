use super::*;
use crate::fixtures::{fixture_model, mixed_solid_beam_model, FixtureId};
use runmat_meshing_core::ElementOrder;
#[test]
fn finalization_profile_bounds_original_canonical_payload_and_component_rank() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = crate::constraints::compilation_tests::mesh(order);
        for model in [
            fixture_model(FixtureId::CantileverLinearStatic),
            mixed_solid_beam_model(mesh.topology.nodes[0].coordinates_m),
        ] {
            let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
            let profile = constraint_plan_finalization_workspace_profile(&plan).unwrap();
            let bytes = plan.canonical_bytes().unwrap();
            let actual =
                runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(
                    &bytes,
                )
                .unwrap();
            assert!(profile.canonical_materialization_bytes >= actual);
            assert_eq!(
                profile.rank_bytes,
                rank::constraint_rank_workspace_profile(&plan.rows)
                    .unwrap()
                    .total_requested_bytes
            );
            assert!(
                profile.total_requested_bytes
                    >= profile.canonical_materialization_bytes + profile.rank_bytes
            );
            assert_eq!(
                CanonicalStructuralConstraintPlan::from_canonical_bytes(&bytes).unwrap(),
                plan
            );
        }
    }
}

#[test]
fn cold_validation_rejects_evidence_overflow_and_undeclared_row_groups() {
    let mesh = crate::constraints::compilation_tests::mesh(ElementOrder::Tet4);
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let original = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let mut overflow = original.clone();
    overflow.evidence.active_row_count = usize::MAX;
    overflow.evidence.redundant_row_count = 1;
    let mut orphan = original.clone();
    let mut row = orphan.rows[0].clone();
    row.constraint_id = "zz-undeclared".into();
    row.identity = crate::constraints::row::row_identity(&row).unwrap();
    orphan.rows.push(row);
    let (active, redundant) = crate::constraints::row::classify(&mut orphan.rows).unwrap();
    orphan.evidence.active_row_count = active;
    orphan.evidence.redundant_row_count = redundant;
    orphan.evidence.rank = active;
    for malformed in [overflow, orphan] {
        // Direct codec encoding deliberately bypasses producer-side plan.validate.
        let bytes = encode_contract(PLAN_PREFIX, PLAN_DOMAIN, &malformed, PLAN_LIMITS).unwrap();
        assert!(matches!(
            CanonicalStructuralConstraintPlan::from_canonical_bytes(&bytes),
            Err(ConstraintPlanError::Invalid(_))
        ));
    }
}
