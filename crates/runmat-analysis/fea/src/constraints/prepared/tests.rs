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

#[test]
fn preparation_inventory_composes_original_occurrences_and_explicit_identity_copies() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = crate::constraints::compilation_tests::mesh(order);
        let constraint = equation();
        let profile = profile::derive(&constraint, Some(&mesh), None).unwrap();
        assert_eq!(profile.target_occurrences, 2);
        assert_eq!(
            profile.occurrence_bytes,
            (2 * std::mem::size_of::<(&RegionTarget, PreparedTarget)>()) as u64
        );
        assert_eq!(
            profile.total_requested_bytes,
            profile.occurrence_bytes + profile.selection_bytes
        );
        let prepared = PreparedTargets::derive(&constraint, Some(&mesh), None).unwrap();
        assert_eq!(prepared.selections.len(), profile.target_occurrences);
    }
    let model = crate::fixtures::fixture_model(
        crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
    );
    let structural = model.structural.as_ref().unwrap();
    let constraint = model
        .constraints
        .iter()
        .find_map(|c| c.structural())
        .unwrap();
    let topology = CanonicalStructuralNodeSpace::derive_explicit(structural).unwrap();
    let profile = profile::derive(constraint, None, Some(structural)).unwrap();
    let prepared =
        PreparedTargets::derive(constraint, None, Some((structural, &topology))).unwrap();
    assert_eq!(prepared.selections.len(), profile.target_occurrences);
    let retained: usize = prepared
        .selections
        .as_slice()
        .iter()
        .map(|(_, selection)| {
            selection.nodes.capacity() * std::mem::size_of::<usize>()
                + selection.resolved.matched_node_identities.capacity()
                    * std::mem::size_of::<runmat_meshing_core::StableDigest>()
        })
        .sum();
    assert!(profile.selection_bytes >= retained as u64);
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        profile::derive(constraint, None, Some(structural)).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
}

#[test]
fn mixed_selection_inventory_bounds_original_ownership_without_materializing_for_counting() {
    use crate::physics::structural::topology::target::{
        mixed_target_workspace_profile, resolve_mixed_structural_target,
        MixedStructuralTargetError, MixedStructuralTargetOwnership,
    };
    let mesh = crate::constraints::compilation_tests::mesh(ElementOrder::Tet4);
    let model = crate::fixtures::mixed_solid_beam_model(
        mesh.topology.nodes[0].coordinates_m.canonical_value(),
    );
    let structural = model.structural.as_ref().unwrap();
    let topology = CanonicalStructuralNodeSpace::derive(&mesh, structural).unwrap();
    for target in [
        RegionTarget::point("node:1"),
        RegionTarget::point("node:102"),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
    ] {
        let profile = mixed_target_workspace_profile(&target, &mesh, structural).unwrap();
        let selected =
            resolve_mixed_structural_target(&target, &mesh, structural, &topology).unwrap();
        if target.region_id == "node:1" {
            assert!(matches!(
                &selected,
                MixedStructuralTargetOwnership::Solid { .. }
            ));
            assert_eq!(profile.solid_candidate_slots, 1);
        } else if target.region_id == "node:102" {
            assert!(matches!(
                &selected,
                MixedStructuralTargetOwnership::Explicit { .. }
            ));
            assert_eq!(profile.solid_candidate_slots, 1);
        }
        assert!(
            profile.solid_candidate_slots + profile.explicit_candidate_slots
                >= selected.canonical_nodes().len()
        );
        assert!(
            profile.total_requested_bytes
                >= std::mem::size_of_val(selected.canonical_nodes()) as u64
        );
    }
    let mut collision = structural.clone();
    collision.nodes[1].node_id = mesh.topology.nodes[0].node_id as u32;
    let runmat_analysis_core::StructuralElementKind::Beam(beam) = &mut collision.elements[0].kind
    else {
        unreachable!()
    };
    beam.node_ids[1] = collision.nodes[1].node_id;
    let collision_topology = CanonicalStructuralNodeSpace::derive(&mesh, &collision).unwrap();
    assert!(matches!(
        resolve_mixed_structural_target(
            &RegionTarget::point("node:1"),
            &mesh,
            &collision,
            &collision_topology,
        ),
        Err(MixedStructuralTargetError::Ambiguous { .. })
    ));
}
