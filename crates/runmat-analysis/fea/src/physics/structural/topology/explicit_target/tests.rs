use super::*;
use crate::fixtures::{fixture_model, FixtureId};
use std::collections::BTreeSet;

#[test]
fn counted_sorted_index_preserves_original_selected_nodes_and_requested_buffers() {
    for fixture in [
        FixtureId::StructuralBeamCantileverEndMomentReference,
        FixtureId::StructuralShellPlateMomentReference,
    ] {
        let model = fixture_model(fixture);
        let structural = model.structural.as_ref().unwrap();
        for selector in [
            format!("node:{}", structural.nodes[0].node_id),
            structural.elements[0].region_id.clone(),
            "absent".into(),
        ] {
            let mut reference = BTreeSet::new();
            for (ordinal, node) in structural.nodes.iter().enumerate() {
                if selector == format!("node:{}", node.node_id) {
                    reference.insert(ordinal);
                }
            }
            for element in structural
                .elements
                .iter()
                .filter(|e| e.region_id == selector)
            {
                let ids = match &element.kind {
                    StructuralElementKind::Beam(beam) => beam.node_ids.as_slice(),
                    StructuralElementKind::Shell(shell) => shell.node_ids.as_slice(),
                };
                for id in ids {
                    if let Some(index) =
                        structural.nodes.iter().position(|node| node.node_id == *id)
                    {
                        reference.insert(index);
                    }
                }
            }
            let profile = explicit_target_workspace_profile(structural, &selector).unwrap();
            let actual = explicit_target_node_indices(structural, &selector).unwrap();
            assert_eq!(actual, reference.into_iter().collect::<Vec<_>>());
            assert!(profile.candidate_slots >= actual.capacity());
            assert_eq!(
                profile.node_index_bytes,
                (structural.nodes.len() * std::mem::size_of::<(u32, usize)>()) as u64
            );
            assert_eq!(
                profile.total_requested_bytes,
                profile.node_index_bytes
                    + (profile.candidate_slots * std::mem::size_of::<usize>()) as u64
            );
        }
    }
}

#[test]
fn duplicate_authored_index_and_cancelled_selection_reject_before_output() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let mut structural = model.structural.unwrap();
    structural.nodes.push(structural.nodes[0].clone());
    assert_eq!(
        explicit_target_node_indices(&structural, "absent").unwrap_err(),
        TopologyStorageError::Index
    );
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        explicit_target_workspace_profile(&structural, "absent").unwrap_err(),
        TopologyStorageError::Cancelled
    );
    assert_eq!(
        explicit_target_node_indices(&structural, "absent").unwrap_err(),
        TopologyStorageError::Cancelled
    );
}
