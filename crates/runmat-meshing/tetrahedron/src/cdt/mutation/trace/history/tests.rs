use super::*;
use crate::cdt::DelaunayVolumeNode;
use runmat_meshing_core::StableDigest;
use std::collections::{BTreeMap, BTreeSet};
fn edit(index: usize) -> TopologyEdit {
    let mut identity = [111; 32];
    identity[..8].copy_from_slice(&(index as u64).to_be_bytes());
    TopologyEdit::AppendNode(DelaunayVolumeNode::explicit(
        StableDigest::from_bytes(identity),
        [index as f64, 0., 0.],
    ))
}
// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Thirty-two retained branches share exact chronological prefixes at 31 and 65,535
//   edits, copy bounded tail handles, retain one payload per original or new edit and release
//   rejected branches. This verifies history ownership, not native allocator peaks, whole-study
//   memory or physical meshing acceptance.
#[test]
fn retained_frontiers_share_long_prefixes_and_release_rejected_branches() {
    for length in [31, 65_535] {
        let expected = (0..length).map(edit).collect::<Vec<_>>();
        let mut history = TopologyEdits::default();
        for entry in &expected {
            history.push(entry.clone());
        }
        let saved = history.clone();
        assert!(saved.iter().eq(expected.iter()));
        assert!(saved
            .blocks
            .iter()
            .zip(&history.blocks)
            .all(|(a, b)| Arc::ptr_eq(a, b)));
        let mut branches = Vec::new();
        for branch in 0..32 {
            let mut next = history.clone();
            next.push(edit(length + branch));
            assert!(next.starts_with(&saved));
            assert!(!saved.starts_with(&next));
            assert!(next.blocks[..saved.blocks.len() - 1]
                .iter()
                .zip(&saved.blocks)
                .all(|(a, b)| Arc::ptr_eq(a, b)));
            branches.push(next);
        }
        let mut allocations = BTreeMap::new();
        for captured in std::iter::once(&saved).chain(branches.iter()) {
            for block in &captured.blocks {
                allocations.insert(Arc::as_ptr(block) as usize, block.len());
            }
        }
        assert!(allocations.values().sum::<usize>() <= length + 32 * BLOCK_EDITS);
        let payloads = std::iter::once(&saved)
            .chain(branches.iter())
            .flat_map(|history| history.blocks.iter())
            .flat_map(|block| block.iter())
            .map(|edit| Arc::as_ptr(edit) as usize)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            payloads.len(),
            length + 32,
            "tails copy handles, never cavity payloads"
        );

        assert!(saved.iter().eq(expected.iter()));
        history.clear();
        drop(branches);
        assert!(saved
            .blocks
            .iter()
            .all(|block| Arc::strong_count(block) == 1));
        assert!(saved.iter().eq(expected.iter()));
    }
}
#[test]
fn truncate_append_preserves_exact_chronology_and_prefix_rejection_across_blocks() {
    for length in [0, 1, 127, 128, 129, 257] {
        let original = (0..300).map(edit).collect::<Vec<_>>();
        let mut history = TopologyEdits::default();
        for entry in &original {
            history.push(entry.clone());
        }
        let saved = history.clone();
        history.truncate(length);
        assert_eq!(history.len(), length);
        assert!(history.iter().eq(original[..length].iter()));
        assert!(saved.starts_with(&history));
        let mut expected = original[..length].to_vec();
        for index in 0..3 {
            let next = edit(1000 + index);
            expected.push(next.clone());
            history.push(next);
        }
        assert!(history.iter().eq(expected.iter()));
        assert!(!saved.starts_with(&history));
        assert!(saved.iter().eq(original.iter()));
        history.truncate(usize::MAX);
        assert_eq!(history.len(), length + 3);
        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.iter().count(), 0);
        assert!(saved.starts_with(&history));
    }
}
