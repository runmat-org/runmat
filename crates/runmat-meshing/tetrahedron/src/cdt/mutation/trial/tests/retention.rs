//! Rejected nested searches must not accumulate undo for restored cavities.
use super::*;

fn undo_entries(journal: &TrialJournal) -> usize {
    journal.cells.len()
        + journal.nodes.len()
        + journal.cell_keys.len()
        + journal.stars.len()
        + journal.faces.len()
        + journal.boundary.len()
        + journal.free.len()
        + journal.dirty.len()
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Rejected nested flips across four and 64 distinct disconnected cavities leave zero
//   parent undo records; a committed child remains reversible, with exact original topology,
//   incidence and search authority restored. This bounds retained undo differences on these
//   fixtures, not arbitrary connected propagation, allocator footprint or native study acceptance.
#[test]
fn parent_undo_drops_rejected_child_history_across_distinct_cavities() {
    for count in [4, 64] {
        let mut owner = neighborhoods(count);
        owner.physical_dirty_faces = Some(BTreeSet::new());
        let original = owner.clone();
        {
            let mut parent = owner.cavity_trial();
            for cluster in 0..count {
                let first = (cluster * 5) as u32;
                let removed = parent.vertex_stars[first as usize].clone();
                {
                    let mut child = parent.cavity_trial();
                    child
                        .replace_cavity(
                            &removed,
                            vec![
                                ([first, first + 1, first + 3, first + 4], None),
                                ([first + 1, first + 2, first + 3, first + 4], None),
                                ([first + 2, first, first + 3, first + 4], None),
                            ],
                        )
                        .unwrap();
                    assert!(undo_entries(&child.trial_journals[0]) > 0);
                }
                assert_eq!(
                    undo_entries(&parent.trial_journals[0]),
                    0,
                    "restored child retained history at cavity {cluster}"
                );
            }
            // An accepted child still remains reversible by its parent.
            {
                let mut child = parent.cavity_trial();
                flip(&mut child);
                child.commit();
            }
            assert!(undo_entries(&parent.trial_journals[0]) > 0);
        }
        assert_storage(&owner, &original);
    }
}
