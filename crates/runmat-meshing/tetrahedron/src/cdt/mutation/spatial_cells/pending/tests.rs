use super::*;
fn key(index: u64) -> CellKey {
    let mut bytes = [0; 32];
    bytes[..8].copy_from_slice(&index.to_be_bytes());
    [StableDigest::from_bytes(bytes); 4]
}
fn change(value: f64) -> Change {
    (
        None,
        Some(Aabb3 {
            min_m: [value; 3],
            max_m: [value + 1.; 3],
        }),
    )
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Checkpoints over 32 and 65,536 deferred spatial updates retain no copied baseline
//   records; changes retain only per-key undo. Nested rollback and committed child changes restore
//   the entire original deferred inventory. This bounds journal entries, not allocator bytes,
//   global spatial-tree rebuilding or native study memory.
#[test]
fn checkpoints_retain_only_changed_keys_across_large_pending_inventory() {
    for count in [32, 65536] {
        let mut pending = PendingChanges::default();
        for i in 0..count {
            pending.set(key(i), Some(change(i as f64)));
        }
        let outer = pending.begin();
        assert_eq!(pending.undo_entries(), 0);
        pending.set(key(0), None);
        assert_eq!(pending.undo_entries(), 1);
        let inner = pending.begin();
        // Temporarily restore the ancestor baseline, then roll back the child.
        pending.set(key(0), Some(change(0.)));
        assert_eq!(pending.undo_entries(), 1);
        pending.rollback(inner);
        assert!(!pending.contains_key(&key(0)));
        assert_eq!(pending.undo_entries(), 1);
        let child = pending.begin();
        pending.set(key(1), None);
        pending.commit(child);
        pending.rollback(outer);
        assert_eq!(pending.len(), count as usize);
        for i in 0..count {
            assert!(change_equal(
                pending.get(&key(i)).copied(),
                Some(change(i as f64))
            ));
        }
        assert_eq!(pending.undo_entries(), 0);
    }
}
#[test]
fn draining_and_rollback_preserve_exact_signed_zero_bounds() {
    let mut pending = PendingChanges::default();
    pending.set(key(1), Some(change(-0.)));
    let parent = pending.begin();
    pending.remove_first();
    pending.set(key(1), Some(change(0.)));
    assert_eq!(pending.undo_entries(), 1);
    pending.rollback(parent);
    assert_eq!(
        pending[&key(1)].1.unwrap().min_m[0].to_bits(),
        (-0.0_f64).to_bits()
    );
}
