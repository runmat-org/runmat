//! Undo storage follows live differences, including nested committed branches.
use super::*;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Two hundred fifty-six temporary quality entries leave only one live parent
//   difference. Undo and comparison originals share the same immutable evidence allocation; nested
//   commit and rollback restore the exact complete quality report. This private journal fixture
//   does not validate physical candidate geometry, whole-task memory or throughput.
#[test]
fn nested_quality_undo_shares_original_evidence_and_discards_transient_history() {
    let base = topology();
    let provenance = metric_contexts(&base);
    let mut request = metric_request();
    request.contributions.clear();
    let options = DelaunayVolumeQualityOptions::default();
    let reference =
        evaluate_delaunay_volume_quality(&base, &request, &provenance, options, &NeverCancelled)
            .unwrap();
    let mut owner =
        WorkingVolumeQuality::new(reference.clone(), &base, options, &NeverCancelled).unwrap();
    let key = *owner.cells.first_key_value().unwrap().0;
    let original = owner.cells[&key].clone();
    {
        let mut parent = owner.trial();
        parent.replace(key, None);
        assert!(Arc::ptr_eq(
            parent.journals[0][&key].as_ref().unwrap(),
            parent.baseline_originals[&key].as_ref().unwrap(),
        ));
        for index in 1_u64..=256 {
            let mut evidence = original.clone();
            let mut bytes = [254; 32];
            bytes[..8].copy_from_slice(&index.to_be_bytes());
            evidence.quality.node_identities[0] = StableDigest::from_bytes(bytes);
            let temporary = cell_key(evidence.quality.node_identities);
            {
                let mut child = parent.trial();
                child.replace(temporary, Some(evidence));
                // Committing a child leaves its net change reversible by parent.
                child.commit();
            }
            parent.replace(temporary, None);
            assert_eq!(parent.journals[0].len(), 1);
            assert_eq!(parent.baseline_originals.len(), 1);
        }
        {
            let mut child = parent.trial();
            child.replace(key, Some(original.clone()));
            assert!(child.journals[0].is_empty());
            // Child rollback restores parent's deleted state and its undo.
        }
        assert!(!parent.cells.contains_key(&key));
        assert_eq!(parent.journals[0].len(), 1);
    }
    assert!(owner.journals.is_empty());
    assert!(owner.baseline_originals.is_empty());
    assert!(owner.cells[&key] == original);
    drop(original);
    assert_eq!(
        owner
            .into_canonical(&base, options, &NeverCancelled)
            .unwrap(),
        reference
    );
}
