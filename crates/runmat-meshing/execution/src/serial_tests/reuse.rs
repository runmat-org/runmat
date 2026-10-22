use super::*;
use crate::AcceptedMeshingStageReceipt;

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An accepted surface-stage receipt rebinds session access, preserves original
//   observations and rejects changed implementation, changed input, understated limits and corrupt
//   storage. Reuse is not fresh numerical execution or a native scaling measurement.
#[test]
fn accepted_stage_reuse_rebinds_authority_and_preserves_original_observations() {
    let mut fixture = Fixture::new();
    let completed = execute_serial_stage(
        &fixture.program,
        &mut fixture.store,
        &SurfaceKernel,
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1024),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let success = completed.attempt_success();
    let implementation = Digest::sha256(b"complete-surface-implementation");
    let mut persistent = MemoryStore::default();
    let receipt = AcceptedMeshingStageReceipt::retain(
        &fixture.store,
        &mut persistent,
        &fixture.host,
        implementation,
        &success,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let mut current = fixture.host.clone();
    current.artifact_access = ArtifactAccess {
        authorization_scope: "new-session".into(),
        encryption_context: Digest::sha256(b"new-context"),
    };
    assert_eq!(
        crate::meshing_stage_reuse_key(&fixture.host, implementation).unwrap(),
        crate::meshing_stage_reuse_key(&current, implementation).unwrap()
    );
    let bytes = serde_json::to_vec(&receipt).unwrap();
    let cold: AcceptedMeshingStageReceipt = serde_json::from_slice(&bytes).unwrap();
    let mut destination = MemoryStore::default();
    let rebound = cold
        .restore(
            &persistent,
            &mut destination,
            &current,
            implementation,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
    assert!(rebound
        .result_objects()
        .iter()
        .all(|reference| reference.authorization_scope == "new-session"
            && reference.encryption_context == current.artifact_access.encryption_context));
    assert_eq!(
        root(&rebound.values().unwrap().0[0]).logical_digest,
        root(&success.values().unwrap().0[0]).logical_digest
    );
    let mut observation = crate::import_stage_evidence_observation(
        &destination,
        &current,
        rebound.result_objects(),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let lineage = observation.reused_from.take().unwrap();
    assert_eq!(
        lineage.original_observation_digest,
        completed.stage_evidence().canonical_digest().unwrap()
    );
    assert_eq!(observation, *completed.stage_evidence());
    assert!(cold
        .restore(
            &persistent,
            &mut destination,
            &current,
            Digest::sha256(b"changed-implementation"),
            ObjectInventoryLimits::default()
        )
        .is_err());
    let mut changed_request = request();
    changed_request.deterministic_seed += 1;
    let changed = Fixture::with_request(changed_request);
    assert!(cold
        .restore(
            &persistent,
            &mut destination,
            &changed.host,
            implementation,
            ObjectInventoryLimits::default()
        )
        .is_err());
    let smaller = ObjectInventoryLimits {
        max_total_bytes: 1,
        ..ObjectInventoryLimits::default()
    };
    assert!(cold
        .restore(
            &persistent,
            &mut destination,
            &current,
            implementation,
            smaller
        )
        .is_err());
    let manifest = root(&success.values().unwrap().0[0]).logical_digest;
    persistent.objects.insert(manifest, b"corrupt".to_vec());
    assert!(cold
        .restore(
            &persistent,
            &mut destination,
            &current,
            implementation,
            ObjectInventoryLimits::default()
        )
        .is_err());
}

#[test]
fn incomplete_stage_receipts_never_enter_the_persistent_cache() {
    let fixture = Fixture::new();
    let mut persistent = MemoryStore::default();
    let incomplete = runmat_execution_runner::AttemptSuccess::Values {
        outputs: Vec::new(),
        result_objects: Vec::new(),
        resources: None,
    };
    assert!(AcceptedMeshingStageReceipt::retain(
        &fixture.store,
        &mut persistent,
        &fixture.host,
        Digest::sha256(b"implementation"),
        &incomplete,
        ObjectInventoryLimits::default()
    )
    .is_err());
    assert!(persistent.objects.is_empty());
}
