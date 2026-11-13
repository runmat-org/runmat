use runmat_execution::value::ValueRefKind;

use super::*;

#[test]
fn pending_owners_release_independently_and_future_drop_cleans_uncommitted_objects() {
    let catalog = RemoteObjectCatalog::new().unwrap();
    let reference = reference("run", b"pending", b"pending");
    let first = catalog
        .reserve_results(&[reference.clone(), reference.clone()])
        .unwrap();
    let second = catalog
        .reserve_results(std::slice::from_ref(&reference))
        .unwrap();
    store(&catalog.store, &reference, b"pending").unwrap();
    insert(
        &mut catalog.state.lock().unwrap().objects,
        reference.clone(),
    )
    .unwrap();
    assert_eq!(
        catalog.state.lock().unwrap().pending_results[&reference.id].1,
        2
    );
    catalog.discard_results(std::slice::from_ref(&reference));
    drop(first);
    assert!(catalog.contains(&reference).unwrap());
    // Dropping a suspended/abandoned task drops its owned lease without an
    // async cleanup operation or a scheduler callback.
    let abandoned = async move {
        let _lease = second;
        std::future::pending::<()>().await;
    };
    drop(abandoned);
    assert!(catalog.state.lock().unwrap().pending_results.is_empty());
    assert!(!catalog.contains(&reference).unwrap());
}

#[test]
fn pending_identity_and_capacity_are_reserved_atomically_before_download() {
    let catalog = RemoteObjectCatalog::new().unwrap();
    let reference = reference("run", b"reserved", b"reserved");
    let lease = catalog
        .reserve_results(std::slice::from_ref(&reference))
        .unwrap();
    let mut foreign = reference.clone();
    foreign.logical_digest = Digest::sha256(b"foreign");
    let unrelated = self::reference("run", b"unrelated", b"unrelated");
    assert!(catalog
        .reserve_results(&[unrelated.clone(), foreign])
        .is_err());
    assert_eq!(catalog.state.lock().unwrap().pending_results.len(), 1);
    assert_eq!(
        catalog.state.lock().unwrap().pending_results[&reference.id].1,
        1
    );
    drop(lease);
    let mut oversized = reference.clone();
    oversized.encoded_length = runmat_execution_artifact::MAX_PROGRAM_EXECUTION_OBJECT_TOTAL_BYTES;
    let full = catalog.reserve_results(&[oversized]).unwrap();
    assert!(catalog
        .reserve_results(std::slice::from_ref(&unrelated))
        .is_err());
    assert!(catalog
        .register(unrelated, Arc::from(&b"unrelated"[..]), "run")
        .is_err());
    assert_eq!(catalog.state.lock().unwrap().pending_results.len(), 1);
    drop(full);
    assert!(catalog.state.lock().unwrap().pending_results.is_empty());
}

#[test]
fn stale_results_are_discarded_without_evicting_inputs_or_commits() {
    let catalog = RemoteObjectCatalog::new().unwrap();
    let input_bytes = Arc::<[u8]>::from(&b"registered input"[..]);
    let input = reference("run", input_bytes.as_ref(), b"input");
    catalog
        .register(input.clone(), Arc::clone(&input_bytes), "run")
        .unwrap();
    assert!(catalog.contains(&input).unwrap());
    let mut driver_alias = input.clone();
    driver_alias.kind = ValueRefKind::DriverObject;
    catalog
        .register(driver_alias.clone(), Arc::clone(&input_bytes), "run")
        .unwrap();
    assert!(catalog.contains(&driver_alias).unwrap());
    let mut conflicting_input = input.clone();
    conflicting_input.encoded_length += 1;
    assert!(catalog.contains(&conflicting_input).is_err());

    let stale_bytes = Arc::<[u8]>::from(&b"stale result"[..]);
    let stale = reference("run", stale_bytes.as_ref(), b"stale");
    let committed_bytes = Arc::<[u8]>::from(&b"committed result"[..]);
    let committed = reference("run", committed_bytes.as_ref(), b"committed");
    store(&catalog.store, &stale, stale_bytes.as_ref()).unwrap();
    store(&catalog.store, &committed, committed_bytes.as_ref()).unwrap();
    {
        let mut state = catalog.state.lock().unwrap();
        insert(&mut state.objects, stale.clone()).unwrap();
        insert(&mut state.objects, committed.clone()).unwrap();
    }
    catalog
        .commit_results(std::slice::from_ref(&committed))
        .unwrap();
    catalog.discard_results(&[stale.clone(), input.clone(), committed.clone()]);

    assert!(catalog.get(&stale).unwrap().is_none());
    assert!(!catalog.contains(&stale).unwrap());
    assert!(catalog.get(&input).unwrap().is_some());
    assert_eq!(
        catalog.get(&committed).unwrap().unwrap().as_ref(),
        committed_bytes.as_ref()
    );
}

fn reference(scope: &str, bytes: &[u8], identity: &[u8]) -> ValueRef {
    ValueRef {
        schema_version: runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION,
        id: ValueId::derive(&[b"remote-object-test", identity]),
        logical_digest: Digest::sha256(bytes),
        encoded_length: bytes.len() as u64,
        media_type: "application/vnd.runmat.test-object".into(),
        value_schema: "runmat.test-object.v1".into(),
        encryption_context: Digest::sha256(b"remote-object-test-context"),
        kind: ValueRefKind::ResultObject,
        authorization_scope: scope.into(),
        resident_fence: None,
    }
}

// witness: FEA-MESH-EXECUTION-001/worker_loss_and_transfer @
//   shared_infrastructure/canonical_mixed_topology/remote_execution; coverage: partial;
//   boundary: The remote meshing object catalog bounds actual files and preserves content identity.
#[test]
fn remote_meshing_catalog_bounds_actual_files_and_preserves_content_identity() {
    use runmat_execution_artifact::ArtifactError;
    let catalog = RemoteObjectCatalog::new().unwrap();
    let bytes = b"data";
    let admitted = reference("run", bytes, b"bounded");
    catalog
        .register(admitted.clone(), Arc::from(&bytes[..]), "run")
        .unwrap();
    let digest = admitted.logical_digest;
    for limit in [0, 3] {
        assert!(matches!(
            catalog.get_by_digest_bounded(digest, limit),
            Err(ArtifactError::Limit(_))
        ));
    }
    assert_eq!(
        catalog.get_by_digest_bounded(digest, 4).unwrap(),
        Some(bytes.to_vec())
    );
    let files = std::fs::read_dir(catalog.store.root())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    assert_eq!(files.len(), 1);
    let path = &files[0];
    // Stale catalog length cannot authorize a larger poisoned backing file.
    std::fs::write(path, b"wrong").unwrap();
    assert!(matches!(
        catalog.get_by_digest_bounded(digest, 4),
        Err(ArtifactError::Limit(_))
    ));
    assert!(matches!(
        catalog.get_by_digest_bounded(digest, 5),
        Err(ArtifactError::Identity(_))
    ));
    let unregistered = reference("run", b"unregistered", b"not-admitted");
    store(&catalog.store, &unregistered, b"unregistered").unwrap();
    assert_eq!(
        catalog
            .get_by_digest_bounded(unregistered.logical_digest, 12)
            .unwrap(),
        None
    );
}
