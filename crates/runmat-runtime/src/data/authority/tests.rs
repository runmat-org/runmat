use super::*;

fn fixture_root(name: &str) -> PathBuf {
    let base = PathBuf::from("/runmat-authority");
    base.join(name)
}

fn object(role: HandleRole, path: &str, reference: ForeignRef) -> Value {
    let mut object = runmat_value::UnboundedObjectBuilder::new(role.class().owned());
    object.insert("__data_path".into(), Value::String(path.into()));
    match role {
        HandleRole::Dataset => {}
        HandleRole::Array(name) => {
            object.insert("__array_name".into(), Value::String(name.into()));
        }
        HandleRole::Transaction(tx_id) => {
            object.insert("__tx_id".into(), Value::String(tx_id.into()));
        }
    }
    Value::Object(object.finish().with_dataset_authority(reference))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn same_public_path_keeps_distinct_owner_and_clone_lease() {
    let first: Arc<dyn FsProvider> = Arc::new(
        runmat_filesystem::MemoryFsProvider::with_current_dir(fixture_root("one")),
    );
    let second: Arc<dyn FsProvider> = Arc::new(
        runmat_filesystem::MemoryFsProvider::with_current_dir(fixture_root("two")),
    );
    let first_ref = issue_dataset(
        None,
        first.clone(),
        fixture_root("one").join("same.data"),
        "same.data".into(),
    )
    .expect("first owner");
    let second_ref = issue_dataset(
        None,
        second.clone(),
        fixture_root("two").join("same.data"),
        "same.data".into(),
    )
    .expect("second owner");
    let original = object(HandleRole::Dataset, "same.data", first_ref);
    let copied = original.clone();
    let other = object(HandleRole::Dataset, "same.data", second_ref);
    let resolved = resolve_object(None, &copied, HandleRole::Dataset).expect("cloned first owner");
    assert!(Arc::ptr_eq(&resolved.filesystem, &first));
    assert_eq!(resolved.root, fixture_root("one").join("same.data"));
    assert!(!Arc::ptr_eq(
        &resolve_object(None, &other, HandleRole::Dataset)
            .expect("second owner")
            .filesystem,
        &resolved.filesystem
    ));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn detached_and_forged_public_fields_cannot_resolve() {
    #[derive(Debug)]
    struct NoopRelease;
    impl ForeignResourceRelease for NoopRelease {
        fn release(&self, _: &ForeignResourceKey) {}
    }

    let filesystem: Arc<dyn FsProvider> = Arc::new(
        runmat_filesystem::MemoryFsProvider::with_current_dir(fixture_root("one")),
    );
    let issued = issue_dataset(
        None,
        filesystem,
        fixture_root("one").join("same.data"),
        "same.data".into(),
    )
    .expect("issued owner");
    let key = issued.resource_key();
    let type_identity = issued.type_identity.clone();
    let detached = ForeignRef::detached(
        key.clone(),
        type_identity.clone(),
        ForeignOwnership::Shared,
        ForeignAffinity::OriginProcess,
        ForeignLifetime::Session,
    );
    assert!(resolve_object(
        None,
        &object(HandleRole::Dataset, "same.data", detached),
        HandleRole::Dataset
    )
    .is_err());
    let different_lease = ForeignRef::managed(
        key,
        type_identity,
        ForeignOwnership::Shared,
        ForeignAffinity::OriginProcess,
        ForeignLifetime::Session,
        Arc::new(NoopRelease),
    );
    assert!(resolve_object(
        None,
        &object(HandleRole::Dataset, "same.data", different_lease),
        HandleRole::Dataset
    )
    .is_err());
    assert!(resolve_object(
        None,
        &object(HandleRole::Dataset, "changed.data", issued),
        HandleRole::Dataset
    )
    .is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn derived_array_and_transaction_roles_are_sealed_independently() {
    let filesystem: Arc<dyn FsProvider> = Arc::new(
        runmat_filesystem::MemoryFsProvider::with_current_dir(fixture_root("one")),
    );
    let parent = object(
        HandleRole::Dataset,
        "same.data",
        issue_dataset(
            None,
            filesystem,
            fixture_root("one").join("same.data"),
            "same.data".into(),
        )
        .unwrap(),
    );
    let parent_authority = resolve_object(None, &parent, HandleRole::Dataset).unwrap();
    let array_ref = issue_child(&parent_authority, HandleRole::Array("samples".into())).unwrap();
    let transaction_ref =
        issue_child(&parent_authority, HandleRole::Transaction("tx-1".into())).unwrap();
    let valid_array = object(
        HandleRole::Array("samples".into()),
        "same.data",
        array_ref.clone(),
    );
    assert!(resolve_object(None, &valid_array, HandleRole::Array("samples".into())).is_ok());
    assert!(resolve_object(
        None,
        &object(HandleRole::Array("other".into()), "same.data", array_ref),
        HandleRole::Array("other".into())
    )
    .is_err());
    let valid_tx = object(
        HandleRole::Transaction("tx-1".into()),
        "same.data",
        transaction_ref.clone(),
    );
    assert!(resolve_object(None, &valid_tx, HandleRole::Transaction("tx-1".into())).is_ok());
    assert!(resolve_object(
        None,
        &object(
            HandleRole::Transaction("tx-2".into()),
            "same.data",
            transaction_ref
        ),
        HandleRole::Transaction("tx-2".into())
    )
    .is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn final_clone_drop_releases_only_its_own_authority_record() {
    let filesystem: Arc<dyn FsProvider> = Arc::new(
        runmat_filesystem::MemoryFsProvider::with_current_dir(fixture_root("one")),
    );
    let first = issue_dataset(
        None,
        filesystem.clone(),
        fixture_root("one").join("first.data"),
        "first.data".into(),
    )
    .unwrap();
    let second = issue_dataset(
        None,
        filesystem,
        fixture_root("one").join("second.data"),
        "second.data".into(),
    )
    .unwrap();
    let first_handle = first.handle;
    let second_handle = second.handle;
    let clone = first.clone();
    drop(first);
    assert!(registry()
        .lock()
        .unwrap()
        .records
        .contains_key(&first_handle));
    drop(clone);
    let records = registry().lock().unwrap();
    assert!(!records.records.contains_key(&first_handle));
    assert!(records.records.contains_key(&second_handle));
    drop(records);
    drop(second);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn session_handle_survives_invocations_but_not_cross_session_or_end() {
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;
    use runmat_execution::cancellation::CancellationSource;
    use std::rc::Rc;

    let session = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let unrelated = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let worker = session
        .fork_parallel_lab(RuntimeServicePorts::default())
        .expect("idle session fork");
    let filesystem: Arc<dyn FsProvider> = Arc::new(
        runmat_filesystem::MemoryFsProvider::with_current_dir(fixture_root("session")),
    );
    let issued = issue_dataset(
        Some(&session),
        filesystem,
        fixture_root("session").join("owned.data"),
        "owned.data".into(),
    )
    .unwrap();
    assert_eq!(issued.lifetime, ForeignLifetime::Session);
    let receiver = object(HandleRole::Dataset, "owned.data", issued);
    let next_invocation = session
        .clone()
        .with_cancellation_token(CancellationSource::new().token());
    assert!(resolve_object(Some(&next_invocation), &receiver, HandleRole::Dataset).is_ok());
    assert!(resolve_object(Some(&unrelated), &receiver, HandleRole::Dataset).is_err());
    assert!(resolve_object(Some(&worker), &receiver, HandleRole::Dataset).is_err());
    assert!(resolve_object(None, &receiver, HandleRole::Dataset).is_err());
    drop(next_invocation);
    drop(session);
    let Value::Object(object) = &receiver else {
        unreachable!()
    };
    let handle = object.dataset_authority().unwrap().handle;
    let records = registry().lock().unwrap();
    let IssuerScope::Session { lease, .. } = &records.records.get(&handle).unwrap().issuer else {
        unreachable!()
    };
    assert!(
        lease.upgrade().is_none(),
        "receiver must not prolong its session"
    );
    drop(records);
    assert!(resolve_object(Some(&unrelated), &receiver, HandleRole::Dataset).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn context_free_embedding_handle_is_explicit_process_local_scope() {
    let filesystem: Arc<dyn FsProvider> = Arc::new(
        runmat_filesystem::MemoryFsProvider::with_current_dir(fixture_root("embedding")),
    );
    let issued = issue_dataset(
        None,
        filesystem,
        fixture_root("embedding").join("owned.data"),
        "owned.data".into(),
    )
    .unwrap();
    assert_eq!(issued.lifetime, ForeignLifetime::Persistent);
    let receiver = object(HandleRole::Dataset, "owned.data", issued);
    assert!(resolve_object(None, &receiver, HandleRole::Dataset).is_ok());
}
