use super::*;

fn fixture_root(name: &str) -> PathBuf {
    #[cfg(windows)]
    let base = PathBuf::from(r"C:\runmat-authority");
    #[cfg(not(windows))]
    let base = PathBuf::from("/runmat-authority");
    base.join(name)
}

fn object(role: HandleRole, path: &str, reference: ForeignRef) -> Value {
    let mut object = ObjectInstance::new(role.class().owned()).with_dataset_authority(reference);
    object
        .properties
        .insert("__data_path".into(), Value::String(path.into()));
    match role {
        HandleRole::Dataset => {}
        HandleRole::Array(name) => {
            object
                .properties
                .insert("__array_name".into(), Value::String(name));
        }
        HandleRole::Transaction(tx_id) => {
            object
                .properties
                .insert("__tx_id".into(), Value::String(tx_id));
        }
    }
    Value::Object(object)
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
        first.clone(),
        fixture_root("one").join("same.data"),
        "same.data".into(),
    )
    .expect("first owner");
    let second_ref = issue_dataset(
        second.clone(),
        fixture_root("two").join("same.data"),
        "same.data".into(),
    )
    .expect("second owner");
    let original = object(HandleRole::Dataset, "same.data", first_ref);
    let copied = original.clone();
    let other = object(HandleRole::Dataset, "same.data", second_ref);
    let resolved = resolve_object(&copied, HandleRole::Dataset).expect("cloned first owner");
    assert!(Arc::ptr_eq(&resolved.filesystem, &first));
    assert_eq!(resolved.root, fixture_root("one").join("same.data"));
    assert!(!Arc::ptr_eq(
        &resolve_object(&other, HandleRole::Dataset)
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
        &object(HandleRole::Dataset, "same.data", different_lease),
        HandleRole::Dataset
    )
    .is_err());
    assert!(resolve_object(
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
            filesystem,
            fixture_root("one").join("same.data"),
            "same.data".into(),
        )
        .unwrap(),
    );
    let parent_authority = resolve_object(&parent, HandleRole::Dataset).unwrap();
    let array_ref = issue_child(&parent_authority, HandleRole::Array("samples".into())).unwrap();
    let transaction_ref =
        issue_child(&parent_authority, HandleRole::Transaction("tx-1".into())).unwrap();
    let valid_array = object(
        HandleRole::Array("samples".into()),
        "same.data",
        array_ref.clone(),
    );
    assert!(resolve_object(&valid_array, HandleRole::Array("samples".into())).is_ok());
    assert!(resolve_object(
        &object(HandleRole::Array("other".into()), "same.data", array_ref),
        HandleRole::Array("other".into())
    )
    .is_err());
    let valid_tx = object(
        HandleRole::Transaction("tx-1".into()),
        "same.data",
        transaction_ref.clone(),
    );
    assert!(resolve_object(&valid_tx, HandleRole::Transaction("tx-1".into())).is_ok());
    assert!(resolve_object(
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
        filesystem.clone(),
        fixture_root("one").join("first.data"),
        "first.data".into(),
    )
    .unwrap();
    let second = issue_dataset(
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
