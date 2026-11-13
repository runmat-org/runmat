use super::*;
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::{CharArray, IntegerStorage, StringArray, Tensor, Value};
use std::fs::File;
use std::sync::Arc;
use tempfile::tempdir;

fn delete_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args =
        crate::execution::RootedValues::new(args).map_err(|error| error.into_runtime_error())?;
    futures::executor::block_on(super::delete_builtin(None, args.into()))
}

fn ident(error: &'static BuiltinErrorDescriptor) -> Option<&'static str> {
    error.identifier
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = DELETE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"delete(filename)"));
    assert!(labels.contains(&"delete(filename1, filename2, ...)"));
    assert!(labels.contains(&"delete(___, ResolveSymbolicLinks=tf)"));
}

#[test]
fn resolve_symbolic_links_accepts_every_integer_class_only_at_zero_or_one() {
    let storages = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];
    for storage in storages {
        let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
        assert!(exact_zero_one(&value).unwrap());
    }
    assert!(!exact_zero_one(&Value::Int(runmat_value::IntValue::U64(0))).unwrap());
    assert!(exact_zero_one(&Value::Int(runmat_value::IntValue::U64(u64::MAX))).is_err());
    let nonscalar = Tensor::new_integer(IntegerStorage::U8(vec![0, 1]), vec![1, 2]).unwrap();
    assert!(exact_zero_one(&Value::Tensor(nonscalar)).is_err());
}

#[cfg(unix)]
#[test]
fn delete_symbolic_link_can_remove_link_or_target_and_leaves_link_when_resolved() {
    use std::os::unix::fs::symlink;

    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempdir().unwrap();
    let target = temp.path().join("target.txt");
    let link = temp.path().join("target-link");
    File::create(&target).unwrap();
    symlink(&target, &link).unwrap();
    delete_builtin(vec![
        Value::from(link.to_string_lossy().to_string()),
        Value::from("ResolveSymbolicLinks"),
        Value::Bool(false),
    ])
    .unwrap();
    assert!(target.exists());
    assert!(std::fs::symlink_metadata(&link).is_err());

    symlink(&target, &link).unwrap();
    delete_builtin(vec![
        Value::from(link.to_string_lossy().to_string()),
        Value::from("ResolveSymbolicLinks"),
        Value::Int(runmat_value::IntValue::U8(1)),
    ])
    .unwrap();
    assert!(!target.exists());
    assert!(std::fs::symlink_metadata(&link)
        .unwrap()
        .file_type()
        .is_symlink());

    let folder_target = temp.path().join("empty-folder");
    let folder_link = temp.path().join("empty-folder-link");
    std::fs::create_dir(&folder_target).unwrap();
    symlink(&folder_target, &folder_link).unwrap();
    delete_builtin(vec![
        Value::from(folder_link.to_string_lossy().to_string()),
        Value::from("ResolveSymbolicLinks"),
        Value::Bool(true),
    ])
    .unwrap();
    assert!(!folder_target.exists());
    assert!(std::fs::symlink_metadata(&folder_link)
        .unwrap()
        .file_type()
        .is_symlink());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_removes_single_file() {
    let _lock = runmat_filesystem::provider_override_lock();

    let temp = tempdir().expect("temp dir");
    let target = temp.path().join("single.txt");
    File::create(&target).expect("create");

    let result =
        delete_builtin(vec![Value::from(target.to_string_lossy().to_string())]).expect("delete");
    assert_eq!(result, Value::Num(0.0));
    assert!(!target.exists());
}

#[test]
fn delete_rejects_malformed_options_before_deleting_any_path() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempdir().unwrap();
    let target = temp.path().join("must-remain.txt");
    File::create(&target).unwrap();
    let path = Value::from(target.to_string_lossy().to_string());

    let unknown = delete_builtin(vec![
        path.clone(),
        Value::from("UnknownOption"),
        Value::Bool(true),
    ])
    .expect_err("unknown option must reject");
    assert_eq!(unknown.identifier(), ident(&DELETE_ERROR_INVALID_INPUT));
    assert!(target.exists());

    let missing = delete_builtin(vec![path, Value::from("ResolveSymbolicLinks")])
        .expect_err("missing option value must reject");
    assert_eq!(missing.identifier(), ident(&DELETE_ERROR_INVALID_INPUT));
    assert!(target.exists());
}

#[test]
fn delete_rejects_nested_resident_input_before_provider_access() {
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle,
    );
    let nested = crate::make_cell(vec![resident], 1, 1).unwrap();
    let error = delete_builtin(vec![nested]).expect_err("nested resident path must reject");
    assert_eq!(error.identifier(), ident(&DELETE_ERROR_INVALID_INPUT));
    assert!(!error.message().to_ascii_lowercase().contains("provider"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_handle_with_resident_property_preserves_opaque_payload() {
    runmat_gc::gc_test_context(|| {
        let allocation =
            crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1]);
        let resident = Value::GpuTensor(allocation.handle);
        let value = futures::executor::block_on(crate::new_handle_object_builtin(
            "ReplFsDeleteTestHandle".to_string(),
        ))
        .expect("handle");
        let Value::HandleObject(handle) = &value else {
            panic!("expected handle")
        };
        runmat_gc::gc_with_value_mut(&handle.target, |target| {
            let Value::Object(object) = target else {
                panic!("expected object")
            };
            object.replace_property_unbounded("data".into(), resident.clone());
        })
        .expect("store resident property");
        assert!(crate::value_contains_gpu(&value));
        let result = delete_builtin(vec![value.clone()]).expect("delete opaque handle");
        assert!(!crate::is_handle_valid(handle));
        assert!(matches!(result, Value::HandleObject(ref deleted) if !deleted.valid));
        let payload = runmat_gc::gc_with_value(&handle.target, |target| {
            let Value::Object(object) = target else {
                panic!("expected object")
            };
            object.properties().get("data").cloned()
        })
        .expect("read retained payload");
        assert_eq!(payload, Some(resident));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_listener_preserves_resident_referents() {
    runmat_gc::gc_test_context(|| {
        let allocation =
            crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1]);
        let resident = Value::GpuTensor(allocation.handle);
        let target = runmat_gc::gc_allocate(resident.clone()).expect("listener target");
        let listener = runmat_value::Listener {
            id: u64::MAX,
            target,
            target_class_name: "ReplFsDeleteTestHandle".into(),
            event_name: "changed".into(),
            callback: target,
            enabled: true,
            valid: true,
        };
        let value = Value::Listener(listener);
        assert!(crate::value_contains_gpu(&value));
        let result = delete_builtin(vec![value]).expect("delete opaque listener");
        let Value::Listener(deleted) = result else {
            panic!("expected listener")
        };
        assert!(!deleted.valid);
        assert!(!deleted.enabled);
        assert_eq!(deleted.target, target);
        assert_eq!(deleted.callback, target);
        assert_eq!(
            runmat_gc::gc_with_value(&target, Clone::clone).expect("retained referent"),
            resident
        );
    });
}

#[test]
fn delete_uses_documented_sequential_multi_target_mutation() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempdir().unwrap();
    let first = temp.path().join("first.txt");
    let missing = temp.path().join("missing.txt");
    File::create(&first).unwrap();
    let error = delete_builtin(vec![
        Value::from(first.to_string_lossy().to_string()),
        Value::from(missing.to_string_lossy().to_string()),
    ])
    .expect_err("later missing target must stop sequential deletion");
    assert_eq!(error.identifier(), ident(&DELETE_ERROR_FILE_NOT_FOUND));
    assert!(!first.exists(), "earlier targets are deleted sequentially");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_removes_files_with_wildcard() {
    let _lock = runmat_filesystem::provider_override_lock();

    let temp = tempdir().expect("temp dir");
    let file_a = temp.path().join("log-01.txt");
    let file_b = temp.path().join("log-02.txt");
    File::create(&file_a).expect("create a");
    File::create(&file_b).expect("create b");

    let pattern = temp.path().join("log-*.txt");
    delete_builtin(vec![Value::from(pattern.to_string_lossy().to_string())]).expect("delete");
    assert!(!file_a.exists());
    assert!(!file_b.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_accepts_string_array() {
    let _lock = runmat_filesystem::provider_override_lock();

    let temp = tempdir().expect("temp dir");
    let file_a = temp.path().join("stageA.dat");
    let file_b = temp.path().join("stageB.dat");
    File::create(&file_a).expect("create a");
    File::create(&file_b).expect("create b");

    let array = StringArray::new(
        vec![
            file_a.to_string_lossy().to_string(),
            file_b.to_string_lossy().to_string(),
        ],
        vec![2],
    )
    .expect("string array");

    delete_builtin(vec![Value::StringArray(array)]).expect("delete");
    assert!(!file_a.exists());
    assert!(!file_b.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_accepts_char_array() {
    let _lock = runmat_filesystem::provider_override_lock();

    let temp = tempdir().expect("temp dir");
    let paths: Vec<_> = ["stageA.tmp", "stageB.tmp"]
        .into_iter()
        .map(|name| temp.path().join(name))
        .collect();
    let path_strings: Vec<String> = paths
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    let max_len = path_strings
        .iter()
        .map(|text| text.encode_utf16().count())
        .max()
        .unwrap();
    let mut data: Vec<char> = Vec::with_capacity(path_strings.len() * max_len);

    for (path, path_string) in paths.iter().zip(path_strings.iter()) {
        File::create(path).expect("create file");
        let mut chars: Vec<char> = path_string.chars().collect();
        chars.extend(std::iter::repeat_n(
            ' ',
            max_len - path_string.encode_utf16().count(),
        ));
        data.extend(&chars);
    }

    let char_array =
        CharArray::from_unicode_scalars(data, path_strings.len(), max_len).expect("char array");
    delete_builtin(vec![Value::CharArray(char_array)]).expect("delete");

    for path in paths {
        assert!(!path.exists(), "{path:?} should be removed");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_accepts_cell_array_of_paths() {
    let _lock = runmat_filesystem::provider_override_lock();

    let temp = tempdir().expect("temp dir");
    let file_a = temp.path().join("cellA.dat");
    let file_b = temp.path().join("cellB.dat");
    File::create(&file_a).expect("create cellA");
    File::create(&file_b).expect("create cellB");

    let cell_value = crate::make_cell(
        vec![
            Value::from(file_a.to_string_lossy().to_string()),
            Value::from(file_b.to_string_lossy().to_string()),
        ],
        1,
        2,
    )
    .expect("cell");

    delete_builtin(vec![cell_value]).expect("delete");
    assert!(!file_a.exists());
    assert!(!file_b.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_empty_string_array_is_noop() {
    let array = StringArray::new(Vec::<String>::new(), vec![0]).expect("empty array");
    let result = delete_builtin(vec![Value::StringArray(array)]).expect("delete");
    assert_eq!(result, Value::Num(0.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_errors_on_empty_string_argument() {
    let err = delete_builtin(vec![Value::from(String::new())]).expect_err("empty string");
    assert_eq!(err.identifier(), ident(&DELETE_ERROR_EMPTY_FILENAME));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_errors_on_string_array_empty_element() {
    let array =
        StringArray::new(vec![String::new()], vec![1]).expect("single empty string element");
    let err = delete_builtin(vec![Value::StringArray(array)]).expect_err("empty element");
    assert_eq!(err.identifier(), ident(&DELETE_ERROR_EMPTY_FILENAME));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_errors_on_char_array_blank_row() {
    let data = vec![' '; 4];
    let char_array = CharArray::from_unicode_scalars(data, 1, 4).expect("char matrix");
    let err = delete_builtin(vec![Value::CharArray(char_array)]).expect_err("blank row");
    assert_eq!(err.identifier(), ident(&DELETE_ERROR_EMPTY_FILENAME));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_errors_on_invalid_pattern() {
    let pattern = "{invalid*";
    let fs = DeleteFilesystem::capture().expect("filesystem");
    let mut cooperation = DeleteCooperation::new(None);
    let err = futures::executor::block_on(delete_target(&fs, &mut cooperation, pattern, false))
        .expect_err("invalid pattern should error");
    assert_eq!(err.identifier(), ident(&DELETE_ERROR_INVALID_PATTERN));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_errors_on_missing_file() {
    let _lock = runmat_filesystem::provider_override_lock();

    let temp = tempdir().expect("temp dir");
    let missing = temp.path().join("missing.txt");
    let missing_str = missing.to_string_lossy().to_string();
    let fs = DeleteFilesystem::capture().expect("filesystem");
    let mut cooperation = DeleteCooperation::new(None);
    let err =
        futures::executor::block_on(delete_target(&fs, &mut cooperation, &missing_str, false))
            .expect_err("error");
    assert_eq!(err.identifier(), ident(&DELETE_ERROR_FILE_NOT_FOUND));
    assert!(err.source.as_deref().is_some_and(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
    }));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_errors_on_directory() {
    let _lock = runmat_filesystem::provider_override_lock();

    let temp = tempdir().expect("temp dir");
    let dir = temp.path().join("dir");
    std::fs::create_dir(&dir).expect("create dir");
    let dir_display = dir.to_string_lossy().to_string();
    let err = delete_single_path(&dir, &dir_display).expect_err("error");
    assert_eq!(err.identifier(), ident(&DELETE_ERROR_IS_DIRECTORY));
}

#[test]
fn captured_virtual_provider_and_cwd_govern_wildcard_after_provider_swap() {
    use runmat_filesystem::MemoryFsProvider;

    let _lock = runmat_filesystem::provider_override_lock();
    let memory = Arc::new(MemoryFsProvider::with_current_dir("/workspace"));
    memory
        .write_project_path("/workspace/log-01.txt", b"first")
        .expect("first virtual file");
    memory
        .write_project_path("/workspace/log-02.txt", b"second")
        .expect("second virtual file");
    memory
        .write_project_path("/workspace/keep.bin", b"keep")
        .expect("unmatched virtual file");

    let captured = runmat_filesystem::with_provider_override(memory.clone(), || {
        DeleteFilesystem::capture().expect("capture exact provider and virtual cwd")
    });
    // The global provider has been restored. The retained operation must not
    // enumerate or remove from the newly active process filesystem.
    let mut cooperation = DeleteCooperation::new(None);
    futures::executor::block_on(delete_target(
        &captured,
        &mut cooperation,
        "log-*.txt",
        false,
    ))
    .expect("delete the captured provider's two matching files");
    assert!(memory.read_project_path("/workspace/log-01.txt").is_err());
    assert!(memory.read_project_path("/workspace/log-02.txt").is_err());
    assert_eq!(
        memory.read_project_path("/workspace/keep.bin").unwrap(),
        b"keep"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelled_invocation_rejects_before_filesystem_mutation_with_typed_cause() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use runmat_filesystem::MemoryFsProvider;
    use std::rc::Rc;

    let _lock = runmat_filesystem::provider_override_lock();
    let memory = Arc::new(MemoryFsProvider::with_current_dir("/workspace"));
    memory
        .write_project_path("/workspace/keep.txt", b"retained")
        .expect("virtual file");
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(cancellation.token());
    let error = runmat_filesystem::with_provider_override(memory.clone(), || {
        futures::executor::block_on(super::delete_builtin(
            Some(runtime),
            crate::execution::RootedValues::new(vec![Value::String("keep.txt".into())])
                .expect("source argument owner")
                .into(),
        ))
        .expect_err("cancelled call")
    });
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(error.source.as_deref().is_some_and(|cause| {
        matches!(
            cause.downcast_ref::<crate::execution::ExecutionServiceError>(),
            Some(crate::execution::ExecutionServiceError::Cancelled)
        )
    }));
    assert_eq!(
        memory.read_project_path("/workspace/keep.txt").unwrap(),
        b"retained"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_handle_returns_invalid_handle() {
    let handle = futures::executor::block_on(crate::new_handle_object_builtin(
        "ReplFsDeleteTestHandle".to_string(),
    ))
    .expect("handle");
    let original_handle = handle.clone();
    let result = delete_builtin(vec![handle]).expect("delete handle");
    match result {
        Value::HandleObject(h) => {
            assert!(!h.valid, "handle should be marked invalid");
            let valid_value =
                futures::executor::block_on(crate::isvalid_builtin(Value::HandleObject(h.clone())))
                    .expect("isvalid");
            match valid_value {
                Value::Bool(flag) => assert!(!flag, "isvalid should report false after delete"),
                other => panic!("expected bool from isvalid, got {other:?}"),
            }
            let original_valid_value =
                futures::executor::block_on(crate::isvalid_builtin(original_handle))
                    .expect("isvalid");
            match original_valid_value {
                Value::Bool(flag) => assert!(!flag, "original alias should also be invalid"),
                other => panic!("expected bool from isvalid, got {other:?}"),
            }
        }
        other => panic!("expected handle result, got {other:?}"),
    }
}

#[test]
fn delete_rejects_symbolic_link_option_for_handle_input() {
    let handle = futures::executor::block_on(crate::new_handle_object_builtin(
        "DeleteResolveOptionHandle".to_string(),
    ))
    .expect("handle");
    let error = delete_builtin(vec![
        handle.clone(),
        Value::from("ResolveSymbolicLinks"),
        Value::Bool(false),
    ])
    .expect_err("filename-only option must reject for handles");
    assert_eq!(error.identifier(), ident(&DELETE_ERROR_INVALID_INPUT));
    assert!(matches!(
        futures::executor::block_on(crate::isvalid_builtin(handle)).unwrap(),
        Value::Bool(true)
    ));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_rejects_mixed_handle_and_filename() {
    let handle = futures::executor::block_on(crate::new_handle_object_builtin(
        "ReplFsDeleteTestHandle".to_string(),
    ))
    .expect("handle");
    let err = delete_builtin(vec![
        handle,
        Value::from("mixed-handle-path.txt".to_string()),
    ])
    .expect_err("expected mixed error");
    assert_eq!(err.identifier(), ident(&DELETE_ERROR_INVALID_HANDLE));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_accepts_cell_of_handles() {
    let handle_a = futures::executor::block_on(crate::new_handle_object_builtin(
        "ReplFsDeleteTestHandle".to_string(),
    ))
    .expect("handle");
    let handle_b = futures::executor::block_on(crate::new_handle_object_builtin(
        "ReplFsDeleteTestHandle".to_string(),
    ))
    .expect("handle");
    let cell = crate::make_cell(vec![handle_a, handle_b], 1, 2).expect("cell of handles");
    let result = delete_builtin(vec![cell]).expect("delete handles");
    assert_eq!(result, Value::Num(0.0));
}

#[cfg(feature = "wgpu")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn delete_runs_with_wgpu_provider_registered() {
    let _lock = runmat_filesystem::provider_override_lock();

    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );

    let temp = tempdir().expect("temp dir");
    let path = temp.path().join("wgpu-file.txt");
    File::create(&path).expect("create file");

    delete_builtin(vec![Value::from(path.to_string_lossy().to_string())]).expect("delete");
    assert!(!path.exists());
}
