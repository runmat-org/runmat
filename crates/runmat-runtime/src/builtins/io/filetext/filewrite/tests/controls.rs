use super::*;

#[test]
fn filewrite_is_gated_before_filesystem_side_effects() {
    let target = unique_path("filewrite_strict_gate");
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = futures::executor::block_on(filewrite_builtin(
        Value::from(target.to_string_lossy().to_string()),
        Value::from("must not be written"),
        Vec::new(),
    ))
    .expect_err("strict compatibility mode must reject RunMat-native filewrite");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FilewriteNativeExtension")
    );
    assert!(!target.exists());
}

#[test]
fn filewrite_integer_metadata_covers_every_native_class() {
    let capability = &FILEWRITE_INTEGER_CAPABILITIES[0];
    assert_eq!(capability.inputs[0].classes.len(), 8);
    assert_eq!(
        capability.backend,
        BuiltinIntegerBackendRule::GatherFallback
    );
    assert_eq!(FILEWRITE_EXTENSIONS[0].id, "filewrite-runmat-native");
}

#[test]
fn filewrite_gathers_resident_integer_bytes_before_host_io() {
    test_support::with_test_provider(|provider| {
        let values = [0_u64, 127, 255];
        let handle = provider
            .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                data: runmat_accelerate_api::HostIntegerDataView::U64(&values),
                shape: &[1, 3],
            })
            .expect("upload integer bytes");
        let target = unique_path("filewrite_resident_integer");
        let result = run_filewrite(
            Value::from(target.to_string_lossy().to_string()),
            Value::GpuTensor(handle),
            Vec::new(),
        )
        .expect("resident integer filewrite");
        assert_eq!(result, Value::Num(3.0));
        assert_eq!(test_support::fs::read(&target).unwrap(), vec![0, 127, 255]);
        test_support::fs::remove_file(target).unwrap();
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = FILEWRITE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"count = filewrite(filename, data)"));
    assert!(labels.contains(&"count = filewrite(filename, data, encoding)"));
    assert!(labels.contains(&"count = filewrite(filename, data, \"Encoding\", encoding)"));
    assert!(labels.contains(&"count = filewrite(filename, data, ..., \"WriteMode\", mode)"));
}
