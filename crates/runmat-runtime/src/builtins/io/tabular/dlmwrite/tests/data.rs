use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn known_invalid_resident_matrix_shape_rejects_before_download_or_file_creation() {
    test_support::with_test_provider(|provider| {
        let _fs_lock = runmat_filesystem::provider_override_lock();
        let filesystem = std::sync::Arc::new(
            runmat_filesystem::MemoryFsProvider::with_current_dir("/tabular"),
        );
        let _selected = runmat_filesystem::replace_provider(filesystem);
        runmat_filesystem::set_current_dir("/tabular").expect("fixture cwd");
        let handle = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 1, 2],
            })
            .expect("resident N-D data");
        provider.reset_telemetry();
        let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
        let path = "/tabular/invalid-resident.csv";
        let error = futures::executor::block_on(super::super::dlmwrite_builtin(
            None,
            Value::from(path),
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect_err("non-singleton page is invalid data geometry");
        assert!(error.message().contains("2-D"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(futures::executor::block_on(runmat_filesystem::metadata_async(path)).is_err());
        handle.release().expect("release resident data");
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_writes_default_comma() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let filename = path.to_string_lossy().into_owned();

    dlmwrite_builtin(Value::from(filename), Value::Tensor(tensor), Vec::new()).unwrap();

    let contents = fs::read_to_string(&path).unwrap();
    let nl = platform_newline();
    assert_eq!(contents, format!("1,2,3{nl}4,5,6{nl}"));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_preserves_typed_integer_matrix_values_exactly_by_default() {
    let path = temp_path("csv");
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX, 17, (1_u64 << 53) + 1, 29]),
        vec![2, 2],
    )
    .expect("typed integer matrix");
    let filename = path.to_string_lossy().into_owned();

    dlmwrite_builtin(Value::from(filename), Value::Tensor(tensor), Vec::new()).unwrap();

    let contents = fs::read_to_string(&path).unwrap();
    let nl = platform_newline();
    assert_eq!(
        contents,
        format!("18446744073709551615,9007199254740993{nl}17,29{nl}")
    );
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dlmwrite_preserves_public_low_rank_matrix_projection() {
    let rank_zero_path = temp_path("csv");
    let rank_zero = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), Vec::new())
        .expect("rank-zero integer");
    dlmwrite_builtin(
        Value::from(rank_zero_path.to_string_lossy().into_owned()),
        Value::Tensor(rank_zero),
        Vec::new(),
    )
    .expect("write rank-zero tensor");
    assert_eq!(
        fs::read_to_string(&rank_zero_path).unwrap(),
        format!("18446744073709551615{}", platform_newline())
    );
    let _ = fs::remove_file(rank_zero_path);

    let rank_one_path = temp_path("csv");
    let rank_one = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        vec![2],
    )
    .expect("rank-one integer");
    dlmwrite_builtin(
        Value::from(rank_one_path.to_string_lossy().into_owned()),
        Value::Tensor(rank_one),
        Vec::new(),
    )
    .expect("write rank-one tensor");
    assert_eq!(
        fs::read_to_string(&rank_one_path).unwrap(),
        format!(
            "9007199254740993{}18446744073709551615{}",
            platform_newline(),
            platform_newline()
        )
    );
    let _ = fs::remove_file(rank_one_path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_accepts_positional_delimiter_and_offsets() {
    let path = temp_path("txt");
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    dlmwrite_builtin(
        Value::from(filename),
        Value::Tensor(tensor),
        vec![
            Value::from(";"),
            Value::Int(IntValue::I32(1)),
            Value::Int(IntValue::I32(2)),
        ],
    )
    .unwrap();

    let contents = fs::read_to_string(&path).unwrap();
    let nl = platform_newline();
    assert_eq!(contents, format!(";;;;{nl};;1;2;3{nl};;4;5;6{nl}"));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_supports_append_and_offsets() {
    let path = temp_path("csv");
    let tensor_a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let tensor_b = Tensor::new(vec![10.0, 13.0, 11.0, 14.0, 12.0, 15.0], vec![3, 2]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    dlmwrite_builtin(
        Value::from(filename.clone()),
        Value::Tensor(tensor_a),
        Vec::new(),
    )
    .unwrap();
    dlmwrite_builtin(
        Value::from(filename),
        Value::Tensor(tensor_b),
        vec![
            Value::from("-append"),
            Value::from("roffset"),
            Value::Int(IntValue::I32(1)),
            Value::from("coffset"),
            Value::Int(IntValue::I32(1)),
        ],
    )
    .unwrap();
    let contents = fs::read_to_string(&path).unwrap();
    let nl = platform_newline();
    assert_eq!(
        contents,
        format!("1,2,3{nl},,{nl},10,14{nl},13,12{nl},11,15{nl}")
    );
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_precision_digits() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![12.34567, std::f64::consts::PI], vec![1, 2]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    dlmwrite_builtin(
        Value::from(filename),
        Value::Tensor(tensor),
        vec![Value::from("precision"), Value::Int(IntValue::I32(3))],
    )
    .unwrap();
    let contents = fs::read_to_string(&path).unwrap();
    let nl = platform_newline();
    assert_eq!(contents, format!("12.3,3.14{nl}"));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_explicit_precision_keeps_numeric_formatting_for_typed_integers() {
    let path = temp_path("csv");
    let tensor = Tensor::new_integer(IntegerStorage::U16(vec![12, 345]), vec![1, 2]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    dlmwrite_builtin(
        Value::from(filename),
        Value::Tensor(tensor),
        vec![Value::from("precision"), Value::Int(IntValue::I32(2))],
    )
    .unwrap();

    let contents = fs::read_to_string(&path).unwrap();
    let nl = platform_newline();
    assert_eq!(contents, format!("12,3.4e+02{nl}"));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_precision_format_string() {
    let path = temp_path("txt");
    let tensor = Tensor::new(vec![0.25, 0.5, 0.75, 1.5], vec![2, 2]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    dlmwrite_builtin(
        Value::from(filename),
        Value::Tensor(tensor),
        vec![
            Value::from("precision"),
            Value::from("%.4f"),
            Value::from("delimiter"),
            Value::from("\t"),
        ],
    )
    .unwrap();
    let contents = fs::read_to_string(&path).unwrap();
    let nl = platform_newline();
    assert_eq!(contents, format!("0.2500\t0.7500{nl}0.5000\t1.5000{nl}"));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_newline_pc() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    dlmwrite_builtin(
        Value::from(filename),
        Value::Tensor(tensor),
        vec![Value::from("newline"), Value::from("pc")],
    )
    .unwrap();
    let contents = fs::read_to_string(&path).unwrap();
    let nl = LineEnding::Pc.as_str();
    assert_eq!(contents, format!("1{nl}2{nl}"));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_coffset_inserts_empty_fields() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    dlmwrite_builtin(
        Value::from(filename),
        Value::Tensor(tensor),
        vec![Value::from("coffset"), Value::Int(IntValue::I32(2))],
    )
    .unwrap();
    let contents = fs::read_to_string(&path).unwrap();
    let nl = platform_newline();
    assert_eq!(contents, format!(",,1,2{nl},,3,4{nl}"));
    let _ = fs::remove_file(path);
}
