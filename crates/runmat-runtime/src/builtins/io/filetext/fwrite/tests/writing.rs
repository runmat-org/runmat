use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nd_character_payload_flattens_every_page_in_column_first_order() {
    let chars =
        runmat_value::CharArray::new_with_shape("abcdefgh".chars().collect(), vec![2, 2, 2])
            .unwrap();
    let values = flatten_char_array(&chars);
    assert_eq!(
        values,
        "acbdegfh"
            .chars()
            .map(|ch| ch as u32 as f64)
            .collect::<Vec<_>>()
    );
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_double_precision_writes_native_endian() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_double");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w+b"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let tensor = Tensor::new(vec![1.5, -2.25], vec![2, 1]).unwrap();
    let args = vec![Value::from("double")];
    let eval =
        run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
    assert_eq!(eval.count(), 2);

    run_fclose(&[Value::Num(fid as f64)]).unwrap();

    let bytes = test_support::fs::read(&path).expect("read");
    let expected: Vec<u8> = if cfg!(target_endian = "little") {
        [1.5f64.to_le_bytes(), (-2.25f64).to_le_bytes()].concat()
    } else {
        [1.5f64.to_be_bytes(), (-2.25f64).to_be_bytes()].concat()
    };
    assert_eq!(bytes, expected);
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_big_endian_uint16() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_be");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w+b"),
        Value::from("ieee-be"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let tensor = Tensor::new(vec![258.0, 772.0], vec![2, 1]).unwrap();
    let args = vec![Value::from("uint16")];
    let eval =
        run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
    assert_eq!(eval.count(), 2);

    run_fclose(&[Value::Num(fid as f64)]).unwrap();

    let bytes = test_support::fs::read(&path).expect("read");
    assert_eq!(bytes, vec![0x01, 0x02, 0x03, 0x04]);
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_skip_inserts_padding() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_skip");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w+b"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let tensor = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap();
    let args = vec![Value::from("uint8"), Value::Num(1.0)];
    let eval =
        run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).expect("fwrite");
    assert_eq!(eval.count(), 3);

    run_fclose(&[Value::Num(fid as f64)]).unwrap();

    let bytes = test_support::fs::read(&path).expect("read");
    assert_eq!(bytes, vec![10u8, 0, 20, 0, 30]);
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_gpu_tensor_gathers_before_write() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_gpu");

    test_support::with_test_provider(|provider| {
        registry::reset_for_tests();
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let args = vec![Value::from("uint16")];
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = run_evaluate(
                &Value::Num(fid as f64),
                &Value::GpuTensor(handle.clone()),
                &args,
            )
            .expect_err("MATLAB mode rejects direct gpuArray fwrite");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:FwriteGpuInputExtension")
            );
        }
        let eval = {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            run_evaluate(&Value::Num(fid as f64), &Value::GpuTensor(handle), &args)
                .expect("RunMat mode accepts direct gpuArray fwrite")
        };
        assert_eq!(eval.count(), 4);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
    });

    let mut file = File::open(&path).expect("open");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read");
    assert_eq!(bytes.len(), 8);
    let mut decoded = Vec::new();
    for chunk in bytes.chunks_exact(2) {
        let value = if cfg!(target_endian = "little") {
            u16::from_le_bytes([chunk[0], chunk[1]])
        } else {
            u16::from_be_bytes([chunk[0], chunk[1]])
        };
        decoded.push(value);
    }
    assert_eq!(decoded, vec![1u16, 2, 3, 4]);
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_invalid_precision_errors() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_invalid_precision");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w+b"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let args = vec![Value::from("bogus-class")];
    let err = unwrap_error_message(
        run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).unwrap_err(),
    );
    assert!(err.contains("unsupported precision"));
    let _ = run_fclose(&[Value::Num(fid as f64)]);
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_negative_skip_errors() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_negative_skip");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w+b"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let tensor = Tensor::new(vec![10.0], vec![1, 1]).unwrap();
    let args = vec![Value::from("uint8"), Value::Num(-1.0)];
    let err = unwrap_error_message(
        run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &args).unwrap_err(),
    );
    assert!(err.contains("skip value must be non-negative"));
    let _ = run_fclose(&[Value::Num(fid as f64)]);
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn fwrite_wgpu_tensor_roundtrip() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_wgpu_roundtrip");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w+b"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let provider = provider::register_wgpu_provider(provider::WgpuProviderOptions::default())
        .expect("wgpu provider");

    let tensor = Tensor::new(vec![0.5, -1.25, 3.75], vec![3, 1]).unwrap();
    let expected = tensor.materialize_f64().clone();
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let handle = provider.upload(&view).expect("upload to gpu");
    let args = vec![Value::from("double")];
    let eval =
        run_evaluate(&Value::Num(fid as f64), &Value::GpuTensor(handle), &args).expect("fwrite");
    assert_eq!(eval.count(), 3);

    run_fclose(&[Value::Num(fid as f64)]).unwrap();

    let mut file = File::open(&path).expect("open");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).expect("read");
    assert_eq!(bytes.len(), 24);
    for (chunk, expected_value) in bytes.chunks_exact(8).zip(expected.iter()) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        let value = if cfg!(target_endian = "little") {
            f64::from_le_bytes(buf)
        } else {
            f64::from_be_bytes(buf)
        };
        assert!(
            (value - expected_value).abs() < 1e-12,
            "mismatch: {} vs {}",
            value,
            expected_value
        );
    }
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_invalid_identifier_errors() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let err = unwrap_error_message(
        run_evaluate(&Value::Num(-1.0), &Value::Num(1.0), &Vec::new()).unwrap_err(),
    );
    assert!(err.contains("file identifier must be non-negative"));
}
