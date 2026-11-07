use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn binary_write_yield_cancellation_stops_before_next_element() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::task::{Context, Poll};

    let cancellation = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("quantum")),
        cancellation.token(),
        || {
            Box::pin(async {
                futures::future::pending::<()>().await;
                Ok(())
            })
        },
    );
    let mut cooperation = crate::builtins::io::filetext::cooperation::FileCooperation::with_budget(
        budget,
        BUILTIN_NAME,
    );
    let mut file = std::io::Cursor::new(Vec::new());
    let mut writing = Box::pin(write_elements(
        &mut file,
        [
            WriteElement::Integer(IntValue::U8(11)),
            WriteElement::Integer(IntValue::U8(22)),
        ],
        WriteSpec {
            input: InputType::UInt8,
        },
        0,
        MachineFormat::Native,
        &mut cooperation,
    ));
    assert!(matches!(
        writing
            .as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    cancellation.cancel(CancellationReason::User);
    let error = futures::executor::block_on(writing).expect_err("cancelled write");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(file.into_inner(), [11], "later element was not written");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nd_character_payload_flattens_every_page_in_column_first_order() {
    let chars = runmat_value::CharArray::from_unicode_scalars_with_shape(
        "abcdefgh".chars().collect(),
        vec![2, 2, 2],
    )
    .unwrap();
    let values: Vec<_> = borrow_elements(&Value::CharArray(chars))
        .expect("character elements")
        .map(|element| element.as_f64())
        .collect();
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
fn borrowed_integer_storage_writes_exact_bytes_for_every_class() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let cases = vec![
        ("int8", IntegerStorage::I8(vec![-7]), vec![249]),
        ("uint8", IntegerStorage::U8(vec![250]), vec![250]),
        (
            "int16",
            IntegerStorage::I16(vec![-258]),
            (-258_i16).to_be_bytes().to_vec(),
        ),
        (
            "uint16",
            IntegerStorage::U16(vec![0x1234]),
            0x1234_u16.to_be_bytes().to_vec(),
        ),
        (
            "int32",
            IntegerStorage::I32(vec![-0x123456]),
            (-0x123456_i32).to_be_bytes().to_vec(),
        ),
        (
            "uint32",
            IntegerStorage::U32(vec![0x89abcdef]),
            0x89abcdef_u32.to_be_bytes().to_vec(),
        ),
        (
            "int64",
            IntegerStorage::I64(vec![i64::MIN + 1]),
            (i64::MIN + 1).to_be_bytes().to_vec(),
        ),
        (
            "uint64",
            IntegerStorage::U64(vec![9_007_199_254_740_993]),
            9_007_199_254_740_993_u64.to_be_bytes().to_vec(),
        ),
        (
            "integer*8",
            IntegerStorage::I64(vec![-4]),
            (-4_i64).to_be_bytes().to_vec(),
        ),
    ];
    for (index, (precision, storage, expected)) in cases.into_iter().enumerate() {
        let path = unique_path(&format!("fwrite_borrowed_{index}"));
        let opened = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
            Value::from("ieee-be"),
        ])
        .expect("fopen");
        let fid = Value::Num(opened.as_open().unwrap().fid as f64);
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("typed storage");
        let result =
            run_evaluate(&fid, &Value::Tensor(tensor), &[Value::from(precision)]).expect("fwrite");
        assert_eq!(result.count(), 1, "{precision}");
        run_fclose(&[fid]).expect("fclose");
        assert_eq!(
            test_support::fs::read(&path).expect("read"),
            expected,
            "{precision}"
        );
        test_support::fs::remove_file(path).expect("remove file");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn borrowed_character_pages_strings_and_logical_values_write_exact_bytes() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let chars = runmat_value::CharArray::from_unicode_scalars_with_shape(
        "abcdefgh".chars().collect(),
        vec![2, 2, 2],
    )
    .unwrap();
    let strings =
        runmat_value::StringArray::new(vec!["ab".into(), "".into(), "c".into()], vec![1, 3])
            .unwrap();
    let logical = runmat_value::LogicalArray::new(vec![0, 1, 1], vec![3, 1]).unwrap();
    for (label, value, expected) in [
        ("pages", Value::CharArray(chars), b"acbdegfh".to_vec()),
        ("strings", Value::StringArray(strings), b"ab\n\nc".to_vec()),
        ("logical", Value::LogicalArray(logical), vec![0, 1, 1]),
    ] {
        let path = unique_path(&format!("fwrite_borrowed_{label}"));
        let opened = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w+b"),
        ])
        .expect("fopen");
        let fid = Value::Num(opened.as_open().unwrap().fid as f64);
        let result = run_evaluate(&fid, &value, &[Value::from("uint8")]).expect("fwrite");
        assert_eq!(result.count(), expected.len(), "{label}");
        run_fclose(&[fid]).expect("fclose");
        assert_eq!(
            test_support::fs::read(&path).expect("read"),
            expected,
            "{label}"
        );
        test_support::fs::remove_file(path).expect("remove file");
    }
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
            shape: tensor.shape(),
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
        shape: tensor.shape(),
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
