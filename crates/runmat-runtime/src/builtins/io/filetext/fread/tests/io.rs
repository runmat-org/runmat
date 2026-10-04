use super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_like_gpu_provider_roundtrip() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_like_gpu_provider_roundtrip");
    let mut file = File::create(&path).expect("create");
    file.write_all(&1.5f64.to_le_bytes()).expect("write");
    drop(file);

    test_support::with_test_provider(|provider| {
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("rb"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let view = HostTensorView {
            data: double_values(&proto),
            shape: &proto.shape,
        };
        let handle = provider.upload(&view).expect("upload prototype");

        let args = vec![
            Value::Num(1.0),
            Value::from("double"),
            Value::from("like"),
            Value::GpuTensor(handle),
        ];
        let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
        match eval.data() {
            Value::GpuTensor(result) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(result.clone())).expect("gather");
                assert_eq!(double_values(&gathered), &[1.5]);
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
    });

    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn fread_wgpu_like_uploads_gpu() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_wgpu_like_uploads_gpu");
    let mut file = File::create(&path).expect("create");
    file.write_all(&2.25f64.to_le_bytes()).expect("write");
    drop(file);

    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
    let gpu_proto = run_call_builtin("gpuArray", &[Value::Tensor(proto)]).expect("gpuArray");

    let args = vec![
        Value::Num(1.0),
        Value::from("double"),
        Value::from("like"),
        gpu_proto,
    ];
    let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
    match eval.data() {
        Value::GpuTensor(handle) => {
            let gathered = test_support::gather(Value::GpuTensor(handle.clone())).expect("gather");
            assert_eq!(double_values(&gathered), &[2.25]);
        }
        other => panic!("expected gpu tensor, got {other:?}"),
    }

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_skip_bytes() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_skip_bytes");
    let mut file = File::create(&path).expect("create");
    file.write_all(&[1u8, 2, 3, 4, 5, 6]).expect("write");
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let args = vec![Value::Num(3.0), Value::from("uint8"), Value::Num(1.0)];
    let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
    assert_eq!(eval.count(), 3);
    match eval.data() {
        Value::Tensor(t) => {
            assert_eq!(double_values(t), &[1.0, 3.0, 5.0]);
        }
        other => panic!("unexpected result {other:?}"),
    }

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[test]
fn fread_repeated_precision_skips_after_each_block() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_repeated_skip");
    let mut file = File::create(&path).expect("create");
    file.write_all(&[1_u8, 2, 99, 3, 4, 99]).expect("write");
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;
    let args = vec![Value::Num(4.0), Value::from("2*uint8"), Value::Num(1.0)];
    let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
    let Value::Tensor(output) = eval.data() else {
        panic!("expected tensor output");
    };
    assert_eq!(double_values(output), &[1.0, 2.0, 3.0, 4.0]);

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_big_endian_machine_format() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_big_endian");
    let mut file = File::create(&path).expect("create");
    file.write_all(&[0x01, 0x02, 0x03, 0x04]).expect("write");
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
        Value::from("ieee-be"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let args = vec![Value::Num(2.0), Value::from("uint16")];
    let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
    assert_eq!(eval.count(), 2);
    match eval.data() {
        Value::Tensor(t) => {
            assert_eq!(double_values(t), &[258.0, 772.0]);
            assert_eq!(t.shape, vec![2, 1]);
        }
        other => panic!("unexpected result {other:?}"),
    }

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_public_binding_projects_zero_one_and_two_outputs() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_output_count");
    let mut file = File::create(&path).expect("create");
    file.write_all(&[7, 9]).expect("write");
    drop(file);
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = Value::Num(open.as_open().unwrap().fid as f64);
    {
        let _outputs = crate::output_count::push_output_count(Some(0));
        let result = futures::executor::block_on(fread_builtin(fid.clone(), Vec::new())).unwrap();
        assert!(matches!(result, Value::OutputList(values) if values.is_empty()));
    }
    futures::executor::block_on(crate::builtins::io::filetext::frewind::evaluate(&fid)).unwrap();
    {
        let _outputs = crate::output_count::push_output_count(Some(1));
        let result = futures::executor::block_on(fread_builtin(fid.clone(), Vec::new())).unwrap();
        assert!(matches!(result, Value::OutputList(values) if values.len() == 1));
    }
    futures::executor::block_on(crate::builtins::io::filetext::frewind::evaluate(&fid)).unwrap();
    {
        let _outputs = crate::output_count::push_output_count(Some(2));
        let result = futures::executor::block_on(fread_builtin(fid.clone(), Vec::new())).unwrap();
        let Value::OutputList(values) = result else {
            panic!("two-output fread result")
        };
        assert_eq!(values.len(), 2);
        assert!(matches!(&values[1], Value::Num(2.0)));
    }
    run_fclose(&[fid]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_invalid_fid_errors() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let err = unwrap_error_message(run_evaluate(&Value::Num(9999.0), &Vec::new()).unwrap_err());
    assert_eq!(err, FREAD_ERROR_INVALID_IDENTIFIER.message);
}
