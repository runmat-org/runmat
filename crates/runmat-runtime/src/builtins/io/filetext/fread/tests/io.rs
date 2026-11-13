use super::*;

struct FailingBinaryRead;

impl std::io::Read for FailingBinaryRead {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "read access was revoked",
        ))
    }
}

impl std::io::Seek for FailingBinaryRead {
    fn seek(&mut self, _: SeekFrom) -> std::io::Result<u64> {
        Ok(0)
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn binary_read_failure_keeps_operational_source() {
    let mut cooperation =
        crate::builtins::io::filetext::cooperation::FileCooperation::new(None, BUILTIN_NAME);
    let error = futures::executor::block_on(read_numeric_values(
        &mut FailingBinaryRead,
        InputType::UInt8,
        1,
        Some(1),
        0,
        Endianness::Little,
        &mut cooperation,
    ))
    .unwrap_err();
    assert!(
        matches!(std::error::Error::source(&error).and_then(|source| source.downcast_ref::<ReadFailure>()), Some(ReadFailure::Io { source, .. }) if source.kind() == std::io::ErrorKind::PermissionDenied)
    );
    assert!(std::error::Error::source(&error).is_some());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn binary_read_yield_cancellation_stops_before_next_byte() {
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
    let mut file = std::io::Cursor::new(vec![11u8, 22, 33]);
    let mut read = Box::pin(read_numeric_values(
        &mut file,
        InputType::UInt8,
        1,
        None,
        0,
        Endianness::Little,
        &mut cooperation,
    ));
    assert!(matches!(
        read.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    cancellation.cancel(CancellationReason::User);
    let error = futures::executor::block_on(read).expect_err("cancelled read");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(file.position(), 1, "later bytes were not read after yield");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dropped_yielding_read_restores_exact_fid_handle() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::task::{Context, Poll};

    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_yield_drop");
    let mut file = File::create(&path).expect("fixture create");
    file.write_all(&[11, 22, 33]).expect("fixture bytes");
    drop(file);
    let opened = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("open fixture");
    let fid = opened.as_open().expect("open form").fid as i32;
    let original = registry::shared_handle(fid).expect("registered handle");
    let mut operation = registry::acquire_file_in(fid, None).expect("checked-out FID");
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("quantum")),
        CancellationSource::new().token(),
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
    let mut read = Box::pin(read_numeric_values(
        operation.file_mut(),
        InputType::UInt8,
        1,
        None,
        0,
        Endianness::Little,
        &mut cooperation,
    ));
    assert!(matches!(
        read.as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
        Poll::Pending
    ));
    drop(read);
    assert_eq!(operation.file_mut().stream_position().unwrap(), 1);
    drop(operation);
    let restored = registry::shared_handle(fid).expect("restored handle");
    assert!(std::sync::Arc::ptr_eq(&original, &restored));
    assert!(restored.lock().expect("handle slot").is_some());
    run_fclose(&[Value::Num(fid as f64)]).expect("close fixture");
    test_support::fs::remove_file(path).expect("remove fixture");
}
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
            shape: proto.shape(),
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

    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("wgpu provider");

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
        gpu_proto.value().clone(),
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
            assert_eq!(t.shape(), &[2, 1]);
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
        let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
        let result = futures::executor::block_on(fread_builtin(
            None,
            fid.clone(),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        assert!(matches!(result, Value::OutputList(values) if values.is_empty()));
    }
    futures::executor::block_on(crate::builtins::io::filetext::frewind::evaluate(&fid)).unwrap();
    {
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let result = futures::executor::block_on(fread_builtin(
            None,
            fid.clone(),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        assert!(matches!(result, Value::OutputList(values) if values.len() == 1));
    }
    futures::executor::block_on(crate::builtins::io::filetext::frewind::evaluate(&fid)).unwrap();
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let result = futures::executor::block_on(fread_builtin(
            None,
            fid.clone(),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
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
