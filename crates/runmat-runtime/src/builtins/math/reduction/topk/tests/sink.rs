use super::*;
use futures::executor::block_on;

#[test]
fn invalid_topk_control_precedes_unavailable_provider_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 2],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let probe = allocation.probe;
    let input = Value::GpuTensor(allocation.handle);
    for kind in [TopKKind::Max, TopKKind::Min] {
        let error = block_on(evaluate_topk(
            kind,
            input.clone(),
            crate::call::FunctionArgumentView::from_slice(&[Value::Num(-1.0)]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), TOPK_ERROR_INVALID_ARGUMENT.identifier);
    }
    assert_eq!(probe.retirements(), 0);
    drop(input);
    assert_eq!(probe.retirements(), 1);
}

#[test]
fn topk_barrier_preserves_explicit_owner_and_requested_resident_outputs() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        runmat_accelerate::ensure_residency_hooks();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 0, 1]), vec![3, 1]).unwrap(),
        )
        .unwrap();
        let input = input
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let k = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).unwrap(),
        )
        .unwrap();
        for kind in [TopKKind::Max, TopKKind::Min] {
            for requested in [0, 1, 2] {
                let _outputs = crate::output_count::push_output_count(Some(requested))
                    .expect("output-count scope");
                for handle in [&input, &k] {
                    runmat_accelerate_api::mark_residency(handle);
                }
                let evaluation = block_on(evaluate_topk(
                    kind,
                    Value::GpuTensor(input.clone()),
                    crate::call::FunctionArgumentView::from_slice(&[Value::GpuTensor(k.clone())]),
                ))
                .unwrap();
                let values = outputs(evaluation.into_value(kind).unwrap());
                assert_eq!(values.len(), requested);
                for value in values {
                    let Value::GpuTensor(output) = value else {
                        panic!("resident requested output")
                    };
                    assert_eq!(output.device_id(), input.device_id());
                    assert!(!gpu_helpers::same_gpu_handle(&output, &input));
                    assert!(!gpu_helpers::same_gpu_handle(&output, &k));
                    assert!(block_on(provider.download_numeric(&output)).is_ok());
                    output.release().unwrap();
                }
                for (role, handle) in [("input", &input), ("k", &k)] {
                    assert!(
                        !runmat_accelerate::fusion_residency::is_resident(handle),
                        "{kind:?}, requested={requested}, role={role}, device={}, buffer={}",
                        handle.device_id(),
                        handle.buffer_id()
                    );
                    assert!(block_on(provider.download_numeric(handle)).is_ok());
                }
            }
            runmat_accelerate_api::mark_residency(&input);
            assert!(block_on(evaluate_topk(
                kind,
                Value::GpuTensor(input.clone()),
                crate::call::FunctionArgumentView::from_slice(&[Value::Num(-1.0)])
            ))
            .is_err());
            assert!(runmat_accelerate::fusion_residency::is_resident(&input));
        }
        input.release().unwrap();
        k.release().unwrap();
    });
}
