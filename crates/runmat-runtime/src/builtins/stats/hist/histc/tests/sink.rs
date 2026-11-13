use super::*;

#[test]
fn invalid_histogram_control_precedes_unavailable_provider_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 2],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let input = Value::GpuTensor(allocation.handle.clone());
    let error = block_on(histc_builtin(
        input,
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn histogram_sink_keeps_borrowed_data_and_edges_reusable() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        runmat_accelerate::ensure_residency_hooks();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[0.0, 1.0, 1.0],
                shape: &[1, 3],
            })
            .unwrap();
        let edges = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[0.0, 1.0, 2.0],
                shape: &[1, 3],
            })
            .unwrap();
        for outputs in [0, 1, 2] {
            let _outputs =
                crate::output_count::push_output_count(Some(outputs)).expect("output-count scope");
            for handle in [&input, &edges] {
                runmat_accelerate_api::mark_residency(handle);
            }
            let result = block_on(histc_builtin(
                Value::GpuTensor(input.clone()),
                Value::GpuTensor(edges.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .unwrap();
            let Value::OutputList(values) = result else {
                panic!("requested output list")
            };
            assert_eq!(values.len(), outputs);
            assert!(values.iter().all(|value| matches!(value, Value::Tensor(_))));
            for handle in [&input, &edges] {
                assert!(!runmat_accelerate::fusion_residency::is_resident(handle));
                assert!(block_on(provider.download(handle)).is_ok());
            }
        }
        runmat_accelerate_api::mark_residency(&input);
        assert!(block_on(histc_builtin(
            Value::GpuTensor(input.clone()),
            Value::GpuTensor(edges.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)])
        ))
        .is_err());
        assert!(runmat_accelerate::fusion_residency::is_resident(&input));
        drop(input);
        drop(edges);
    });
}
