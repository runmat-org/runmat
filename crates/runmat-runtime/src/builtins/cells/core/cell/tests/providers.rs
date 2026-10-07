use super::*;

#[test]
fn invalid_cell_option_precedes_unavailable_size_provider_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 2],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let input = Value::GpuTensor(allocation.handle.clone());
    let error = run(vec![input, Value::from("bad-option")]).unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::CELL_ERROR_INVALID_INPUT.identifier
    );
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn runtime_owned_cell_sink_invalidates_only_successful_resident_controls() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        runmat_accelerate::ensure_residency_hooks();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[2.0, 3.0],
                shape: &[1, 2],
            })
            .unwrap();
        for args in [
            vec![Value::GpuTensor(handle.clone())],
            vec![Value::from("like"), Value::GpuTensor(handle.clone())],
        ] {
            runmat_accelerate_api::mark_residency(&handle);
            assert!(runmat_accelerate::fusion_residency::is_resident(&handle));
            assert!(run(args).is_ok());
            assert!(!runmat_accelerate::fusion_residency::is_resident(&handle));
            assert!(block_on(provider.download(&handle)).is_ok());
        }
        runmat_accelerate_api::mark_residency(&handle);
        assert!(run(vec![
            Value::GpuTensor(handle.clone()),
            Value::from("bad-option")
        ])
        .is_err());
        assert!(runmat_accelerate::fusion_residency::is_resident(&handle));
        let physical = handle.allocation_lease().downgrade();
        handle.release().unwrap();
        assert!(physical.acquire().is_none());
    });
}
use crate::builtins::common::test_support;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;
use runmat_value::Tensor;

#[test]
fn deterministic_provider_supplies_size_and_prototype_shape() {
    let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let sizes = Tensor::new(vec![3.0, 2.0], vec![1, 2]).unwrap();
        let size_data = sizes.materialize_f64();
        let size_view = runmat_accelerate_api::HostTensorView {
            data: &size_data,
            shape: sizes.shape(),
        };
        let size_handle = provider.upload(&size_view).unwrap();
        output(vec![Value::GpuTensor(size_handle)], &[3, 2]);

        let prototype = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let prototype_data = prototype.materialize_f64();
        let prototype_view = runmat_accelerate_api::HostTensorView {
            data: &prototype_data,
            shape: prototype.shape(),
        };
        let prototype_handle = provider.upload(&prototype_view).unwrap();
        output(
            vec![Value::from("like"), Value::GpuTensor(prototype_handle)],
            &[2, 1],
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_supplies_size_and_prototype_shape() {
    let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };

    let sizes = Tensor::new(vec![2.0, 3.0, 1.0], vec![1, 3]).unwrap();
    let size_data = sizes.materialize_f64();
    let size_view = runmat_accelerate_api::HostTensorView {
        data: &size_data,
        shape: sizes.shape(),
    };
    output(
        vec![Value::GpuTensor(provider.upload(&size_view).unwrap())],
        &[2, 3],
    );

    let prototype = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
    let prototype_data = prototype.materialize_f64();
    let prototype_view = runmat_accelerate_api::HostTensorView {
        data: &prototype_data,
        shape: prototype.shape(),
    };
    output(
        vec![
            Value::from("like"),
            Value::GpuTensor(provider.upload(&prototype_view).unwrap()),
        ],
        &[2, 3],
    );
}
