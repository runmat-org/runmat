use super::*;
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
            shape: &sizes.shape,
        };
        let size_handle = provider.upload(&size_view).unwrap();
        output(vec![Value::GpuTensor(size_handle)], &[3, 2]);

        let prototype = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let prototype_data = prototype.materialize_f64();
        let prototype_view = runmat_accelerate_api::HostTensorView {
            data: &prototype_data,
            shape: &prototype.shape,
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
        shape: &sizes.shape,
    };
    output(
        vec![Value::GpuTensor(provider.upload(&size_view).unwrap())],
        &[2, 3],
    );

    let prototype = Tensor::new(vec![1.0; 6], vec![2, 3]).unwrap();
    let prototype_data = prototype.materialize_f64();
    let prototype_view = runmat_accelerate_api::HostTensorView {
        data: &prototype_data,
        shape: &prototype.shape,
    };
    output(
        vec![
            Value::from("like"),
            Value::GpuTensor(provider.upload(&prototype_view).unwrap()),
        ],
        &[2, 3],
    );
}
