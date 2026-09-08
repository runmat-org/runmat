use super::*;
use crate::builtins::common::test_support;
use runmat_accelerate_api::HostTensorView;

#[test]
fn gathers_provider_resident_cell_contents() {
    test_support::with_test_provider(|provider| {
        let value = Tensor::new(vec![0.5], vec![1, 1]).unwrap();
        let materialized = value.materialize_f64();
        let handle = provider
            .upload(&HostTensorView {
                data: &materialized,
                shape: &value.shape,
            })
            .unwrap();
        let result = call(
            Value::FunctionHandle("sin".into()),
            vec![cell(vec![Value::GpuTensor(handle)], &[1, 1])],
        )
        .unwrap();
        assert!((tensor_values(result)[0] - 0.5f64.sin()).abs() < 1e-12);
    });
}

#[test]
fn nonuniform_output_preserves_callback_residency() {
    test_support::with_test_provider(|provider| {
        let value = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let materialized = value.materialize_f64();
        let handle = provider
            .upload(&HostTensorView {
                data: &materialized,
                shape: &value.shape,
            })
            .unwrap();
        let callback = Value::FunctionHandle("gpuArray".into());
        let result = call(
            callback,
            vec![
                cell(vec![Value::Num(2.0)], &[1, 1]),
                Value::String("UniformOutput".into()),
                Value::Bool(false),
            ],
        )
        .unwrap();
        let Value::Cell(output) = result else {
            panic!("expected cell")
        };
        assert!(matches!(output.data[0], Value::GpuTensor(_)));
        drop(handle);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn gathers_cell_contents_from_an_actual_wgpu_provider() {
    use runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider;
    use runmat_accelerate_api::AccelProvider;

    let provider = match ensure_wgpu_provider() {
        Ok(Some(provider)) => provider,
        _ => return,
    };
    let value = Tensor::new(vec![0.25], vec![1, 1]).unwrap();
    let materialized = value.materialize_f64();
    let handle = provider
        .upload(&HostTensorView {
            data: &materialized,
            shape: &value.shape,
        })
        .unwrap();
    let result = call(
        Value::FunctionHandle("sin".into()),
        vec![cell(vec![Value::GpuTensor(handle)], &[1, 1])],
    )
    .unwrap();
    assert!((tensor_values(result)[0] - 0.25f64.sin()).abs() < 1e-12);
}
