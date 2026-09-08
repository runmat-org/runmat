use super::*;
use runmat_accelerate_api::HostTensorView;
use runmat_value::Tensor;

#[test]
fn gathers_field_values_through_their_owner() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.5], vec![1, 1]).unwrap();
        let materialized = tensor.materialize_f64();
        let handle = provider
            .upload(&HostTensorView {
                data: &materialized,
                shape: &tensor.shape,
            })
            .unwrap();
        let mut structure = StructValue::new();
        structure.insert("angle", Value::GpuTensor(handle));
        let Value::Tensor(output) =
            call(Value::FunctionHandle("sin".into()), structure, Vec::new()).unwrap()
        else {
            panic!("expected tensor")
        };
        assert!((output.materialize_f64()[0] - 0.5f64.sin()).abs() < 1e-12);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn gathers_field_values_from_an_actual_wgpu_provider() {
    use runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider;
    use runmat_accelerate_api::AccelProvider;
    let provider = match ensure_wgpu_provider() {
        Ok(Some(provider)) => provider,
        _ => return,
    };
    let tensor = Tensor::new(vec![0.25], vec![1, 1]).unwrap();
    let materialized = tensor.materialize_f64();
    let handle = provider
        .upload(&HostTensorView {
            data: &materialized,
            shape: &tensor.shape,
        })
        .unwrap();
    let mut structure = StructValue::new();
    structure.insert("angle", Value::GpuTensor(handle));
    let Value::Tensor(output) =
        call(Value::FunctionHandle("sin".into()), structure, Vec::new()).unwrap()
    else {
        panic!("expected tensor")
    };
    assert!((output.materialize_f64()[0] - 0.25f64.sin()).abs() < 1e-12);
}
