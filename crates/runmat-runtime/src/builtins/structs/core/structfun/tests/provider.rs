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
                shape: tensor.shape(),
            })
            .unwrap();
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert("angle", Value::GpuTensor(handle));
        let result = call(
            Value::FunctionHandle("sin".into()),
            structure.finish(),
            Vec::new(),
        )
        .unwrap();
        let Value::Tensor(output) = result.value() else {
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
            shape: tensor.shape(),
        })
        .unwrap();
    let precision = runmat_accelerate_api::handle_precision(&handle).expect("floating upload");
    let retained_input = handle.clone();
    let mut structure = runmat_value::UnboundedStructValueBuilder::new();
    structure.insert("angle", Value::GpuTensor(handle));
    let result = call(
        Value::FunctionHandle("sin".into()),
        structure.finish(),
        Vec::new(),
    )
    .unwrap();
    let Value::Tensor(output) = result.value() else {
        panic!("expected tensor")
    };
    let (expected, tolerance) = match precision {
        runmat_accelerate_api::ProviderPrecision::F32 => {
            (f64::from(0.25f32.sin()), 4.0 * f32::EPSILON as f64)
        }
        runmat_accelerate_api::ProviderPrecision::F64 => (0.25f64.sin(), 1e-12),
    };
    assert_eq!(output.shape(), vec![1, 1]);
    let values = output.materialize_f64();
    assert!(values[0] > 0.0);
    assert!((values[0] - expected).abs() <= tolerance);
    let input = crate::builtins::common::test_support::gather(Value::GpuTensor(retained_input))
        .expect("borrowed input remains readable");
    assert_eq!(input.materialize_f64(), vec![0.25]);
}
