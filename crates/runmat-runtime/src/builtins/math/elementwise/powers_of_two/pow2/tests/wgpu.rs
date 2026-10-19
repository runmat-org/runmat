use futures::executor::block_on;
use runmat_value::{Tensor, Value};

use crate::builtins::common::{gpu_helpers, test_support};

use super::super::{binary, unary};

fn provider() -> Option<&'static dyn runmat_accelerate_api::AccelProvider> {
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .ok()
    .map(|provider| provider as &dyn runmat_accelerate_api::AccelProvider)
}

#[test]
fn actual_wgpu_unary_matches_host() {
    let Some(provider) = provider() else { return };
    let input = Tensor::new(vec![-3.5, -1.0, 0.0, 2.0, 4.25], vec![5, 1]).expect("input");
    let expected = unary::transform_real(input.clone()).expect("host");
    let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
    let output = block_on(unary::evaluate(Value::GpuTensor(handle))).expect("device");
    let actual = test_support::gather(output).expect("gather");
    let tolerance = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    for (actual, expected) in actual
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64())
    {
        assert!((actual - expected).abs() <= tolerance);
    }
}

#[test]
fn actual_wgpu_binary_matches_host() {
    let Some(provider) = provider() else { return };
    let left = Tensor::new(vec![0.5, 1.5, 3.0], vec![3, 1]).expect("left");
    let right = Tensor::new(vec![3.0, -2.0, 5.5], vec![3, 1]).expect("right");
    let expected = block_on(binary::evaluate(
        Value::Tensor(left.clone()),
        Value::Tensor(right.clone()),
    ))
    .expect("host");
    let left = gpu_helpers::upload_tensor(provider, &left).expect("upload left");
    let right = gpu_helpers::upload_tensor(provider, &right).expect("upload right");
    let output = block_on(binary::evaluate(
        Value::GpuTensor(left),
        Value::GpuTensor(right),
    ))
    .expect("device");
    let actual = test_support::gather(output).expect("gather");
    let expected = match expected {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    };
    let tolerance = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-3,
    };
    for (actual, expected) in actual
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64())
    {
        assert!((actual - expected).abs() <= tolerance);
    }
}
