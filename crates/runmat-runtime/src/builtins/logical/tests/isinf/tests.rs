use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{ComplexTensor, IntegerStorage, Tensor};

fn run(value: Value) -> Value {
    block_on(isinf_builtin(value)).expect("isinf should succeed")
}

fn logical_bits(value: Value) -> Vec<u8> {
    match value {
        Value::Bool(value) => vec![u8::from(value)],
        Value::LogicalArray(value) => value.data.to_vec(),
        other => panic!("expected a logical result, got {other:?}"),
    }
}

#[test]
fn classifies_scalars_dense_complex_and_integer_storage() {
    assert_eq!(run(Value::Num(f64::INFINITY)), Value::Bool(true));
    assert_eq!(run(Value::Num(f64::NEG_INFINITY)), Value::Bool(true));
    assert_eq!(run(Value::Num(f64::NAN)), Value::Bool(false));

    let dense = Tensor::new(vec![1.0, f64::INFINITY, f64::NAN], vec![1, 3]).expect("tensor");
    assert_eq!(logical_bits(run(Value::Tensor(dense))), vec![0, 1, 0]);

    let complex = ComplexTensor::new(vec![(1.0, 2.0), (0.0, f64::NEG_INFINITY)], vec![1, 2])
        .expect("complex tensor");
    assert_eq!(logical_bits(run(Value::ComplexTensor(complex))), vec![0, 1]);

    let integer = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![1, 2])
        .expect("integer tensor");
    assert_eq!(logical_bits(run(Value::Tensor(integer))), vec![0, 0]);
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error =
        block_on(isinf_builtin(Value::Num(f64::INFINITY))).expect_err("two outputs must fail");
    assert_eq!(error.identifier(), Some("RunMat:isinf:TooManyOutputs"));
}

#[test]
fn provider_matches_host_and_preserves_residency() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, f64::INFINITY, f64::NAN], vec![1, 3]).expect("tensor");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let output = run(Value::GpuTensor(handle));
        assert!(matches!(output, Value::GpuTensor(_)));
        let gathered = test_support::gather(output).expect("gather");
        assert_eq!(
            gathered
                .into_numeric_storage()
                .expect("logical storage")
                .materialize_f64(),
            vec![0.0, 1.0, 0.0]
        );
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_matches_host_and_preserves_residency() {
    let _guard = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("register WGPU provider");
    let provider = runmat_accelerate_api::provider().expect("WGPU provider");
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let tensor = Tensor::new(vec![1.0, f64::INFINITY, f64::NAN], vec![1, 3]).expect("tensor");
    let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
    let output = run(Value::GpuTensor(handle));
    assert!(matches!(output, Value::GpuTensor(_)));
    let gathered = test_support::gather(output).expect("gather");
    assert_eq!(
        gathered
            .into_numeric_storage()
            .expect("logical storage")
            .materialize_f64(),
        vec![0.0, 1.0, 0.0]
    );
}
