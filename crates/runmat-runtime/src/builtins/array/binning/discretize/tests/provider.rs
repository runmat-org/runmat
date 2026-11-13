use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn resident_integer_inputs_are_gathered_without_losing_precision() {
    test_support::with_test_provider(|provider| {
        let base = 9_007_199_254_740_992_u64;
        let input =
            Tensor::new_integer(IntegerStorage::U64(vec![base + 1, base + 2]), vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
        let edges = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![base, base + 1, base + 2]),
                vec![1, 3],
            )
            .unwrap(),
        );
        let output = tensor(call(Value::GpuTensor(handle), edges, Vec::new()).unwrap());
        assert_eq!(output.materialize_f64(), vec![2.0, 2.0]);
    });
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires an actual compatible WGPU adapter"]
fn actual_wgpu_input_is_gathered_to_the_documented_host_result() {
    let _accel_guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("discretize gather qualification requires an actual WGPU adapter");
    let input = Tensor::new_integer(IntegerStorage::I32(vec![-1, 1]), vec![1, 2]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
    let edges = Value::Tensor(Tensor::new(vec![-2.0, 0.0, 2.0], vec![1, 3]).unwrap());
    let output = tensor(call(Value::GpuTensor(handle.clone()), edges, Vec::new()).unwrap());
    assert_eq!(output.materialize_f64(), vec![1.0, 2.0]);
    handle.release().expect("release WGPU input");
}
