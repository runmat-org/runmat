use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{ComplexTensor, IntegerStorage, NumericDType, Tensor};

#[test]
fn sqrt_integer_gpu_rejects_values_that_round_at_floating_boundary() {
    test_support::with_test_provider(|provider| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = 9_007_199_254_740_993_u64;
        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![0, wide]), vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let error = sqrt_builtin(Value::GpuTensor(handle))
            .expect_err("wide integer must not round at the sqrt boundary");
        assert!(error.message().contains("exactly representable as double"));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![4, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = sqrt_builtin(Value::GpuTensor(handle)).expect("sqrt");
        let gathered = test_support::gather(result).expect("gather");
        let expected: Vec<f64> = tensor.materialize_f64().iter().map(|&v| v.sqrt()).collect();
        assert_eq!(gathered.shape, vec![4, 1]);
        for (gpu, cpu) in gathered.materialize_f64().iter().zip(expected.iter()) {
            assert!((gpu - cpu).abs() < 1e-12);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_gpu_negative_restores_complex_result_to_owner() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![-1.0, 9.0], vec![1, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = sqrt_builtin(Value::GpuTensor(handle)).expect("sqrt");
        assert!(matches!(result, Value::GpuTensor(_)));
        let result = block_on(gpu_helpers::gather_value_async(&result)).expect("gather");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert!(ct.materialize_f64()[0].0.abs() < 1e-12);
                assert!((ct.materialize_f64()[0].1 - 1.0).abs() < 1e-12);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    });
}

#[test]
fn sqrt_complex_gpu_gathers_without_losing_class_or_components() {
    test_support::with_test_provider(|provider| {
        let tensor = ComplexTensor::from_f32(vec![(3.0, 4.0), (-4.0, 0.0)], vec![1, 2])
            .expect("complex single tensor");
        let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload");
        let output = sqrt_builtin(Value::GpuTensor(handle)).expect("sqrt");
        let gathered = block_on(gpu_helpers::gather_value_async(&output)).expect("gather");
        let Value::ComplexTensor(gathered) = gathered else {
            panic!("expected complex tensor");
        };
        assert_eq!(gathered.shape, vec![1, 2]);
        assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
        let values = gathered
            .as_f32_slice()
            .expect("native complex single storage");
        assert_eq!(<(f32, f32)>::from(values[0]), (2.0, 1.0));
        assert_eq!(<(f32, f32)>::from(values[1]), (0.0, 2.0));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn sqrt_wgpu_matches_cpu_elementwise() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![4, 1]).unwrap();
    let cpu = super::super::host::evaluate(Value::Tensor(tensor.clone())).expect("cpu sqrt");
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let handle = runmat_accelerate_api::provider()
        .unwrap()
        .upload(&view)
        .expect("upload");
    let gpu_value = block_on(super::super::provider::evaluate(handle)).expect("gpu sqrt");
    let gathered = test_support::gather(gpu_value).expect("gather");
    match cpu {
        Value::Tensor(ct) => {
            assert_eq!(gathered.shape, ct.shape);
            for (gpu, cpu) in gathered
                .materialize_f64()
                .iter()
                .zip(ct.materialize_f64().iter())
            {
                let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                    runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                    runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
                };
                assert!((gpu - cpu).abs() < tol, "|{gpu} - {cpu}| >= {tol}");
            }
        }
        Value::Num(_) => panic!("expected tensor result from cpu path"),
        other => panic!("unexpected cpu result {other:?}"),
    }
}
