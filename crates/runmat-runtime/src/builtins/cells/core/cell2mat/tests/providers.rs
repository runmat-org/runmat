use super::*;
use crate::builtins::common::test_support;
use runmat_value::Tensor;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cell2mat_gpu_cells_are_gathered() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let cell = crate::make_cell(vec![Value::GpuTensor(handle.clone())], 1, 1).expect("cell");
        let result = run(cell).expect("cell2mat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), tensor.materialize_f64());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn cell2mat_wgpu_cells_are_gathered() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let handle = provider.upload(&view).expect("upload");
    let cell = crate::make_cell(vec![Value::GpuTensor(handle.clone())], 1, 1).expect("cell");
    let result = run(cell).expect("cell2mat");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.materialize_f64(), tensor.materialize_f64());
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}
