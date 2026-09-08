use super::*;
use crate::builtins::common::test_support;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mat2cell_gpu_falls_back_to_host() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new((1..=6).map(|v| v as f64).collect(), vec![3, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = run(
            Value::GpuTensor(handle),
            vec![row_vector(&[1.0, 2.0]), row_vector(&[1.0, 1.0])],
        )
        .expect("mat2cell");
        let cell = match result {
            Value::Cell(ca) => ca,
            other => panic!("expected cell array, got {other:?}"),
        };
        assert_eq!(cell.shape, vec![2, 2]);
        let block = cell.data[3].clone();
        let gathered = test_support::gather(block).expect("gather");
        assert_eq!(gathered.materialize_f64(), vec![5.0, 6.0]);
        assert_eq!(gathered.shape, vec![2, 1]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn mat2cell_wgpu_matches_cpu_partitions() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );

    let tensor = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![4, 2]).unwrap();
    let cpu_result = run(
        Value::Tensor(tensor.clone()),
        vec![row_vector(&[1.0, 3.0]), row_vector(&[1.0, 1.0])],
    )
    .expect("cpu mat2cell");

    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let handle = runmat_accelerate_api::provider()
        .expect("wgpu provider")
        .upload(&view)
        .expect("upload");
    let gpu_result = run(
        Value::GpuTensor(handle),
        vec![row_vector(&[1.0, 3.0]), row_vector(&[1.0, 1.0])],
    )
    .expect("gpu mat2cell");

    let cpu_cell = match cpu_result {
        Value::Cell(ca) => ca,
        other => panic!("expected cell array, got {other:?}"),
    };
    let gpu_cell = match gpu_result {
        Value::Cell(ca) => ca,
        other => panic!("expected cell array, got {other:?}"),
    };
    assert_eq!(cpu_cell.shape, gpu_cell.shape);
    assert_eq!(cpu_cell.data.len(), gpu_cell.data.len());
    for (cpu, gpu) in cpu_cell.data.iter().zip(gpu_cell.data.iter()) {
        let cpu_val = cpu.clone();
        let gpu_val = gpu.clone();
        let cpu_tensor = test_support::gather(cpu_val).expect("cpu gather");
        let gpu_tensor = test_support::gather(gpu_val).expect("gpu gather");
        assert_eq!(cpu_tensor.shape, gpu_tensor.shape);
        assert_eq!(cpu_tensor.materialize_f64(), gpu_tensor.materialize_f64());
    }
}
