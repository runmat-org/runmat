use super::*;
use crate::builtins::common::test_support;
use runmat_accelerate_api::{
    HostIntegerDataView, HostIntegerTensorView, HostTensorView, IntegerElementType,
};
use runmat_value::IntegerStorage;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn max_gpu_dim1_matches_cpu() {
    let tensor = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let eval_cpu = evaluate(Value::Tensor(tensor.clone()), &[]).expect("cpu");
    let (values_cpu, indices_cpu) = eval_cpu.into_pair();

    test_support::with_test_provider(|provider| {
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let eval_gpu = evaluate(Value::GpuTensor(handle), &[]).expect("gpu");
        let (values_gpu, indices_gpu) = eval_gpu.into_pair();
        match (&values_gpu, &indices_gpu) {
            (Value::GpuTensor(_), Value::GpuTensor(_)) => {}
            other => panic!("expected GPU tensors, got {other:?}"),
        }
        let gathered_vals = test_support::gather(values_gpu).expect("gather values");
        let gathered_idx = test_support::gather(indices_gpu).expect("gather indices");
        let expected_vals = match values_cpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor values from cpu eval, got {other:?}"),
        };
        let expected_idx = match indices_cpu {
            Value::Tensor(t) => t,
            other => panic!("expected tensor indices from cpu eval, got {other:?}"),
        };
        assert_eq!(gathered_vals.shape(), expected_vals.shape());
        assert_eq!(
            gathered_vals.materialize_f64(),
            expected_vals.materialize_f64()
        );
        assert_eq!(gathered_idx.shape(), expected_idx.shape());
        assert_eq!(
            gathered_idx.materialize_f64(),
            expected_idx.materialize_f64()
        );
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn max_gpu_uint64_reduction_matches_cpu_and_preserves_residency() {
    let data = [u64::MAX, u64::MAX, 9, 9];
    let tensor =
        Tensor::new_integer(IntegerStorage::U64(data.to_vec()), vec![2, 2]).expect("uint64 tensor");
    let (values_cpu, indices_cpu) = evaluate(Value::Tensor(tensor.clone()), &[])
        .expect("cpu max")
        .into_pair();
    assert_eq!(
        values_cpu,
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 9]), vec![1, 2])
                .expect("expected values")
        )
    );
    assert_eq!(
        indices_cpu,
        Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).expect("expected indices"))
    );

    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&data),
                shape: &[2, 2],
            })
            .expect("upload exact uint64");
        let (values_gpu, indices_gpu) = evaluate(Value::GpuTensor(handle), &[])
            .expect("gpu max")
            .into_pair();
        let Value::GpuTensor(values_handle) = &values_gpu else {
            panic!("expected resident uint64 values, got {values_gpu:?}");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(values_handle),
            Some(IntegerElementType::U64)
        );
        assert!(matches!(indices_gpu, Value::GpuTensor(_)));
        assert_eq!(
            test_support::gather(values_gpu).expect("gather values"),
            match values_cpu {
                Value::Tensor(values) => values,
                other => panic!("expected CPU tensor values, got {other:?}"),
            }
        );
        assert_eq!(
            test_support::gather(indices_gpu).expect("gather indices"),
            match indices_cpu {
                Value::Tensor(indices) => indices,
                other => panic!("expected CPU tensor indices, got {other:?}"),
            }
        );
    });
}
