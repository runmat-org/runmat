use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};

fn run(value: Value) -> crate::BuiltinResult<Value> {
    block_on(length_builtin(value))
}

#[test]
fn classifies_scalar_vector_matrix_nd_and_empty_shapes() {
    assert_eq!(run(Value::Num(5.0)).unwrap(), Value::Num(1.0));
    assert_eq!(
        run(Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()
        ))
        .unwrap(),
        Value::Num(3.0)
    );
    assert_eq!(
        run(Value::Tensor(
            Tensor::new(vec![0.0; 10], vec![2, 5]).unwrap()
        ))
        .unwrap(),
        Value::Num(5.0)
    );
    assert_eq!(
        run(Value::Tensor(
            Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap()
        ))
        .unwrap(),
        Value::Num(4.0)
    );
    assert_eq!(
        run(Value::Tensor(Tensor::new(vec![], vec![0, 0, 5]).unwrap())).unwrap(),
        Value::Num(5.0)
    );
    assert_eq!(
        run(Value::Tensor(Tensor::new(vec![], vec![0, 7]).unwrap())).unwrap(),
        Value::Num(7.0)
    );
    assert_eq!(
        run(Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap())).unwrap(),
        Value::Num(0.0)
    );
}

#[test]
fn uses_outer_dimensions_for_text_numeric_and_container_storage() {
    assert_eq!(
        run(Value::CharArray(CharArray::new_row("RunMat"))).unwrap(),
        Value::Num(6.0)
    );
    assert_eq!(
        run(Value::StringArray(
            StringArray::new(vec!["a".into(), "bb".into()], vec![2, 1]).unwrap()
        ))
        .unwrap(),
        Value::Num(2.0)
    );
    assert_eq!(
        run(Value::LogicalArray(
            LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap()
        ))
        .unwrap(),
        Value::Num(2.0)
    );
    assert_eq!(
        run(Value::ComplexTensor(
            ComplexTensor::new(vec![(0.0, 0.0); 12], vec![3, 4]).unwrap()
        ))
        .unwrap(),
        Value::Num(4.0)
    );
    let cells = CellArray::new(
        vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ],
        2,
        2,
    )
    .unwrap();
    assert_eq!(run(Value::Cell(cells)).unwrap(), Value::Num(2.0));
}

#[test]
fn rejects_tables_with_catalog_error() {
    let table = crate::builtins::table::table_from_columns(
        vec!["A".into()],
        vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
        )],
    )
    .unwrap();
    let error = run(table).expect_err("length(table)");
    assert_eq!(error.identifier(), Some("RunMat:length:UnsupportedTable"));
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = run(Value::Num(1.0)).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:length:TooManyOutputs"));
}

#[test]
fn resident_shape_requires_no_payload_read() {
    test_support::with_test_provider(|provider| {
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![0.0; 12], vec![3, 4]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            run(Value::GpuTensor(handle.clone())).unwrap(),
            Value::Num(4.0)
        );
        provider.free(&handle).ok();
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_reads_shape_without_payload_transfer() {
    let _state = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default()).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(
        provider,
        &Tensor::new(vec![0.0; 24], vec![6, 4]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        run(Value::GpuTensor(handle.clone())).unwrap(),
        Value::Num(6.0)
    );
    provider.free(&handle).ok();
}
