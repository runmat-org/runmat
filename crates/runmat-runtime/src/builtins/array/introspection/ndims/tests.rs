use super::*;
#[cfg(feature = "wgpu")]
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, IntegerStorage, LogicalArray, StringArray, Tensor, Value,
};

fn run(value: Value) -> crate::BuiltinResult<Value> {
    block_on(ndims_builtin(value))
}

#[test]
fn applies_minimum_rank_and_trailing_singleton_rules() {
    assert_eq!(
        run(Value::Num(std::f64::consts::PI)).unwrap(),
        Value::Num(2.0)
    );
    assert_eq!(
        run(Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()
        ))
        .unwrap(),
        Value::Num(2.0)
    );
    assert_eq!(
        run(Value::Tensor(
            Tensor::new(vec![0.0; 24], vec![2, 3, 4]).unwrap()
        ))
        .unwrap(),
        Value::Num(3.0)
    );
    assert_eq!(
        run(Value::Tensor(
            Tensor::new(vec![0.0; 40], vec![5, 1, 1, 8]).unwrap()
        ))
        .unwrap(),
        Value::Num(4.0)
    );
    assert_eq!(
        run(Value::Tensor(
            Tensor::new(vec![0.0; 6], vec![2, 3, 1, 1]).unwrap()
        ))
        .unwrap(),
        Value::Num(2.0)
    );
}

#[test]
fn every_integer_class_uses_shape_only() {
    let cases = [
        IntegerStorage::I8(vec![1; 6]),
        IntegerStorage::I16(vec![1; 6]),
        IntegerStorage::I32(vec![1; 6]),
        IntegerStorage::I64(vec![1; 6]),
        IntegerStorage::U8(vec![1; 6]),
        IntegerStorage::U16(vec![1; 6]),
        IntegerStorage::U32(vec![1; 6]),
        IntegerStorage::U64(vec![1; 6]),
    ];
    for storage in cases {
        let tensor = Tensor::new_integer(storage, vec![2, 3, 1, 1]).unwrap();
        assert_eq!(run(Value::Tensor(tensor)).unwrap(), Value::Num(2.0));
    }
}

#[test]
fn uses_outer_dimensions_for_text_and_containers() {
    assert_eq!(
        run(Value::CharArray(CharArray::new_row("RunMat"))).unwrap(),
        Value::Num(2.0)
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
            LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap()
        ))
        .unwrap(),
        Value::Num(2.0)
    );
    assert_eq!(
        run(Value::ComplexTensor(
            ComplexTensor::new(vec![(0.0, 0.0); 18], vec![3, 3, 2]).unwrap()
        ))
        .unwrap(),
        Value::Num(3.0)
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
fn empty_resident_shape_is_scalar_metadata_not_missing_metadata() {
    let handle = runmat_accelerate_api::GpuTensorHandle {
        shape: Vec::new(),
        device_id: u32::MAX,
        buffer_id: u64::MAX,
        descriptor: Default::default(),
    };
    assert_eq!(run(Value::GpuTensor(handle)).unwrap(), Value::Num(2.0));
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = run(Value::Num(1.0)).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:ndims:TooManyOutputs"));
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
        &Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX; 64]), vec![4, 4, 4]).unwrap(),
    )
    .unwrap();
    assert_eq!(
        run(Value::GpuTensor(handle.clone())).unwrap(),
        Value::Num(3.0)
    );
    provider.free(&handle).ok();
}
