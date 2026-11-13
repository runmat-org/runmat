use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

#[path = "tests/core.rs"]
mod core;
#[path = "tests/integer.rs"]
mod integer;
#[path = "tests/labels.rs"]
mod label_cases;
#[path = "tests/provider.rs"]
mod provider;

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape).unwrap())
}

fn classify(sample: Value, training: Value, group: Value, rest: Vec<Value>) -> Vec<Value> {
    let Value::OutputList(values) = block_on(classify_builtin(
        sample,
        training,
        group,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
    .expect("classify") else {
        panic!("expected output list");
    };
    values
        .try_into_owned()
        .expect("unique output list")
        .try_into_unbounded_vec()
        .expect("unbounded output list")
}

fn with_outputs<T>(count: usize, f: impl FnOnce() -> T) -> T {
    let _guard = crate::output_count::push_output_count(Some(count)).expect("output-count scope");
    f()
}

fn valid_floating_inputs() -> (Value, Value, Value) {
    (
        tensor(vec![0.2, 2.2], vec![2, 1]),
        tensor(vec![0.0, 0.5, 2.0, 2.5], vec![4, 1]),
        tensor(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]),
    )
}

fn all_integer_storages(values: &[i8]) -> Vec<IntegerStorage> {
    vec![
        IntegerStorage::I8(values.to_vec()),
        IntegerStorage::I16(values.iter().map(|value| i16::from(*value)).collect()),
        IntegerStorage::I32(values.iter().map(|value| i32::from(*value)).collect()),
        IntegerStorage::I64(values.iter().map(|value| i64::from(*value)).collect()),
        IntegerStorage::U8(values.iter().map(|value| *value as u8).collect()),
        IntegerStorage::U16(values.iter().map(|value| *value as u16).collect()),
        IntegerStorage::U32(values.iter().map(|value| *value as u32).collect()),
        IntegerStorage::U64(values.iter().map(|value| *value as u64).collect()),
    ]
}
