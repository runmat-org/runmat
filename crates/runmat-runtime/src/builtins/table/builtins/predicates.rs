use super::*;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "istable",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn istable_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(
        value,
        Value::Object(ref object) if object.is_class(TABLE_CLASS)
    )))
}

#[runtime_builtin(
    name = "istimetable",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn istimetable_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(
        value,
        Value::Object(ref object) if object.is_class(TIMETABLE_CLASS)
    )))
}

#[runtime_builtin(
    name = "iscategorical",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn iscategorical_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(
        value,
        Value::Object(ref object) if object.is_class(CATEGORICAL_CLASS)
    )))
}

#[runtime_builtin(
    name = "isordinal",
    binding_variant = "default",
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn isordinal_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(
        value,
        Value::Object(ref object)
            if object.is_class(CATEGORICAL_CLASS)
                && matches!(object.properties().get("Ordinal"), Some(Value::Bool(true)))
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage};

    #[test]
    fn universal_table_predicates_return_false_for_all_integer_classes() {
        for integer in [
            IntValue::I8(-1),
            IntValue::I16(-2),
            IntValue::I32(-3),
            IntValue::I64(i64::MIN),
            IntValue::U8(1),
            IntValue::U16(2),
            IntValue::U32(3),
            IntValue::U64(u64::MAX),
        ] {
            let value = Value::Int(integer);
            assert_eq!(
                block_on(istable_builtin(value.clone())).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(
                block_on(istimetable_builtin(value.clone())).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(
                block_on(iscategorical_builtin(value.clone())).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(
                block_on(isordinal_builtin(value)).unwrap(),
                Value::Bool(false)
            );
        }
    }

    #[test]
    fn universal_table_predicates_do_not_gather_resident_integer() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("integer tensor");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload integer");
            let value = Value::GpuTensor(handle);
            assert_eq!(
                block_on(istable_builtin(value.clone())).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(
                block_on(istimetable_builtin(value.clone())).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(
                block_on(iscategorical_builtin(value.clone())).unwrap(),
                Value::Bool(false)
            );
            assert_eq!(
                block_on(isordinal_builtin(value)).unwrap(),
                Value::Bool(false)
            );
        });
    }
}
