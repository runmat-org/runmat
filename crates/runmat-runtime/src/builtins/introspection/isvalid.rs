//! Runtime binding for canonical handle/listener validity query.
use runmat_builtins::ISVALID_ERROR_INVALID_INPUT;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runtime_builtin(
    name = "isvalid",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::isvalid"
)]
pub async fn isvalid_builtin(v: Value) -> crate::BuiltinResult<Value> {
    if !matches!(v, Value::HandleObject(_) | Value::Listener(_)) {
        return Err(
            crate::build_runtime_error(ISVALID_ERROR_INVALID_INPUT.message)
                .with_builtin("isvalid")
                .with_identifier(
                    ISVALID_ERROR_INVALID_INPUT
                        .identifier
                        .expect("isvalid error identifier"),
                )
                .build()
                .into(),
        );
    }
    crate::isvalid_builtin(v).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    #[test]
    fn all_integer_classes_reject_as_invalid_handle_inputs() {
        for value in [
            IntValue::I8(-1),
            IntValue::I16(-2),
            IntValue::I32(-3),
            IntValue::I64(i64::MIN),
            IntValue::U8(1),
            IntValue::U16(2),
            IntValue::U32(3),
            IntValue::U64(u64::MAX),
        ] {
            let error = block_on(isvalid_builtin(Value::Int(value)))
                .expect_err("integer is not a handle input");
            assert_eq!(error.identifier(), ISVALID_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[test]
    fn resident_integer_rejects_without_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("integer tensor");
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("upload integer");
            let error = block_on(isvalid_builtin(Value::GpuTensor(handle)))
                .expect_err("resident integer is not a handle input");
            assert_eq!(error.identifier(), ISVALID_ERROR_INVALID_INPUT.identifier);
        });
    }
}
