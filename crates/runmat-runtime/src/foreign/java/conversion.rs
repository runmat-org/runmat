use runmat_java::JavaValue;
use runmat_value::{CharArray, IntValue, Tensor, Value};

use super::super::{foreign_error, ForeignErrorKind};
use crate::RuntimeError;

pub(super) fn value_to_java(value: Value) -> Result<JavaValue, RuntimeError> {
    match value {
        Value::Bool(value) => Ok(JavaValue::Boolean(value)),
        Value::String(value) => Ok(JavaValue::String(value)),
        Value::Num(value) => Ok(JavaValue::Double(value)),
        Value::Int(value) => match value {
            IntValue::I8(value) => Ok(JavaValue::Byte(value)),
            IntValue::I16(value) => Ok(JavaValue::Short(value)),
            IntValue::I32(value) => Ok(JavaValue::Int(value)),
            IntValue::I64(value) => Ok(JavaValue::Long(value)),
            IntValue::U8(value) => Ok(JavaValue::Short(i16::from(value))),
            IntValue::U16(value) => Ok(JavaValue::Int(i32::from(value))),
            IntValue::U32(value) => Ok(JavaValue::Long(i64::from(value))),
            IntValue::U64(value) => i64::try_from(value).map(JavaValue::Long).map_err(|_| {
                invalid_conversion("uint64 exceeds Java long; use java.math.BigInteger")
            }),
        },
        other => Err(invalid_conversion(format!(
            "RunMat value {} is not yet convertible to Java",
            value_kind(&other)
        ))),
    }
}

pub(super) fn scalar_from_java(value: JavaValue) -> Result<Value, RuntimeError> {
    match value {
        JavaValue::Null => Tensor::new(Vec::new(), vec![0, 0])
            .map(Value::Tensor)
            .map_err(invalid_conversion),
        JavaValue::Boolean(value) => Ok(Value::Bool(value)),
        JavaValue::Byte(value) => Ok(Value::Int(IntValue::I8(value))),
        JavaValue::Short(value) => Ok(Value::Int(IntValue::I16(value))),
        JavaValue::Int(value) => Ok(Value::Int(IntValue::I32(value))),
        JavaValue::Long(value) => Ok(Value::Int(IntValue::I64(value))),
        JavaValue::Float(value) => Tensor::from_f32(vec![value], vec![1, 1])
            .map(Value::Tensor)
            .map_err(invalid_conversion),
        JavaValue::Double(value) => Ok(Value::Num(value)),
        JavaValue::Char(value) => char::from_u32(u32::from(value))
            .map(|value| Value::CharArray(CharArray::new_row(&value.to_string())))
            .ok_or_else(|| invalid_conversion("Java char is not a valid Unicode scalar")),
        JavaValue::String(value) => Ok(Value::String(value)),
        JavaValue::Array { .. } | JavaValue::Object { .. } => Err(invalid_conversion(
            "Java compound value requires session-owned conversion",
        )),
    }
}

pub(super) fn invalid_conversion(message: impl Into<String>) -> RuntimeError {
    foreign_error(ForeignErrorKind::InvalidCall, message)
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Tensor(_) => "numeric array",
        Value::LogicalArray(_) => "logical array",
        Value::StringArray(_) => "string array",
        Value::CharArray(_) => "character array",
        Value::Cell(_) => "cell array",
        Value::Struct(_) => "structure",
        Value::Foreign(_) => "foreign object",
        _ => "value",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_signed_and_widened_unsigned_integer_values() {
        assert_eq!(
            value_to_java(Value::Int(IntValue::I32(-7))).unwrap(),
            JavaValue::Int(-7)
        );
        assert_eq!(
            value_to_java(Value::Int(IntValue::U32(u32::MAX))).unwrap(),
            JavaValue::Long(i64::from(u32::MAX))
        );
        assert!(value_to_java(Value::Int(IntValue::U64(u64::MAX))).is_err());
    }

    #[test]
    fn does_not_treat_runtime_foreign_handles_as_java_object_handles() {
        let reference = runmat_value::ForeignRef::detached(
            runmat_value::ForeignResourceKey {
                host_identity: "java-process".into(),
                handle: 17,
                generation: 4,
            },
            runmat_types::ForeignTypeIdentity {
                family: runmat_java::JAVA_ADAPTER_ID.into(),
                name: "java.lang.Object".into(),
                version: runmat_java::JAVA_ADAPTER_VERSION,
            },
            runmat_types::ForeignOwnership::Shared,
            runmat_types::ForeignAffinity::OriginProcess,
            runmat_types::ForeignLifetime::Session,
        );
        assert!(value_to_java(Value::Foreign(reference)).is_err());
    }
}
