use runmat_java::JavaParameterType;
use runmat_java::JavaValue;
use runmat_value::{
    CharArray, IntValue, IntegerStorage, LogicalArray, NumericStorage, Tensor, Value,
};

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
            IntValue::U64(value) => Ok(i64::try_from(value)
                .map(JavaValue::Long)
                .unwrap_or(JavaValue::UnsignedLong(value))),
        },
        Value::Tensor(value) => tensor_to_java(value),
        Value::LogicalArray(value) => Ok(JavaValue::Array {
            component: JavaParameterType::Boolean,
            elements: value
                .data
                .into_iter()
                .map(|value| JavaValue::Boolean(value != 0))
                .collect(),
        }),
        Value::StringArray(value) => Ok(JavaValue::Array {
            component: JavaParameterType::String,
            elements: value.data.into_iter().map(JavaValue::String).collect(),
        }),
        Value::CharArray(value) => value
            .row_string()
            .map(JavaValue::String)
            .ok_or_else(|| invalid_conversion("Java string conversion requires a character row")),
        Value::Cell(value) => Ok(JavaValue::Array {
            component: JavaParameterType::Object("java.lang.Object".into()),
            elements: value
                .data
                .into_iter()
                .map(value_to_java)
                .collect::<Result<_, _>>()?,
        }),
        other => Err(invalid_conversion(format!(
            "RunMat value {} is not yet convertible to Java",
            value_kind(&other)
        ))),
    }
}

fn tensor_to_java(value: Tensor) -> Result<JavaValue, RuntimeError> {
    if value
        .shape
        .iter()
        .filter(|dimension| **dimension > 1)
        .count()
        > 1
    {
        return Err(invalid_conversion(
            "multidimensional numeric values require an explicit Java array",
        ));
    }
    let (component, elements) = match value.numeric_dtype() {
        runmat_value::NumericDType::F64 => (
            JavaParameterType::Double,
            value
                .as_f64_slice()
                .expect("double tensor has double storage")
                .iter()
                .copied()
                .map(JavaValue::Double)
                .collect(),
        ),
        runmat_value::NumericDType::F32 => (
            JavaParameterType::Float,
            value
                .as_f32_slice()
                .expect("single tensor has single storage")
                .iter()
                .copied()
                .map(JavaValue::Float)
                .collect(),
        ),
        dtype => {
            let storage = value
                .integer_storage()
                .expect("integer tensor has integer storage");
            let component = match dtype {
                runmat_value::NumericDType::I8 => JavaParameterType::Byte,
                runmat_value::NumericDType::I16 => JavaParameterType::Short,
                runmat_value::NumericDType::I32 => JavaParameterType::Int,
                runmat_value::NumericDType::I64 => JavaParameterType::Long,
                runmat_value::NumericDType::U8 => JavaParameterType::Short,
                runmat_value::NumericDType::U16 => JavaParameterType::Int,
                runmat_value::NumericDType::U32 => JavaParameterType::Long,
                runmat_value::NumericDType::U64 => {
                    JavaParameterType::Object("java.math.BigInteger".into())
                }
                runmat_value::NumericDType::F64 | runmat_value::NumericDType::F32 => unreachable!(),
            };
            let elements = storage
                .exact_values()
                .into_iter()
                .map(|value| match value {
                    IntValue::I8(value) => JavaValue::Byte(value),
                    IntValue::I16(value) => JavaValue::Short(value),
                    IntValue::I32(value) => JavaValue::Int(value),
                    IntValue::I64(value) => JavaValue::Long(value),
                    IntValue::U8(value) => JavaValue::Short(i16::from(value)),
                    IntValue::U16(value) => JavaValue::Int(i32::from(value)),
                    IntValue::U32(value) => JavaValue::Long(i64::from(value)),
                    IntValue::U64(value) => JavaValue::UnsignedLong(value),
                })
                .collect();
            (component, elements)
        }
    };
    Ok(JavaValue::Array {
        component,
        elements,
    })
}

pub(super) fn array_from_java(
    component: JavaParameterType,
    elements: Vec<JavaValue>,
) -> Result<Value, RuntimeError> {
    let length = elements.len();
    let shape = vec![1, length];
    match component {
        JavaParameterType::Boolean => LogicalArray::new(
            elements
                .into_iter()
                .map(|value| match value {
                    JavaValue::Boolean(value) => Ok(u8::from(value)),
                    _ => Err(invalid_conversion(
                        "Java boolean array contains an invalid value",
                    )),
                })
                .collect::<Result<_, _>>()?,
            shape,
        )
        .map(Value::LogicalArray)
        .map_err(invalid_conversion),
        JavaParameterType::Byte
        | JavaParameterType::Short
        | JavaParameterType::Int
        | JavaParameterType::Long => {
            let storage = match component {
                JavaParameterType::Byte => IntegerStorage::I8(collect_array(elements, |value| {
                    let JavaValue::Byte(value) = value else {
                        return None;
                    };
                    Some(value)
                })?),
                JavaParameterType::Short => {
                    IntegerStorage::I16(collect_array(elements, |value| {
                        let JavaValue::Short(value) = value else {
                            return None;
                        };
                        Some(value)
                    })?)
                }
                JavaParameterType::Int => IntegerStorage::I32(collect_array(elements, |value| {
                    let JavaValue::Int(value) = value else {
                        return None;
                    };
                    Some(value)
                })?),
                JavaParameterType::Long => IntegerStorage::I64(collect_array(elements, |value| {
                    let JavaValue::Long(value) = value else {
                        return None;
                    };
                    Some(value)
                })?),
                _ => unreachable!(),
            };
            Tensor::new_integer(storage, shape)
                .map(Value::Tensor)
                .map_err(invalid_conversion)
        }
        JavaParameterType::Float => Tensor::from_numeric_storage(
            NumericStorage::F32(collect_array(elements, |value| {
                let JavaValue::Float(value) = value else {
                    return None;
                };
                Some(value)
            })?),
            shape,
        )
        .map(Value::Tensor)
        .map_err(invalid_conversion),
        JavaParameterType::Double => Tensor::new(
            collect_array(elements, |value| {
                let JavaValue::Double(value) = value else {
                    return None;
                };
                Some(value)
            })?,
            shape,
        )
        .map(Value::Tensor)
        .map_err(invalid_conversion),
        JavaParameterType::Char => CharArray::new(
            collect_array(elements, |value| {
                let JavaValue::Char(value) = value else {
                    return None;
                };
                char::from_u32(u32::from(value))
            })?,
            1,
            length,
        )
        .map(Value::CharArray)
        .map_err(invalid_conversion),
        JavaParameterType::String => runmat_value::StringArray::new(
            collect_array(elements, |value| {
                let JavaValue::String(value) = value else {
                    return None;
                };
                Some(value)
            })?,
            shape,
        )
        .map(Value::StringArray)
        .map_err(invalid_conversion),
        JavaParameterType::Object(_) | JavaParameterType::Array(_) => Err(invalid_conversion(
            "Java reference arrays require session-owned conversion",
        )),
    }
}

fn collect_array<T>(
    elements: Vec<JavaValue>,
    convert: impl Fn(JavaValue) -> Option<T>,
) -> Result<Vec<T>, RuntimeError> {
    elements
        .into_iter()
        .map(|value| {
            convert(value).ok_or_else(|| invalid_conversion("Java array element type mismatch"))
        })
        .collect()
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
        JavaValue::UnsignedLong(value) => Ok(Value::Int(IntValue::U64(value))),
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
        assert_eq!(
            value_to_java(Value::Int(IntValue::U64(u64::MAX))).unwrap(),
            JavaValue::UnsignedLong(u64::MAX)
        );
    }

    #[test]
    fn maps_each_numeric_vector_to_an_exact_java_array_component() {
        let cases = [
            (
                IntegerStorage::I8(vec![-8]),
                JavaParameterType::Byte,
                JavaValue::Byte(-8),
            ),
            (
                IntegerStorage::I16(vec![-16]),
                JavaParameterType::Short,
                JavaValue::Short(-16),
            ),
            (
                IntegerStorage::I32(vec![-32]),
                JavaParameterType::Int,
                JavaValue::Int(-32),
            ),
            (
                IntegerStorage::I64(vec![-64]),
                JavaParameterType::Long,
                JavaValue::Long(-64),
            ),
            (
                IntegerStorage::U8(vec![u8::MAX]),
                JavaParameterType::Short,
                JavaValue::Short(i16::from(u8::MAX)),
            ),
            (
                IntegerStorage::U16(vec![u16::MAX]),
                JavaParameterType::Int,
                JavaValue::Int(i32::from(u16::MAX)),
            ),
            (
                IntegerStorage::U32(vec![u32::MAX]),
                JavaParameterType::Long,
                JavaValue::Long(i64::from(u32::MAX)),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX]),
                JavaParameterType::Object("java.math.BigInteger".into()),
                JavaValue::UnsignedLong(u64::MAX),
            ),
        ];

        for (storage, component, element) in cases {
            let value = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
            assert_eq!(
                value_to_java(value).unwrap(),
                JavaValue::Array {
                    component,
                    elements: vec![element],
                }
            );
        }
    }

    #[test]
    fn java_arrays_become_row_vectors_without_numeric_widening() {
        let value = array_from_java(
            JavaParameterType::Long,
            vec![JavaValue::Long(i64::MIN), JavaValue::Long(i64::MAX)],
        )
        .unwrap();
        let Value::Tensor(value) = value else {
            panic!("Java long[] must become a numeric tensor");
        };
        assert_eq!(value.shape, vec![1, 2]);
        assert_eq!(
            value.integer_storage(),
            Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX]))
        );
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
