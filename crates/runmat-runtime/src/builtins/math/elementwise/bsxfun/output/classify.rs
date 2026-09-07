use runmat_builtins::BSXFUN_ERROR_FUNCTION_ERROR;
use runmat_value::{NumericDType, NumericScalar, Value};

#[derive(Debug, PartialEq)]
pub(in crate::builtins::math::elementwise::bsxfun) enum ClassifiedValue {
    Logical(bool),
    Numeric(NumericClassedValue),
    Complex(ComplexClassedValue),
    Character(char),
}

#[derive(Debug, PartialEq)]
pub(in crate::builtins::math::elementwise::bsxfun) struct ComplexClassedValue {
    pub class: NumericDType,
    pub real: NumericScalar,
    pub imaginary: NumericScalar,
}

#[derive(Debug, PartialEq)]
pub(in crate::builtins::math::elementwise::bsxfun) struct NumericClassedValue {
    pub class: NumericDType,
    pub value: NumericScalar,
}

pub(in crate::builtins::math::elementwise::bsxfun) fn value(
    value: &Value,
) -> crate::BuiltinResult<ClassifiedValue> {
    match value {
        Value::Bool(value) => Ok(ClassifiedValue::Logical(*value)),
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok(ClassifiedValue::Logical(array.data[0] != 0))
        }
        Value::Num(value) => Ok(numeric(NumericScalar::F64(*value))),
        Value::Int(value) => Ok(numeric(NumericScalar::from(value.clone()))),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => tensor
            .numeric_value_at(0)
            .map(numeric)
            .ok_or_else(|| super::super::error::internal("missing scalar numeric value")),
        Value::Complex(real, imaginary) => Ok(complex(
            NumericScalar::F64(*real),
            NumericScalar::F64(*imaginary),
        )),
        Value::ComplexTensor(tensor)
            if crate::builtins::common::tensor::is_scalar_complex_tensor(tensor) =>
        {
            let (real, imaginary) = tensor.numeric_value_at(0).ok_or_else(|| {
                super::super::error::internal("missing scalar complex numeric value")
            })?;
            Ok(complex(real, imaginary))
        }
        Value::CharArray(array) if array.rows * array.cols == 1 => Ok(
            ClassifiedValue::Character(array.data.first().copied().unwrap_or('\0')),
        ),
        other => Err(super::super::error::detail(
            &BSXFUN_ERROR_FUNCTION_ERROR,
            Some(format!(
                "callback must return scalar numeric, logical, complex, or character values (got {other:?})"
            )),
        )),
    }
}

fn complex(real: NumericScalar, imaginary: NumericScalar) -> ClassifiedValue {
    debug_assert_eq!(real.numeric_dtype(), imaginary.numeric_dtype());
    ClassifiedValue::Complex(ComplexClassedValue {
        class: real.numeric_dtype(),
        real,
        imaginary,
    })
}

fn numeric(value: NumericScalar) -> ClassifiedValue {
    ClassifiedValue::Numeric(NumericClassedValue {
        class: value.numeric_dtype(),
        value,
    })
}

pub(super) fn inconsistent(expected: &str, actual: &ClassifiedValue) -> crate::RuntimeError {
    super::super::error::detail(
        &BSXFUN_ERROR_FUNCTION_ERROR,
        Some(format!(
            "callback output class must be consistent (expected {expected}, got {})",
            class_name(actual)
        )),
    )
}

fn class_name(value: &ClassifiedValue) -> &'static str {
    match value {
        ClassifiedValue::Logical(_) => "logical",
        ClassifiedValue::Numeric(value) => value.class.class_name(),
        ClassifiedValue::Complex(value) => value.class.class_name(),
        ClassifiedValue::Character(_) => "char",
    }
}
