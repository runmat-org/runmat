//! Class admission shared by elementwise arithmetic and page-product preflight.
//! Scalar flags describe the operation's operands (one page for pagefun), not
//! necessarily the full backing array. No payload conversion or provider access.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum OperandClass {
    Integer(IntegerClass),
    RealDouble,
    Other,
}

fn class(value: &Value) -> OperandClass {
    match value {
        Value::Int(value) => OperandClass::Integer(IntegerClass::from_int_value(value)),
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                OperandClass::Integer(IntegerClass::from_storage(storage))
            } else if tensor.numeric_dtype() == runmat_value::NumericDType::F64 {
                OperandClass::RealDouble
            } else {
                OperandClass::Other
            }
        }
        Value::Num(_) => OperandClass::RealDouble,
        Value::GpuTensor(handle)
            if !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real =>
        {
            if let Some(class) = runmat_accelerate_api::handle_integer_class(handle) {
                OperandClass::Integer(class)
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F64)
            {
                OperandClass::RealDouble
            } else {
                OperandClass::Other
            }
        }
        _ => OperandClass::Other,
    }
}

pub(crate) fn validate_integer_binary_inputs(
    left: &Value,
    right: &Value,
    left_scalar: bool,
    right_scalar: bool,
    builtin: &str,
) -> Result<(), String> {
    match (class(left), class(right)) {
        (OperandClass::Integer(left), OperandClass::Integer(right)) if left != right => Err(
            format!("{builtin}: integer operands must have the same integer class"),
        ),
        (OperandClass::Integer(_), OperandClass::Integer(_)) => Ok(()),
        (OperandClass::Integer(_), other) if other != OperandClass::RealDouble || !right_scalar => {
            Err(format!(
                "{builtin}: integer arrays can only be combined with scalar double values"
            ))
        }
        (other, OperandClass::Integer(_)) if other != OperandClass::RealDouble || !left_scalar => {
            Err(format!(
                "{builtin}: integer arrays can only be combined with scalar double values"
            ))
        }
        _ => Ok(()),
    }
}
