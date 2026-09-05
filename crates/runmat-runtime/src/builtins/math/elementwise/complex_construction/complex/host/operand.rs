use runmat_builtins::COMPLEX_ERROR_INVALID_INPUT;
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::common::tensor;
use crate::BuiltinResult;

use super::super::{error, BUILTIN_NAME};

pub(in crate::builtins::math::elementwise::complex_construction::complex) struct RealInput {
    pub(in crate::builtins::math::elementwise::complex_construction::complex) tensor: Tensor,
    pub(in crate::builtins::math::elementwise::complex_construction::complex) is_scalar_double:
        bool,
}

pub(in crate::builtins::math::elementwise::complex_construction::complex) fn from_value(
    value: Value,
) -> BuiltinResult<RealInput> {
    let is_scalar_double = matches!(value, Value::Num(_))
        || matches!(&value, Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) && tensor.numeric_dtype() == NumericDType::F64);
    match value {
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(error(&COMPLEX_ERROR_INVALID_INPUT, "inputs must be real"))
        }
        Value::String(_) | Value::StringArray(_) => Err(error(
            &COMPLEX_ERROR_INVALID_INPUT,
            "expected numeric input, got string",
        )),
        Value::CharArray(_) => Err(error(
            &COMPLEX_ERROR_INVALID_INPUT,
            "expected numeric input, got char",
        )),
        other => tensor::value_into_tensor_for(BUILTIN_NAME, other)
            .map(|tensor| RealInput {
                tensor,
                is_scalar_double,
            })
            .map_err(|detail| error(&COMPLEX_ERROR_INVALID_INPUT, detail)),
    }
}
