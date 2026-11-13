use runmat_builtins::{
    POW2_INTEGER_BINARY_EXPONENT_EXTENSION, POW2_INTEGER_SIGNIFICAND_EXTENSION,
    POW2_INTEGER_UNARY_EXPONENT_EXTENSION,
};
use runmat_value::Value;

use crate::BuiltinResult;

use super::BUILTIN_NAME;

pub(super) async fn validate_unary(exponent: &Value) -> BuiltinResult<()> {
    crate::builtins::common::validation::reject_typed_complex_integer(exponent, BUILTIN_NAME)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        exponent,
        &POW2_INTEGER_UNARY_EXPONENT_EXTENSION,
        BUILTIN_NAME,
        "unary exponent",
    )
    .await
}

pub(super) async fn validate_binary(significand: &Value, exponent: &Value) -> BuiltinResult<()> {
    crate::builtins::common::validation::reject_typed_complex_integer(significand, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(exponent, BUILTIN_NAME)?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        significand,
        &POW2_INTEGER_SIGNIFICAND_EXTENSION,
        BUILTIN_NAME,
        "significand",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        exponent,
        &POW2_INTEGER_BINARY_EXPONENT_EXTENSION,
        BUILTIN_NAME,
        "binary exponent",
    )
    .await
}
