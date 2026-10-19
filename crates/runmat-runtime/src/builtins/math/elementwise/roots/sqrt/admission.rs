use runmat_builtins::SQRT_INTEGER_INPUT_EXTENSION;
use runmat_value::{SymbolicFunction, Value};

use super::BUILTIN_NAME;
use crate::builtins::math::symbolic::symbolic_function;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;

pub(super) async fn admit(value: &Value) -> BuiltinResult<Option<RootedValue>> {
    let _input = RootedValueLease::for_values(std::iter::once(value))
        .map_err(|cause| cause.into_runtime_error())?;
    if crate::builtins::common::validation::is_typed_complex_integer(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SQRT_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
        crate::builtins::common::validation::reject_typed_complex_integer_for(
            value,
            runmat_builtins::SQRT_CATALOG_ENTRY.identity,
            &runmat_builtins::SQRT_ERROR_INVALID_INPUT,
        )?;
    }
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        value,
        &SQRT_INTEGER_INPUT_EXTENSION,
        BUILTIN_NAME,
        "input",
    )
    .await?;
    symbolic_function(value, SymbolicFunction::Sqrt)
        .map(|symbolic| RootedValue::new(symbolic).map_err(|cause| cause.into_runtime_error()))
        .transpose()
}
