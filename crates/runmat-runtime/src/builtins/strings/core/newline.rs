//! Newline string scalar construction.

use runmat_builtins::{NEWLINE_DESCRIPTOR, NEWLINE_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "newline",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::newline"
)]
pub(crate) fn newline_builtin() -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &NEWLINE_DESCRIPTOR,
        &NEWLINE_ERROR_TOO_MANY_OUTPUTS,
        "newline",
    )?;
    Ok(Value::string("\n".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_string_scalar_and_rejects_excess_outputs() {
        assert_eq!(newline_builtin().unwrap(), Value::String("\n".into()));
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = newline_builtin().expect_err("fixed output bound");
        assert_eq!(
            error.identifier(),
            NEWLINE_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }
}
