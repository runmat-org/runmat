//! Scalar-string predicate; numeric and resident inputs are metadata-only false cases.

use runmat_builtins::{IS_STRING_SCALAR_DESCRIPTOR, IS_STRING_SCALAR_ERROR_TOO_MANY_OUTPUTS};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::BuiltinResult;

#[runtime_builtin(
    name = "isStringScalar",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::is_string_scalar"
)]
pub(crate) fn is_string_scalar_builtin(value: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &IS_STRING_SCALAR_DESCRIPTOR,
        &IS_STRING_SCALAR_ERROR_TOO_MANY_OUTPUTS,
        "isStringScalar",
    )?;
    Ok(Value::Bool(match value {
        Value::String(_) => true,
        Value::StringArray(array) => array.data().len() == 1,
        _ => false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::StringArray;

    #[test]
    fn string_shape_and_output_bound() {
        assert_eq!(
            is_string_scalar_builtin(Value::String("x".into())).unwrap(),
            Value::Bool(true)
        );
        let vector = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
        assert_eq!(
            is_string_scalar_builtin(Value::StringArray(vector)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            is_string_scalar_builtin(Value::Num(1.0)).unwrap(),
            Value::Bool(false)
        );
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error =
            is_string_scalar_builtin(Value::String("x".into())).expect_err("fixed output bound");
        assert_eq!(
            error.identifier(),
            IS_STRING_SCALAR_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }
}
