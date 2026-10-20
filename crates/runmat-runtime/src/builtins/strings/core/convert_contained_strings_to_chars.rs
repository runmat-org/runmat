//! Recursive string-to-character conversion in cells and structures.

use runmat_macros::runtime_builtin;
use runmat_value::{StructValue, Value};

use crate::builtins::strings::core::conversion::{convert_variadic, string_to_chars};
use crate::BuiltinResult;

const NAME: &str = "convertContainedStringsToChars";

#[runtime_builtin(
    name = "convertContainedStringsToChars",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::convert_contained_strings_to_chars"
)]
pub(crate) async fn convert_contained_strings_to_chars_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    convert_variadic(value, rest, convert_value)
}

fn convert_value(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::String(_) | Value::StringArray(_) => string_to_chars(value, NAME),
        Value::Cell(cell) => {
            let values = cell
                .data()
                .iter()
                .cloned()
                .map(convert_value)
                .collect::<BuiltinResult<Vec<_>>>()?;
            crate::make_cell_with_shape(values, cell.shape().to_vec()).map_err(|message| {
                crate::build_runtime_error(message)
                    .with_builtin(NAME)
                    .build()
            })
        }
        Value::Struct(st) => Ok(Value::Struct(StructValue::try_from_fields_unbounded(
            st.into_fields_unbounded()
                .map(|(name, value)| convert_value(value).map(|value| (name, value))),
        )?)),
        Value::StructArray(array) => {
            crate::aggregate::structure::operations::map_identity_preserving(&array, convert_value)
                .map(Value::StructArray)
        }
        other => Ok(other),
    }
}
