use crate::builtins::common::shape::dims_to_row_tensor;
use crate::{BuiltinResult, RuntimeError};
use runmat_builtins::CELLFUN_ERROR_FUNCTION_ERROR;
use runmat_value::Value;

use super::error;

pub(super) fn value(
    context: Option<&crate::context::RuntimeContext>,
    raw_error: &RuntimeError,
    linear_index: usize,
    shape: &[usize],
) -> BuiltinResult<crate::execution::RootedValue> {
    let (identifier, message) = identifier_and_message(raw_error);
    let identifier = Value::String(runmat_value::StringElement::Text(
        crate::call::argument_text::copy_str(identifier, context)?,
    ));
    let message = Value::String(runmat_value::StringElement::Text(
        crate::call::argument_text::copy_str(message, context)?,
    ));
    let indices = dims_to_row_tensor(&linear_indices(linear_index, shape))
        .map_err(|reason| error::internal(format!("cellfun: {reason}")))?;
    let fields = [
        ("identifier", identifier),
        ("message", message),
        ("index", Value::Num((linear_index + 1) as f64)),
        ("indices", Value::Tensor(indices)),
    ];
    crate::execution::construct_struct_from_fields_in_optional_context(
        context,
        fields.iter().map(|(name, value)| (*name, value)),
    )
}

fn identifier_and_message(error: &RuntimeError) -> (&str, &str) {
    if let Some(identifier) = error.identifier() {
        return (identifier, error.message());
    }
    split_message(error.message())
}

fn split_message(raw: &str) -> (&str, &str) {
    let trimmed = raw.trim();
    let mut separators = trimmed.match_indices(':');
    separators.next();
    if let Some((second, _)) = separators.next() {
        let identifier = trimmed[..second].trim();
        let message = trimmed[second + 1..].trim();
        if !identifier.is_empty() && identifier.contains(':') {
            return (
                identifier,
                if message.is_empty() { trimmed } else { message },
            );
        }
    } else if trimmed.len() >= 7
        && (trimmed[..7].eq_ignore_ascii_case("matlab:")
            || trimmed[..7].eq_ignore_ascii_case("runmat:"))
    {
        return (trimmed, "");
    }
    (
        CELLFUN_ERROR_FUNCTION_ERROR
            .identifier
            .expect("cellfun function error descriptor must define an identifier"),
        trimmed,
    )
}

fn linear_indices(mut index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![1];
    }
    shape
        .iter()
        .map(|dimension| {
            if *dimension == 0 {
                1
            } else {
                let coordinate = index % dimension + 1;
                index /= dimension;
                coordinate
            }
        })
        .collect()
}
