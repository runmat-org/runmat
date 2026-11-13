use runmat_value::Value;

pub(super) fn make_error_struct(
    context: Option<&crate::context::RuntimeContext>,
    raw_error: &str,
    linear_index: usize,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let (identifier, message) = split_error_message(raw_error);
    let fields = [
        (
            "identifier",
            Value::String(runmat_value::StringElement::Text(
                crate::call::argument_text::copy_str(identifier, context)?,
            )),
        ),
        (
            "message",
            Value::String(runmat_value::StringElement::Text(
                crate::call::argument_text::copy_str(message, context)?,
            )),
        ),
        ("index", Value::Num((linear_index + 1) as f64)),
    ];
    crate::execution::construct_struct_from_fields_in_optional_context(
        context,
        fields.iter().map(|(name, value)| (*name, value)),
    )
}

fn split_error_message(raw: &str) -> (&str, &str) {
    let trimmed = raw.trim();
    let mut indices = trimmed.match_indices(':');
    if let Some((_, _)) = indices.next() {
        if let Some((second_idx, _)) = indices.next() {
            let identifier = trimmed[..second_idx].trim();
            let message = trimmed[second_idx + 1..].trim();
            if !identifier.is_empty() && identifier.contains(':') {
                return (
                    identifier,
                    if message.is_empty() { trimmed } else { message },
                );
            }
        } else if trimmed.get(..7).is_some_and(|prefix| {
            prefix.eq_ignore_ascii_case("matlab:") || prefix.eq_ignore_ascii_case("runmat:")
        }) {
            return (trimmed, "");
        }
    }
    ("RunMat:arrayfun:FunctionError", trimmed)
}
