use super::*;
pub(crate) fn ensure_indexing_context_classes_registered() {
    INDEXING_CONTEXT_REGISTERED.get_or_init(|| {
        register_indexing_context_class(INDEXING_CONTEXT_CLASS);
        register_indexing_context_class(LEGACY_INDEXING_CONTEXT_CLASS);
    });
}

pub(super) fn register_indexing_context_class(name: runmat_types::StaticClassIdentity) {
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: name.into(),
        parent: None,
        properties: std::collections::HashMap::new(),
        methods: std::collections::HashMap::new(),
    });
    crate::class_registry::register_class_enumerations(
        &name.owned(),
        [
            INDEXING_CONTEXT_STATEMENT.to_string(),
            INDEXING_CONTEXT_EXPRESSION.to_string(),
            INDEXING_CONTEXT_ASSIGNMENT.to_string(),
        ],
    );
}

pub(super) fn num_args_error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl Into<String>,
) -> crate::RuntimeError {
    crate::runtime_descriptor_error_with_detail(
        NUM_ARGUMENTS_FROM_SUBSCRIPT_METHOD.display_name(),
        descriptor,
        detail.into(),
    )
}

pub(super) fn parse_indexing_context(value: &Value) -> crate::BuiltinResult<IndexingContext> {
    let text = match value {
        Value::Object(object)
            if object.class_name == INDEXING_CONTEXT_CLASS
                || object.class_name == LEGACY_INDEXING_CONTEXT_CLASS =>
        {
            match object.properties.get(ENUM_MEMBER_PROPERTY) {
                Some(Value::String(member)) => member.clone(),
                Some(Value::CharArray(chars)) => chars.data.iter().collect(),
                _ => {
                    return Err(num_args_error(
                        &NUM_ARGUMENTS_ERROR_CONTEXT,
                        "indexing context enum object is missing its member name",
                    ))
                }
            }
        }
        other => String::try_from(other).map_err(|err| {
            num_args_error(
                &NUM_ARGUMENTS_ERROR_CONTEXT,
                format!("indexing context must be an IndexingContext enum or text: {err}"),
            )
        })?,
    };
    match text
        .trim()
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "statement" => Ok(IndexingContext::Statement),
        "expression" => Ok(IndexingContext::Expression),
        "assignment" => Ok(IndexingContext::Assignment),
        _ => Err(num_args_error(
            &NUM_ARGUMENTS_ERROR_CONTEXT,
            format!("unsupported indexing context '{text}'"),
        )),
    }
}

pub(super) fn indexing_context_value(context: IndexingContext) -> Value {
    let member = match context {
        IndexingContext::Statement => INDEXING_CONTEXT_STATEMENT,
        IndexingContext::Expression => INDEXING_CONTEXT_EXPRESSION,
        IndexingContext::Assignment => INDEXING_CONTEXT_ASSIGNMENT,
    };
    let mut value = runmat_value::ObjectInstance::new(INDEXING_CONTEXT_CLASS.to_string());
    value.properties.insert(
        ENUM_MEMBER_PROPERTY.to_string(),
        Value::String(member.to_string()),
    );
    Value::Object(value)
}
