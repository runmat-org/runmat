use super::*;
pub(crate) fn ensure_indexing_context_classes_registered() {
    static INDEXING_CONTEXT_REGISTRATION: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(INDEXING_CONTEXT_CLASS);
    static LEGACY_INDEXING_CONTEXT_REGISTRATION: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(LEGACY_INDEXING_CONTEXT_CLASS);
    INDEXING_CONTEXT_REGISTRATION.ensure(|| {
        register_indexing_context_class(INDEXING_CONTEXT_CLASS);
    });
    LEGACY_INDEXING_CONTEXT_REGISTRATION.ensure(|| {
        register_indexing_context_class(LEGACY_INDEXING_CONTEXT_CLASS);
    });
}

pub(super) fn register_indexing_context_class(name: runmat_types::StaticClassIdentity) {
    crate::class_registry::register_builtin_class(
        name,
        crate::class_registry::RuntimeClass {
            events: Default::default(),
            name: name.into(),
            parent: None,
            properties: std::collections::HashMap::new(),
            methods: std::collections::HashMap::new(),
        },
    );
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
            if object.class_name().is(INDEXING_CONTEXT_CLASS)
                || object.class_name().is(LEGACY_INDEXING_CONTEXT_CLASS) =>
        {
            match object.property(ENUM_MEMBER_PROPERTY) {
                Some(Value::String(runmat_value::StringElement::Text(member))) => {
                    member.as_str().to_owned()
                }
                Some(Value::CharArray(chars)) => {
                    super::super::text::decode_utf16_with_error(chars.data(), || {
                        num_args_error(
                            &NUM_ARGUMENTS_ERROR_CONTEXT,
                            "indexing context member is not valid UTF-16 text",
                        )
                    })?
                }
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
    let member = indexing_context_member(context);
    let mut value = runmat_value::UnboundedObjectBuilder::new(INDEXING_CONTEXT_CLASS);
    value.insert(
        ENUM_MEMBER_PROPERTY.to_string(),
        Value::string(member.to_string()),
    );
    Value::Object(value.finish())
}

pub(super) fn indexing_context_value_in_context(
    runtime: &crate::context::RuntimeContext,
    context: IndexingContext,
) -> Result<crate::execution::RootedValue, crate::RuntimeError> {
    let member = Value::String(
        crate::call::argument_text::copy_str(indexing_context_member(context), Some(runtime))?
            .into(),
    );
    crate::execution::object::construct_object_from_borrowed_fields(
        Some(runtime),
        INDEXING_CONTEXT_CLASS.display_name(),
        [(ENUM_MEMBER_PROPERTY, &member)].into_iter(),
    )
}

fn indexing_context_member(context: IndexingContext) -> &'static str {
    match context {
        IndexingContext::Statement => INDEXING_CONTEXT_STATEMENT,
        IndexingContext::Expression => INDEXING_CONTEXT_EXPRESSION,
        IndexingContext::Assignment => INDEXING_CONTEXT_ASSIGNMENT,
    }
}
