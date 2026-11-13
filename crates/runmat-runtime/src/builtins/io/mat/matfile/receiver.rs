use super::*;

pub(super) fn matfile_object(
    runtime: Option<&crate::context::RuntimeContext>,
    source: &str,
    writable: bool,
    authority: runmat_value::ForeignRef,
) -> BuiltinResult<RootedValue> {
    let fields = [
        (SOURCE_FIELD, Value::string(source)),
        (WRITABLE_FIELD, Value::Bool(writable)),
    ];
    let properties = crate::execution::structure::construct_struct_from_fields_in_optional_context(
        runtime,
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    crate::execution::object::construct_object_from_fields_with_native(
        runtime,
        MATFILE_CLASS.display_name(),
        std::iter::once((PROPERTIES_FIELD, properties.value())),
        Some(crate::execution::object::ObjectNativeInitialization::MatfileAuthority(authority)),
    )
}

pub(super) fn into_matfile_object(value: Value, builtin: &str) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(MATFILE_CLASS) => Ok(object),
        other => Err(matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            format!("{builtin}: receiver must be a {MATFILE_CLASS} object, got {other:?}"),
        )),
    }
}

pub(super) enum PreparedWritableValue {
    Known(bool),
    Resident(runmat_accelerate_api::GpuTensorHandle),
}

pub(super) struct PreparedWritablePair {
    pub(super) value: PreparedWritableValue,
}

pub(super) struct PreparedWritableOptions(pub(super) Vec<PreparedWritablePair>);

pub(super) fn prepare_writable_options(
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<PreparedWritableOptions> {
    if !rest.len().is_multiple_of(2) {
        return Err(matfile_error_with(
            &MATFILE_ERROR_INVALID_OPTION,
            "matfile: name-value arguments must come in pairs",
        ));
    }
    let mut pairs = Vec::with_capacity(rest.len() / 2);
    let mut values = rest.into_iter();
    while let Some(name) = values.next() {
        let value = values.next().expect("validated complete Writable pair");
        let name = value_to_string_scalar(&name)?.ok_or_else(|| {
            matfile_error_with(
                &MATFILE_ERROR_INVALID_OPTION,
                "matfile: option names must be character vectors or string scalars",
            )
        })?;
        if MatFileOptionName::parse(&name) != Some(MatFileOptionName::Writable) {
            return Err(matfile_error_with(
                &MATFILE_ERROR_INVALID_OPTION,
                format!("matfile: unsupported option '{name}'"),
            ));
        }
        let value = match value {
            Value::GpuTensor(handle) => PreparedWritableValue::Resident(handle),
            value => {
                PreparedWritableValue::Known(value_to_bool_scalar(&value).ok_or_else(|| {
                    matfile_error_with(
                        &MATFILE_ERROR_INVALID_OPTION,
                        "matfile: Writable must be a scalar logical or numeric value",
                    )
                })?)
            }
        };
        pairs.push(PreparedWritablePair { value });
    }
    Ok(PreparedWritableOptions(pairs))
}

impl PreparedWritableOptions {
    pub(super) fn identity_controls(&self) -> Vec<MatHandleControl> {
        self.0
            .iter()
            .map(|pair| match &pair.value {
                PreparedWritableValue::Known(value) => MatHandleControl::KnownWritable(*value),
                PreparedWritableValue::Resident(handle) => {
                    MatHandleControl::ResidentWritable(handle.wire_descriptor())
                }
            })
            .collect()
    }

    pub(super) async fn resolve(
        self,
        cooperation: &mut MatCodecCooperation<'_>,
    ) -> BuiltinResult<bool> {
        let mut writable = false;
        for pair in self.0 {
            writable = match pair.value {
                PreparedWritableValue::Known(value) => value,
                PreparedWritableValue::Resident(handle) => {
                    let resident = Value::GpuTensor(handle);
                    let gathered = cooperation
                        .await_io(gather_if_needed_async(&resident))
                        .await??;
                    value_to_bool_scalar(&gathered).ok_or_else(|| {
                        matfile_error_with(
                            &MATFILE_ERROR_INVALID_OPTION,
                            "matfile: Writable must be a scalar logical or numeric value",
                        )
                    })?
                }
            };
        }
        Ok(writable)
    }
}

pub(super) fn normalise_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw = value_to_string_scalar(value)?.ok_or_else(|| {
        matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            "matfile: filename must be a character vector or string scalar",
        )
    })?;
    if raw.trim().is_empty() {
        return Err(matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            "matfile: filename must not be empty",
        ));
    }
    let mut path = PathBuf::from(raw);
    if path.extension().is_none() {
        path.set_extension("mat");
    }
    Ok(path)
}

pub(super) fn member_name(value: &Value, builtin: &str) -> BuiltinResult<String> {
    value_to_string_scalar(value)?.ok_or_else(|| {
        matfile_error_with(
            &MATFILE_ERROR_INVALID_ARGUMENT,
            format!("{builtin}: member name must be a character vector or string scalar"),
        )
    })
}

pub(super) fn reject_internal_field(field: &str) -> BuiltinResult<()> {
    if field.starts_with("__runmat_") {
        return Err(matfile_error_with(
            &MATFILE_ERROR_SELECTION,
            format!("matfile: variable '{field}' was not found"),
        ));
    }
    Ok(())
}

pub(super) fn validate_variable_name(field: &str, builtin: &str) -> BuiltinResult<()> {
    let mut chars = field.chars();
    let Some(first) = chars.next() else {
        return Err(matfile_error_with(
            &MATFILE_ERROR_SELECTION,
            format!("{builtin}: variable name must not be empty"),
        ));
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || chars.any(|ch| !(ch == '_' || ch.is_ascii_alphanumeric()))
    {
        return Err(matfile_error_with(
            &MATFILE_ERROR_SELECTION,
            format!("{builtin}: unsupported MAT-file variable name '{field}'"),
        ));
    }
    Ok(())
}

pub(super) fn replace_or_append(entries: &mut Vec<(String, Value)>, field: String, value: Value) {
    if let Some((_, existing)) = entries.iter_mut().find(|(name, _)| name == &field) {
        *existing = value;
    } else {
        entries.push((field, value));
    }
}

fn value_to_string_scalar(value: &Value) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(Some(s.as_str().to_owned())),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                matfile_error_with(
                    &MATFILE_ERROR_INVALID_ARGUMENT,
                    "matfile: malformed UTF-16 text",
                )
            })
            .map(Some)
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            ))
        }
        _ => Ok(None),
    }
}

fn value_to_bool_scalar(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Num(n) if n.is_finite() => Some(*n != 0.0),
        Value::Int(i) => Some(!i.is_zero()),
        _ => None,
    }
}

pub(in crate::builtins::io::mat) fn matfile_error(message: impl Into<String>) -> RuntimeError {
    matfile_error_with(&MATFILE_ERROR_INVALID_ARGUMENT, message)
}
