use runmat_value::{ObjectInstance, Value};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::*;
use crate::class_registry::declaration::{
    immutable_properties, register_value_class, static_methods,
};

pub(super) const VALUE_FIELD: &[&str] = &["value"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TypedStudyValue<T> {
    pub(super) value: T,
}

pub(super) fn to_typed_value<T: Serialize>(
    builtin: &'static str,
    class_name: &'static str,
    value: T,
) -> BuiltinResult<Value> {
    serializable_to_object(
        builtin,
        &ERROR_INTERNAL,
        class_name,
        &TypedStudyValue { value },
    )
}

pub(super) fn from_typed_value<T: DeserializeOwned>(
    builtin: &'static str,
    class_name: &'static str,
    value: &Value,
) -> BuiltinResult<T> {
    let wrapped: TypedStudyValue<T> = object_payload(builtin, value, class_name)?;
    Ok(wrapped.value)
}

/// One immutable study-value class: its portable codec schema and its static
/// constructor methods.
pub(super) struct ImmutableStudyClass {
    pub(super) class_name: &'static str,
    pub(super) type_identity: &'static str,
    pub(super) fields: &'static [&'static str],
    pub(super) methods: &'static [(&'static str, &'static str)],
}

/// Register the process-global portable codec schemas once per process.
pub(super) fn register_immutable_schemas(classes: &[ImmutableStudyClass]) {
    for class in classes {
        register_immutable_schema(class.class_name, class.type_identity, class.fields);
    }
}

/// Register the runtime classes in the current class registry. Callers guard
/// this with a `ClassRegistration` so every fresh registry receives them.
pub(super) fn register_immutable_runtime_classes(classes: &[ImmutableStudyClass]) {
    for class in classes {
        register_value_class(
            class.class_name,
            immutable_properties(class.fields),
            static_methods(class.methods.iter().copied()),
        );
    }
}

fn register_immutable_schema(
    class_name: &'static str,
    type_identity: &'static str,
    fields: &'static [&'static str],
) {
    crate::execution::value_codec::register_immutable_value_class(
        crate::execution::value_codec::ImmutableValueClassSchema {
            class_name,
            type_identity,
            schema_version: 1,
            allowed_fields: fields,
            required_fields: fields,
            validate: validate_immutable_value,
        },
    )
    .unwrap_or_else(|error| panic!("failed to register {class_name}: {error}"));
}

pub(super) fn register_immutable_class(
    class_name: &'static str,
    type_identity: &'static str,
    fields: &'static [&'static str],
    methods: &[(&'static str, &'static str)],
) {
    register_immutable_schema(class_name, type_identity, fields);
    register_value_class(
        class_name,
        immutable_properties(fields),
        static_methods(methods.iter().copied()),
    );
}

fn validate_immutable_value(object: &ObjectInstance) -> Result<(), String> {
    if object
        .properties
        .keys()
        .any(|name| name.starts_with("__runmat_"))
    {
        return Err(format!(
            "immutable study value `{}` contains an opaque runtime property",
            object.class_name
        ));
    }
    Ok(())
}

pub(super) fn object_or_cell<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    class_name: &'static str,
) -> BuiltinResult<Vec<T>> {
    match value {
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| object_payload(builtin, item, class_name))
            .collect(),
        Value::Object(_) => Ok(vec![object_payload(builtin, value, class_name)?]),
        other => Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("expected `{class_name}` or a cell array of `{class_name}`; got {other:?}"),
        )),
    }
}

pub(super) fn nonempty_id(
    builtin: &'static str,
    field: &str,
    value: String,
) -> BuiltinResult<String> {
    if value.trim().is_empty() {
        Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            format!("{field} must not be empty"),
        ))
    } else {
        Ok(value)
    }
}
