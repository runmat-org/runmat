//! Boundary adapter; the reusable projection shape belongs to common JSON.
use super::*;
use crate::builtins::common::json::typed_object_projection::{Projection, PROPERTY};

/// Family-owned process-lifetime field lists. Metadata is optional, never part
/// of the domain's required fields, and every schema retains its own validator.
pub(in crate::builtins::fea) type ProjectionFields = std::sync::OnceLock<Vec<Vec<&'static str>>>;

pub(in crate::builtins::fea) fn register_schemas(
    owner: &'static ProjectionFields,
    schemas: Vec<crate::execution::value_codec::ImmutableValueClassSchema>,
) {
    let fields = owner.get_or_init(|| {
        schemas
            .iter()
            .map(|schema| {
                schema
                    .allowed_fields
                    .iter()
                    .copied()
                    .filter(|field| *field != PROPERTY)
                    .chain(std::iter::once(PROPERTY))
                    .collect()
            })
            .collect()
    });
    assert_eq!(
        fields.len(),
        schemas.len(),
        "FEA projection schema inventory changed"
    );
    for (mut schema, fields) in schemas.into_iter().zip(fields) {
        schema.allowed_fields = fields.as_slice();
        crate::execution::value_codec::register_immutable_value_class(schema)
            .unwrap_or_else(|error| panic!("failed to register projected FEA schema: {error}"));
    }
}

/// FEA-local registration: private constant shape provenance is not a public
/// property and does not grant other host value classes permission to carry it.
pub(in crate::builtins::fea) fn register_value_class(
    name: impl Into<runmat_types::ClassIdentity>,
    mut properties: std::collections::HashMap<
        runmat_types::MemberName,
        crate::class_registry::RuntimeProperty,
    >,
    methods: std::collections::HashMap<
        runmat_types::MethodName,
        crate::class_registry::RuntimeMethod,
    >,
) {
    let (name_field, property) = crate::class_registry::declaration::immutable_property(
        PROPERTY,
        runmat_types::MemberAccess::Private,
    );
    properties.insert(name_field, property);
    crate::class_registry::declaration::register_value_class(name, properties, methods);
}

pub(super) fn to_json(
    builtin: &'static str,
    object: &ObjectInstance,
) -> BuiltinResult<serde_json::Value> {
    to_json_omitting(builtin, object, &[])
}

pub(super) fn to_json_omitting(
    builtin: &'static str,
    object: &ObjectInstance,
    omitted: &[&str],
) -> BuiltinResult<serde_json::Value> {
    let mut fields = serde_json::Map::new();
    for (key, value) in object.properties().iter() {
        if key.starts_with("__runmat_") && key != PROPERTY {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                "opaque runtime property in typed object",
            ));
        }
        if key != PROPERTY && !omitted.contains(&key) {
            fields.insert(key.to_owned(), value_to_json(builtin, value)?);
        }
    }
    let mut json = serde_json::Value::Object(fields);
    if let Some(metadata) = object.property(PROPERTY) {
        let projection = Projection::from_json(value_to_json(builtin, metadata)?)
            .map_err(|message| builtin_error(builtin, &ERROR_INPUT, message))?;
        projection
            .restore_omitting(&mut json, omitted)
            .map_err(|message| builtin_error(builtin, &ERROR_INPUT, message))?;
    }
    Ok(json)
}

pub(super) fn validate(object: &ObjectInstance) -> Result<(), String> {
    if let Some(metadata) = object.property(PROPERTY) {
        let projection = Projection::from_json(
            value_to_json("fea.object", metadata).map_err(|e| e.to_string())?,
        )?;
        // Validate only the provenance branches. Unrelated private host-bound
        // properties (e.g. frozen signal generators) need not be JSON values.
        let mut fields = serde_json::Map::new();
        for key in projection.root_fields()? {
            if let Some(value) = object.property(key) {
                fields.insert(
                    key.to_owned(),
                    value_to_json("fea.object", value).map_err(|e| e.to_string())?,
                );
            }
        }
        projection.restore(&mut serde_json::Value::Object(fields))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
