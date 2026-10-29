//! Boundary adapter; the reusable projection shape belongs to common JSON.
use super::*;
use crate::builtins::common::json::typed_object_projection::{Projection, PROPERTY};

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
