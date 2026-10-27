//! Bounded field/property projections of canonical aggregate owners.
use super::{value_to_json, MAX_OBJECT_FIELDS, MAX_STRUCT_FIELDS};
use runmat_value::{ObjectInstance, StructValue};
use serde_json::{json, Map as JsonMap, Value as JsonValue};

pub(super) fn struct_to_json(st: &StructValue, depth: usize) -> JsonValue {
    let mut fields = JsonMap::new();
    let mut truncated = false;
    for (idx, (name, field_value)) in st.fields.iter().enumerate() {
        if idx >= MAX_STRUCT_FIELDS {
            truncated = true;
            break;
        }
        fields.insert(name.to_string(), value_to_json(field_value, depth));
    }
    json!({
        "kind": "struct",
        "fieldOrder": st.field_names().take(MAX_STRUCT_FIELDS).map(str::to_owned).collect::<Vec<_>>(),
        "fields": fields,
        "totalFields": st.fields.len(),
        "truncated": truncated,
    })
}

pub(super) fn object_to_json(obj: &ObjectInstance, depth: usize) -> JsonValue {
    let mut fields = JsonMap::new();
    let mut truncated = false;
    for (idx, (name, value)) in obj.properties().iter().enumerate() {
        if idx >= MAX_OBJECT_FIELDS {
            truncated = true;
            break;
        }
        fields.insert(name.to_string(), value_to_json(value, depth));
    }
    json!({
        "kind": "object",
        "className": obj.class_name().display_name(),
        "properties": fields,
        "propertyCount": obj.properties().len(),
        "truncated": truncated,
    })
}
