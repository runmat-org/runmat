//! Sparse, data-free provenance for nulls omitted by typed object projections.
//! The shape is an ordinary private Value property, not an alternate data payload.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

// Implementation-private, not a public class API. Existing reflection may
// enumerate private names, but class access rules prohibit reading or writing it.
pub(crate) const PROPERTY: &str = "__runmat_json_projection";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Projection {
    version: u32,
    root: Shape,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum Shape {
    Null,
    Object { fields: BTreeMap<String, Shape> },
    Array { length: usize, elements: Vec<Slot> },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Slot {
    index: usize,
    shape: Shape,
}

impl Projection {
    pub(crate) fn capture(value: &mut Value) -> Option<Self> {
        capture(value).map(|root| Self { version: 1, root })
    }

    pub(crate) fn from_json(value: Value) -> Result<Self, String> {
        let projection: Self = serde_json::from_value(value).map_err(|e| e.to_string())?;
        if projection.version != 1 {
            return Err("unsupported typed object projection version".into());
        }
        let Shape::Object { fields } = &projection.root else {
            return Err("typed object projection root must be an object".into());
        };
        if fields.contains_key(PROPERTY) {
            return Err("typed object projection cannot describe its own metadata".into());
        }
        validate(&projection.root, 0)?;
        Ok(projection)
    }

    pub(crate) fn to_json(&self) -> Result<Value, serde_json::Error> {
        serde_json::to_value(self)
    }

    pub(crate) fn restore(&self, value: &mut Value) -> Result<(), String> {
        restore(&self.root, value)
    }

    pub(crate) fn root_fields(&self) -> Result<impl Iterator<Item = &str>, String> {
        let Shape::Object { fields } = &self.root else {
            return Err("typed object projection root must be an object".into());
        };
        Ok(fields.keys().map(String::as_str))
    }

    pub(crate) fn restore_omitting(
        &self,
        value: &mut Value,
        omitted: &[&str],
    ) -> Result<(), String> {
        let Shape::Object { fields } = &self.root else {
            return Err("typed object projection root must be an object".into());
        };
        let object = value
            .as_object_mut()
            .ok_or("typed object projection root mismatch")?;
        for (key, child) in fields {
            if omitted.contains(&key.as_str()) {
                continue;
            }
            restore_field(key, child, object)?;
        }
        Ok(())
    }
}

fn capture(value: &mut Value) -> Option<Shape> {
    match value {
        Value::Null => Some(Shape::Null),
        Value::Object(fields) => {
            let mut shape = BTreeMap::new();
            fields.retain(|key, value| {
                if let Some(child) = capture(value) {
                    let keep = !matches!(child, Shape::Null);
                    shape.insert(key.clone(), child);
                    keep
                } else {
                    true
                }
            });
            (!shape.is_empty()).then_some(Shape::Object { fields: shape })
        }
        Value::Array(values) => {
            let elements = values
                .iter_mut()
                .enumerate()
                .filter_map(|(index, value)| capture(value).map(|shape| Slot { index, shape }))
                .collect::<Vec<_>>();
            (!elements.is_empty()).then_some(Shape::Array {
                length: values.len(),
                elements,
            })
        }
        _ => None,
    }
}

fn validate(shape: &Shape, depth: usize) -> Result<(), String> {
    if depth > 128 {
        return Err("typed object projection is too deep".into());
    }
    match shape {
        Shape::Null => Ok(()),
        Shape::Object { fields } => {
            if fields.is_empty() {
                return Err("empty typed object projection branch".into());
            }
            fields
                .values()
                .try_for_each(|shape| validate(shape, depth + 1))
        }
        Shape::Array { length, elements } => {
            if elements.is_empty()
                || elements.iter().any(|slot| slot.index >= *length)
                || elements
                    .windows(2)
                    .any(|pair| pair[0].index >= pair[1].index)
            {
                return Err("invalid typed object projection array slots".into());
            }
            elements
                .iter()
                .try_for_each(|slot| validate(&slot.shape, depth + 1))
        }
    }
}

fn restore(shape: &Shape, value: &mut Value) -> Result<(), String> {
    let mismatch = || "typed object projection does not match its properties".to_owned();
    match shape {
        Shape::Null => {
            // Array nulls have the canonical empty-array presentation in Value.
            if !value.is_null() && !value.as_array().is_some_and(Vec::is_empty) {
                return Err(mismatch());
            }
            *value = Value::Null;
        }
        Shape::Object { fields } => {
            let object = value.as_object_mut().ok_or_else(mismatch)?;
            for (key, child) in fields {
                restore_field(key, child, object)?;
            }
        }
        Shape::Array { length, elements } => {
            let array = value.as_array_mut().ok_or_else(mismatch)?;
            if array.len() != *length {
                return Err(mismatch());
            }
            for slot in elements {
                restore(&slot.shape, array.get_mut(slot.index).ok_or_else(mismatch)?)?;
            }
        }
    }
    Ok(())
}

fn restore_field(
    key: &str,
    shape: &Shape,
    object: &mut serde_json::Map<String, Value>,
) -> Result<(), String> {
    if matches!(shape, Shape::Null) {
        if let Some(presentation) = object.get_mut(key) {
            // An explicitly exposed absent choice uses the same canonical
            // empty presentation as an array null. Provenance still permits
            // only null/[], never overwriting a nonempty public property.
            restore(shape, presentation)?;
        } else {
            object.insert(key.to_owned(), Value::Null);
        }
    } else {
        restore(
            shape,
            object
                .get_mut(key)
                .ok_or("typed object projection property is missing")?,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
