use runmat_value::{StructArray, StructValue, Value};

pub(super) enum Target {
    Scalar(StructValue),
    Array(StructArray),
}

impl Target {
    pub(super) fn parse(value: Value) -> crate::BuiltinResult<Self> {
        match value {
            Value::Struct(structure) => Ok(Self::Scalar(structure)),
            Value::StructArray(array) => Ok(Self::Array(array)),
            other => Err(super::error::invalid_target(&other)),
        }
    }

    pub(super) fn source_order(&self) -> Vec<String> {
        match self {
            Self::Scalar(structure) => structure.field_names().cloned().collect(),
            Self::Array(array) => array.field_names().cloned().collect(),
        }
    }

    pub(super) fn validate_schema(&self, fields: &[String]) -> crate::BuiltinResult<()> {
        match self {
            Self::Scalar(structure) => validate_fields(structure, fields),
            Self::Array(array) => validate_field_set(array.field_names(), fields),
        }
    }

    pub(super) fn reorder(self, fields: &[String]) -> crate::BuiltinResult<Value> {
        match self {
            Self::Scalar(structure) => reorder_structure(structure, fields).map(Value::Struct),
            Self::Array(mut array) => {
                array
                    .reorder_fields(fields)
                    .map_err(super::error::rebuild)?;
                Ok(Value::StructArray(array))
            }
        }
    }
}

fn validate_fields(structure: &StructValue, fields: &[String]) -> crate::BuiltinResult<()> {
    if structure.fields.len() != fields.len()
        || !fields
            .iter()
            .all(|name| structure.fields.contains_key(name))
    {
        return Err(super::error::field_mismatch());
    }
    Ok(())
}

fn validate_field_set<'a>(
    source: impl Iterator<Item = &'a String>,
    fields: &[String],
) -> crate::BuiltinResult<()> {
    let source = source.collect::<Vec<_>>();
    if source.len() != fields.len() || !fields.iter().all(|name| source.contains(&name)) {
        return Err(super::error::field_mismatch());
    }
    Ok(())
}

fn reorder_structure(
    mut structure: StructValue,
    fields: &[String],
) -> crate::BuiltinResult<StructValue> {
    let mut reordered = StructValue::new();
    for name in fields {
        let value = structure
            .remove(name)
            .ok_or_else(|| super::error::unknown_field(name))?;
        reordered.insert(name.clone(), value);
    }
    Ok(reordered)
}
