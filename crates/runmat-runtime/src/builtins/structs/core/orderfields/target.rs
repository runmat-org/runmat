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
            Self::Scalar(structure) => structure.field_names().map(str::to_owned).collect(),
            Self::Array(array) => array.field_names().map(str::to_owned).collect(),
        }
    }

    pub(super) fn validate_schema(&self, fields: &[String]) -> crate::BuiltinResult<()> {
        match self {
            Self::Scalar(structure) => validate_fields(structure, fields),
            Self::Array(array) => validate_field_set(array.field_names(), fields),
        }
    }

    pub(super) fn reorder(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
        fields: &[String],
    ) -> crate::BuiltinResult<crate::execution::RootedValue> {
        match self {
            Self::Scalar(structure) => super::super::construction::scalar(
                runtime,
                fields.iter().map(|name| {
                    (
                        name.as_str(),
                        structure.fields.get(name).expect("validated field order"),
                    )
                }),
            ),
            Self::Array(array) => super::super::construction::array(
                runtime,
                fields.iter().map(|name| {
                    (
                        name.as_str(),
                        array.column_owner(name).expect("validated field order"),
                    )
                }),
                array.shape_owner(),
            ),
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
    source: impl Iterator<Item = &'a str>,
    fields: &[String],
) -> crate::BuiltinResult<()> {
    let source = source.collect::<Vec<_>>();
    if source.len() != fields.len() || !fields.iter().all(|name| source.contains(&name.as_str())) {
        return Err(super::error::field_mismatch());
    }
    Ok(())
}
