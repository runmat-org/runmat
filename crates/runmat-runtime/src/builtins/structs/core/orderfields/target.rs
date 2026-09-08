use runmat_value::{CellArray, StructValue, Value};

pub(super) struct StructArray {
    elements: Vec<StructValue>,
    shape: Vec<usize>,
}

pub(super) enum Target {
    Scalar(StructValue),
    Array(StructArray),
}

impl Target {
    pub(super) fn parse(value: Value) -> crate::BuiltinResult<Self> {
        match value {
            Value::Struct(structure) => Ok(Self::Scalar(structure)),
            Value::Cell(array) => StructArray::from_cell(array).map(Self::Array),
            other => Err(super::error::invalid_target(&other)),
        }
    }

    pub(super) fn source_order(&self) -> Vec<String> {
        self.representative()
            .map(|structure| structure.field_names().cloned().collect())
            .unwrap_or_default()
    }

    pub(super) fn validate_schema(&self, fields: &[String]) -> crate::BuiltinResult<()> {
        match self {
            Self::Scalar(structure) => validate_fields(structure, fields),
            Self::Array(array) => array
                .elements
                .iter()
                .try_for_each(|structure| validate_fields(structure, fields)),
        }
    }

    pub(super) fn reorder(self, fields: &[String]) -> crate::BuiltinResult<Value> {
        match self {
            Self::Scalar(structure) => reorder_structure(structure, fields).map(Value::Struct),
            Self::Array(array) => array.reorder(fields).map(Value::Cell),
        }
    }

    fn representative(&self) -> Option<&StructValue> {
        match self {
            Self::Scalar(structure) => Some(structure),
            Self::Array(array) => array.elements.first(),
        }
    }
}

impl StructArray {
    fn from_cell(array: CellArray) -> crate::BuiltinResult<Self> {
        let mut elements = Vec::with_capacity(array.data.len());
        for value in array.data {
            let Value::Struct(structure) = value else {
                return Err(super::error::invalid_target_kind());
            };
            elements.push(structure);
        }
        Ok(Self {
            elements,
            shape: array.shape,
        })
    }

    fn reorder(self, fields: &[String]) -> crate::BuiltinResult<CellArray> {
        let values = self
            .elements
            .into_iter()
            .map(|structure| reorder_structure(structure, fields).map(Value::Struct))
            .collect::<crate::BuiltinResult<Vec<_>>>()?;
        CellArray::new_with_shape(values, self.shape).map_err(super::error::rebuild)
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
