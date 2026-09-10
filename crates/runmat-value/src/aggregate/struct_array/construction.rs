use super::{total_len, StructArray, StructValue};
use crate::Value;
use indexmap::IndexMap;
use std::collections::HashSet;

impl StructArray {
    pub fn new(elements: Vec<StructValue>, shape: Vec<usize>) -> Result<Self, String> {
        let field_names = elements
            .first()
            .map(|value| value.field_names().cloned().collect())
            .unwrap_or_default();
        Self::with_fields(field_names, elements, shape)
    }

    pub fn with_fields(
        field_names: Vec<String>,
        elements: Vec<StructValue>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let array = Self::validated_elements(field_names, elements, shape)?;
        if array.len() == 1 {
            return Err("scalar structures must use Value::Struct".into());
        }
        Ok(array)
    }

    pub fn empty(field_names: Vec<String>, shape: Vec<usize>) -> Result<Self, String> {
        Self::with_fields(field_names, Vec::new(), shape)
    }

    pub fn normalize(
        field_names: Vec<String>,
        elements: Vec<StructValue>,
        shape: Vec<usize>,
    ) -> Result<Value, String> {
        let array = Self::validated_elements(field_names, elements, shape)?;
        if array.len() == 1 {
            return array
                .into_elements()
                .pop()
                .map(Value::Struct)
                .ok_or_else(|| "structure scalar is missing its element".to_string());
        }
        Ok(Value::StructArray(array))
    }

    /// Constructs a scalar or array from field-major boundary storage.
    /// Values for each ordered field must be contiguous in linear array order.
    pub fn normalize_field_major(
        field_names: Vec<String>,
        field_values: Vec<Value>,
        shape: Vec<usize>,
    ) -> Result<Value, String> {
        let length = total_len(&shape)?;
        let expected = field_names
            .len()
            .checked_mul(length)
            .ok_or_else(|| "structure array field storage exceeds platform limits".to_string())?;
        if field_values.len() != expected {
            return Err(format!(
                "structure array requires {expected} field values but {} were supplied",
                field_values.len()
            ));
        }
        if field_names.iter().collect::<HashSet<_>>().len() != field_names.len() {
            return Err("structure array field names must be unique".into());
        }
        let mut values = field_values.into_iter();
        let fields = field_names
            .into_iter()
            .map(|name| {
                let column = values.by_ref().take(length).collect::<Vec<_>>();
                (name, column)
            })
            .collect::<IndexMap<_, _>>();
        Self::normalize_columns(fields, shape)
    }

    pub(crate) fn from_columns(
        fields: IndexMap<String, Vec<Value>>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let length = total_len(&shape)?;
        if length == 1 {
            return Err("scalar structures must use Value::Struct".into());
        }
        Self::validate_columns(&fields, length)?;
        Ok(Self { fields, shape })
    }

    pub(crate) fn normalize_columns(
        fields: IndexMap<String, Vec<Value>>,
        shape: Vec<usize>,
    ) -> Result<Value, String> {
        let length = total_len(&shape)?;
        Self::validate_columns(&fields, length)?;
        if length == 1 {
            let fields = fields
                .into_iter()
                .map(|(name, mut values)| {
                    values
                        .pop()
                        .map(|value| (name, value))
                        .ok_or_else(|| "structure scalar is missing a field value".to_string())
                })
                .collect::<Result<IndexMap<_, _>, _>>()?;
            return Ok(Value::Struct(StructValue { fields }));
        }
        Ok(Value::StructArray(Self { fields, shape }))
    }

    fn validated_elements(
        field_names: Vec<String>,
        elements: Vec<StructValue>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let length = total_len(&shape)?;
        if length != elements.len() {
            return Err(format!(
                "structure array shape describes {length} elements but {} were supplied",
                elements.len()
            ));
        }
        let declared_count = field_names.len();
        let unique = field_names.iter().collect::<HashSet<_>>();
        if unique.len() != declared_count {
            return Err("structure array field names must be unique".into());
        }
        if elements
            .iter()
            .any(|element| !element.field_names().eq(field_names.iter()))
        {
            return Err("structure array elements must have one ordered field schema".into());
        }
        let mut fields = field_names
            .into_iter()
            .map(|name| (name, Vec::with_capacity(length)))
            .collect::<IndexMap<_, _>>();
        for mut element in elements {
            for (name, values) in &mut fields {
                let value = element
                    .remove(name)
                    .ok_or_else(|| "structure array element is missing a field".to_string())?;
                values.push(value);
            }
        }
        Ok(Self { fields, shape })
    }

    fn validate_columns(
        fields: &IndexMap<String, Vec<Value>>,
        length: usize,
    ) -> Result<(), String> {
        if fields.values().any(|values| values.len() != length) {
            return Err("structure array fields must contain one value per element".into());
        }
        Ok(())
    }

    pub fn into_elements(self) -> Vec<StructValue> {
        let length = self.len();
        let mut columns = self
            .fields
            .into_iter()
            .map(|(name, values)| (name, values.into_iter()))
            .collect::<Vec<_>>();
        (0..length)
            .map(|_| StructValue {
                fields: columns
                    .iter_mut()
                    .filter_map(|(name, values)| values.next().map(|value| (name.clone(), value)))
                    .collect(),
            })
            .collect()
    }

    pub fn into_elements_and_shape(self) -> (Vec<StructValue>, Vec<usize>) {
        let shape = self.shape.clone();
        (self.into_elements(), shape)
    }
}
