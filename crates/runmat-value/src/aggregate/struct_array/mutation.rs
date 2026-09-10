use super::{column_major_strides, total_len, StructArray, StructValue};
use crate::Value;
use indexmap::IndexMap;
use std::collections::HashSet;

impl StructArray {
    pub fn grow(self, shape: Vec<usize>, fill: &StructValue) -> Result<Self, String> {
        if !fill.field_names().eq(self.fields.keys()) {
            return Err(
                "structure array growth fill must preserve the ordered field schema".into(),
            );
        }
        let new_len = total_len(&shape)?;
        if self
            .shape
            .iter()
            .enumerate()
            .any(|(dimension, extent)| shape.get(dimension).copied().unwrap_or(1) < *extent)
        {
            return Err("structure array growth cannot shrink a dimension".into());
        }
        let old_strides = column_major_strides(&self.shape)?;
        let new_strides = column_major_strides(&shape)?;
        let destinations = (0..self.len())
            .map(|index| remap_growth_index(index, &self.shape, &old_strides, &new_strides))
            .collect::<Result<Vec<_>, _>>()?;
        let fields =
            self.fields
                .into_iter()
                .map(|(name, values)| {
                    let fill_value = fill.fields.get(&name).ok_or_else(|| {
                        "structure array growth fill is missing a field".to_string()
                    })?;
                    let mut output = vec![fill_value.clone(); new_len];
                    for (value, destination) in values.into_iter().zip(&destinations) {
                        output[*destination] = value;
                    }
                    Ok((name, output))
                })
                .collect::<Result<IndexMap<_, _>, String>>()?;
        Self::from_columns(fields, shape)
    }

    pub fn replace_linear_scalar(
        mut self,
        indices: &[usize],
        replacement: StructValue,
    ) -> Result<Self, String> {
        validate_indices(indices.iter().copied(), self.len())?;
        if !replacement.field_names().eq(self.fields.keys()) {
            return Err("structure array assignment must preserve the ordered field schema".into());
        }
        for (name, value) in replacement.fields {
            let values = self
                .fields
                .get_mut(&name)
                .ok_or_else(|| "structure array assignment is missing a field".to_string())?;
            for index in indices {
                values[*index] = value.clone();
            }
        }
        Ok(self)
    }

    pub fn replace_linear_array(
        mut self,
        indices: &[usize],
        replacements: Self,
    ) -> Result<Self, String> {
        if indices.len() != replacements.len() {
            return Err("structure array assignment count does not match selection".into());
        }
        validate_indices(indices.iter().copied(), self.len())?;
        if !replacements.fields.keys().eq(self.fields.keys()) {
            return Err("structure array assignment must preserve the ordered field schema".into());
        }
        for (name, replacement_values) in replacements.fields {
            let values = self
                .fields
                .get_mut(&name)
                .ok_or_else(|| "structure array assignment is missing a field".to_string())?;
            for (index, value) in indices.iter().copied().zip(replacement_values) {
                values[index] = value;
            }
        }
        Ok(self)
    }

    pub fn remove_linear(
        self,
        removal: &HashSet<usize>,
        output_shape: Vec<usize>,
    ) -> Result<Value, String> {
        validate_indices(removal.iter().copied(), self.len())?;
        let fields = self
            .fields
            .into_iter()
            .map(|(name, values)| {
                let values = values
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, value)| (!removal.contains(&index)).then_some(value))
                    .collect();
                (name, values)
            })
            .collect();
        Self::normalize_columns(fields, output_shape)
    }

    pub fn replicate_scalar(structure: StructValue, shape: Vec<usize>) -> Result<Value, String> {
        let length = total_len(&shape)?;
        let fields = structure
            .fields
            .into_iter()
            .map(|(name, value)| (name, vec![value; length]))
            .collect();
        Self::normalize_columns(fields, shape)
    }

    pub fn grow_scalar(
        structure: StructValue,
        shape: Vec<usize>,
        fill: &StructValue,
    ) -> Result<Self, String> {
        if !structure.field_names().eq(fill.field_names()) {
            return Err(
                "structure array growth fill must preserve the ordered field schema".into(),
            );
        }
        let length = total_len(&shape)?;
        if length < 2 {
            return Err("structure array growth must produce a nonscalar array".into());
        }
        let fields = structure
            .fields
            .into_iter()
            .map(|(name, first)| {
                let fill_value = fill
                    .fields
                    .get(&name)
                    .ok_or_else(|| "structure array growth fill is missing a field".to_string())?;
                let mut values = vec![fill_value.clone(); length];
                values[0] = first;
                Ok((name, values))
            })
            .collect::<Result<IndexMap<_, _>, String>>()?;
        Self::from_columns(fields, shape)
    }
}

fn validate_indices(indices: impl IntoIterator<Item = usize>, length: usize) -> Result<(), String> {
    if let Some(index) = indices.into_iter().find(|index| *index >= length) {
        return Err(format!(
            "structure array index {} is out of bounds",
            index + 1
        ));
    }
    Ok(())
}

fn remap_growth_index(
    old_index: usize,
    shape: &[usize],
    old_strides: &[usize],
    new_strides: &[usize],
) -> Result<usize, String> {
    shape
        .iter()
        .enumerate()
        .try_fold(0usize, |index, (dimension, extent)| {
            let coordinate = if *extent == 0 {
                0
            } else {
                old_index / old_strides[dimension] % extent
            };
            if coordinate == 0 && dimension >= new_strides.len() {
                return Ok(index);
            }
            coordinate
                .checked_mul(
                    new_strides
                        .get(dimension)
                        .copied()
                        .ok_or_else(|| "structure array growth cannot shrink rank".to_string())?,
                )
                .and_then(|offset| index.checked_add(offset))
                .ok_or_else(|| "structure array growth exceeds platform limits".to_string())
        })
}
