use super::StructArray;
use crate::UnboundedOrderedMapBuilder;
use crate::Value;

impl StructArray {
    pub fn select_linear_unbounded(
        &self,
        indices: &[usize],
        shape: Vec<usize>,
    ) -> Result<Value, String> {
        validate_indices(indices, self.len())?;
        let fields = self
            .fields
            .iter()
            .map(|(name, values)| {
                (
                    name.to_owned(),
                    indices.iter().map(|index| values[*index].clone()).collect(),
                )
            })
            .collect::<UnboundedOrderedMapBuilder<_>>();
        Self::normalize_columns(fields, shape)
    }

    pub fn take_linear_unbounded(
        self,
        indices: &[usize],
        shape: Vec<usize>,
    ) -> Result<Value, String> {
        validate_unique_indices(indices, self.len())?;
        let fields = self
            .into_columns_unbounded()
            .map(|(name, values)| {
                let mut source = values.into_iter().map(Some).collect::<Vec<_>>();
                let selected = indices
                    .iter()
                    .map(|index| {
                        source[*index].take().ok_or_else(|| {
                            "structure array move selection contains a duplicate".to_string()
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((name, selected))
            })
            .collect::<Result<UnboundedOrderedMapBuilder<_>, String>>()?;
        Self::normalize_columns(fields, shape)
    }
}

fn validate_indices(indices: &[usize], length: usize) -> Result<(), String> {
    if let Some(index) = indices.iter().find(|index| **index >= length) {
        return Err(format!(
            "structure array index {} is out of bounds",
            index + 1
        ));
    }
    Ok(())
}

fn validate_unique_indices(indices: &[usize], length: usize) -> Result<(), String> {
    validate_indices(indices, length)?;
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("structure array move selection contains a duplicate".into());
    }
    Ok(())
}
