use runmat_value::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct VisibleDimensions(Vec<u64>);

impl VisibleDimensions {
    pub(super) async fn from_value(value: &Value) -> Result<Self, String> {
        visible_dimensions(value).await.map(Self)
    }

    pub(super) fn reported_size(&self) -> &[u64] {
        let rank = effective_rank(&self.0);
        &self.0[..rank.min(self.0.len())]
    }

    pub(super) fn extent(&self, one_based_dimension: u64) -> u64 {
        let Some(index) = one_based_dimension.checked_sub(1) else {
            return 1;
        };
        usize::try_from(index)
            .ok()
            .and_then(|index| self.0.get(index))
            .copied()
            .unwrap_or(1)
    }

    pub(super) fn largest_extent(&self) -> u64 {
        self.0.iter().copied().max().unwrap_or(0)
    }

    pub(super) fn rank(&self) -> usize {
        effective_rank(&self.0)
    }

    pub(super) fn product(&self) -> Option<u64> {
        checked_product(self.0.iter().copied())
    }

    pub(super) fn selected_product(&self, selectors: &[u64]) -> Option<u64> {
        checked_product(selectors.iter().map(|dimension| self.extent(*dimension)))
    }

    pub(super) fn collapsed_outputs(&self, output_count: usize) -> Option<Vec<u64>> {
        if output_count == 0 {
            return Some(Vec::new());
        }
        if output_count == 1 {
            return None;
        }
        let mut outputs = Vec::with_capacity(output_count);
        for index in 0..output_count {
            if index + 1 == output_count && output_count < self.0.len() {
                outputs.push(checked_product(self.0[index..].iter().copied())?);
            } else {
                outputs.push(self.0.get(index).copied().unwrap_or(1));
            }
        }
        Some(outputs)
    }
}

pub(super) async fn visible_dimensions(value: &Value) -> Result<Vec<u64>, String> {
    match value {
        Value::Object(object) if crate::builtins::table::is_tabular_object(object) => Ok(vec![
            u64::try_from(
                crate::builtins::table::table_height(object).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?,
            u64::try_from(
                crate::builtins::table::table_width(object).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?,
        ]),
        Value::Distributed(handle) => {
            handle.validate().map_err(|error| error.to_string())?;
            Ok(normalize_dimensions(&handle.global_shape))
        }
        other => crate::builtins::common::shape::value_dimensions(other)
            .await
            .map(|dimensions| {
                dimensions
                    .into_iter()
                    .map(|dimension| dimension as u64)
                    .collect()
            })
            .map_err(|error| error.message().to_string()),
    }
}

fn checked_product(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values
        .into_iter()
        .try_fold(1_u64, |product, value| product.checked_mul(value))
}

pub(super) fn normalize_dimensions(dimensions: &[u64]) -> Vec<u64> {
    match dimensions {
        [] | [1] | [1, 1] => vec![1, 1],
        [dimension] => vec![1, *dimension],
        _ => dimensions.to_vec(),
    }
}

pub(super) fn effective_rank(dimensions: &[u64]) -> usize {
    dimensions
        .iter()
        .rposition(|dimension| *dimension != 1)
        .map_or(2, |index| (index + 1).max(2))
}

#[cfg(test)]
#[path = "dimension_metadata/tests.rs"]
mod tests;
