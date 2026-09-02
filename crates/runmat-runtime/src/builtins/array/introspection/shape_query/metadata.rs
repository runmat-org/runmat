use runmat_value::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::builtins::array::introspection) struct VisibleDimensions(Vec<u64>);

impl VisibleDimensions {
    pub(in crate::builtins::array::introspection) async fn from_value(
        value: &Value,
    ) -> Result<Self, String> {
        let dimensions = match value {
            Value::Object(object) if crate::builtins::table::is_tabular_object(object) => vec![
                u64::try_from(
                    crate::builtins::table::table_height(object)
                        .map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?,
                u64::try_from(
                    crate::builtins::table::table_width(object)
                        .map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?,
            ],
            _ => super::super::dimension_metadata::visible_dimensions(value).await?,
        };
        Ok(Self(dimensions))
    }

    pub(in crate::builtins::array::introspection) fn reported_size(&self) -> &[u64] {
        let rank = super::super::dimension_metadata::effective_rank(&self.0);
        &self.0[..rank.min(self.0.len())]
    }

    pub(in crate::builtins::array::introspection) fn extent(
        &self,
        one_based_dimension: u64,
    ) -> u64 {
        let Some(index) = one_based_dimension.checked_sub(1) else {
            return 1;
        };
        usize::try_from(index)
            .ok()
            .and_then(|index| self.0.get(index))
            .copied()
            .unwrap_or(1)
    }

    pub(in crate::builtins::array::introspection) fn product(&self) -> Option<u64> {
        checked_product(self.0.iter().copied())
    }

    pub(in crate::builtins::array::introspection) fn selected_product(
        &self,
        selectors: &[u64],
    ) -> Option<u64> {
        checked_product(selectors.iter().map(|dimension| self.extent(*dimension)))
    }

    pub(in crate::builtins::array::introspection) fn collapsed_outputs(
        &self,
        output_count: usize,
    ) -> Option<Vec<u64>> {
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

fn checked_product(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values
        .into_iter()
        .try_fold(1_u64, |product, value| product.checked_mul(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_remaining_dimensions_into_final_output() {
        let dimensions = VisibleDimensions(vec![3, 4, 5]);
        assert_eq!(dimensions.collapsed_outputs(2), Some(vec![3, 20]));
        assert_eq!(dimensions.collapsed_outputs(4), Some(vec![3, 4, 5, 1]));
    }

    #[test]
    fn selected_and_total_products_are_checked() {
        let dimensions = VisibleDimensions(vec![2, 3, 4]);
        assert_eq!(dimensions.product(), Some(24));
        assert_eq!(dimensions.selected_product(&[1, 3, 8]), Some(8));
        assert_eq!(VisibleDimensions(vec![u64::MAX, 2]).product(), None);
    }

    #[test]
    fn reported_size_omits_trailing_singletons_beyond_rank_two() {
        let dimensions = VisibleDimensions(vec![2, 3, 1, 1]);
        assert_eq!(dimensions.reported_size(), &[2, 3]);
        assert_eq!(VisibleDimensions(vec![1, 1, 1]).reported_size(), &[1, 1]);
    }
}
