use super::*;
#[derive(Clone)]
pub(super) enum CorrColumn {
    Floating(Vec<f64>),
    Integer(Vec<IntValue>),
}

impl CorrColumn {
    pub(super) fn from_tensor(tensor: &Tensor, rows: usize, col: usize) -> BuiltinResult<Self> {
        let start = col * rows;
        let end = start + rows;
        if let Some(storage) = tensor.integer_storage() {
            let values = (start..end)
                .map(|index| {
                    storage.value_at(index).ok_or_else(|| {
                        corr_internal(format!("corr: integer column index {index} is invalid"))
                    })
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(Self::Integer(values))
        } else {
            let values = tensor::tensor_values_f64_cow(tensor);
            Ok(Self::Floating(values[start..end].to_vec()))
        }
    }

    pub(super) fn is_nan(&self, index: usize) -> bool {
        matches!(self, Self::Floating(values) if values[index].is_nan())
    }

    pub(super) fn compare(&self, left: usize, right: usize) -> Ordering {
        match self {
            Self::Floating(values) => values[left]
                .partial_cmp(&values[right])
                .unwrap_or(Ordering::Equal),
            Self::Integer(values) => {
                exact_int_to_i128(&values[left]).cmp(&exact_int_to_i128(&values[right]))
            }
        }
    }

    pub(super) fn centered_values(&self, indices: &[usize]) -> Vec<f64> {
        match self {
            Self::Floating(values) => indices.iter().map(|index| values[*index]).collect(),
            Self::Integer(values) => {
                let anchor = indices
                    .first()
                    .map(|index| exact_int_to_i128(&values[*index]))
                    .unwrap_or(0);
                indices
                    .iter()
                    .map(|index| (exact_int_to_i128(&values[*index]) - anchor) as f64)
                    .collect()
            }
        }
    }

    pub(super) fn ranks(&self, indices: &[usize], weights: Option<&[f64]>) -> Vec<f64> {
        let mut order = (0..indices.len()).collect::<Vec<_>>();
        order.sort_by(|left, right| self.compare(indices[*left], indices[*right]));
        let mut ranks = vec![0.0; indices.len()];
        let mut start = 0usize;
        let mut cumulative_weight = 0.0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len()
                && self.compare(indices[order[start]], indices[order[end]]) == Ordering::Equal
            {
                end += 1;
            }
            let rank = if let Some(weights) = weights {
                let group_weight = order[start..end]
                    .iter()
                    .map(|position| weights[indices[*position]])
                    .sum::<f64>();
                let group_count = (end - start) as f64;
                let mean_weight = group_weight / group_count;
                let rank = cumulative_weight + (group_count + 1.0) * mean_weight / 2.0;
                cumulative_weight += group_weight;
                rank
            } else {
                (start + 1 + end) as f64 / 2.0
            };
            for position in &order[start..end] {
                ranks[*position] = rank;
            }
            start = end;
        }
        ranks
    }
}

pub(super) fn exact_int_to_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}
