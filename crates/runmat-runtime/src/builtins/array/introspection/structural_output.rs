use runmat_value::{Tensor, Value};

#[derive(Debug)]
pub(super) enum StructuralOutputError {
    NotExactlyRepresentable(u64),
    Tensor(String),
}

pub(super) fn exact_double(value: u64) -> Option<f64> {
    if value != 0 {
        let significant_bits = u64::BITS - value.leading_zeros();
        let discarded_bits = significant_bits.saturating_sub(f64::MANTISSA_DIGITS);
        if value.trailing_zeros() < discarded_bits {
            return None;
        }
    }
    Some(value as f64)
}

pub(super) fn row_vector(values: &[u64]) -> Result<Value, StructuralOutputError> {
    let data = values
        .iter()
        .copied()
        .map(|value| {
            exact_double(value).ok_or(StructuralOutputError::NotExactlyRepresentable(value))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let shape = vec![1, data.len()];
    Tensor::new(data, shape)
        .map(crate::builtins::common::tensor::tensor_into_value)
        .map_err(|error| StructuralOutputError::Tensor(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_double_accepts_only_exact_structural_integers() {
        assert_eq!(
            exact_double(9_007_199_254_740_992),
            Some(9_007_199_254_740_992.0)
        );
        assert_eq!(exact_double(9_007_199_254_740_993), None);
        assert_eq!(
            exact_double(9_007_199_254_740_994),
            Some(9_007_199_254_740_994.0)
        );
    }

    #[test]
    fn empty_row_has_one_by_zero_shape() {
        let Value::Tensor(value) = row_vector(&[]).expect("empty row") else {
            panic!("expected tensor");
        };
        assert_eq!(value.shape, vec![1, 0]);
    }
}
