use runmat_value::Value;

pub(super) async fn visible_dimensions(value: &Value) -> Result<Vec<u64>, String> {
    match value {
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
mod tests {
    use super::*;

    #[test]
    fn normalizes_scalar_and_vector_shapes() {
        assert_eq!(normalize_dimensions(&[]), vec![1, 1]);
        assert_eq!(normalize_dimensions(&[1]), vec![1, 1]);
        assert_eq!(normalize_dimensions(&[3]), vec![1, 3]);
    }

    #[test]
    fn effective_rank_ignores_only_trailing_singletons() {
        assert_eq!(effective_rank(&[2, 3, 1, 1]), 2);
        assert_eq!(effective_rank(&[1, 1, 3, 1]), 3);
    }
}
