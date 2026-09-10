use serde_json::Value as JsonValue;

pub(crate) fn rectangular_leaves(values: &[JsonValue]) -> Option<(Vec<usize>, Vec<&JsonValue>)> {
    if values.is_empty() {
        return None;
    }
    if values.iter().all(|value| !value.is_array()) {
        return Some((vec![values.len()], values.iter().collect()));
    }
    if !values.iter().all(JsonValue::is_array) {
        return None;
    }
    let mut shape = None;
    let mut leaves = Vec::new();
    for value in values {
        let (child_shape, mut child_leaves) = rectangular_leaves(value.as_array()?)?;
        match &shape {
            Some(expected) if expected != &child_shape => return None,
            None => shape = Some(child_shape),
            _ => {}
        }
        leaves.append(&mut child_leaves);
    }
    let mut output_shape = vec![values.len()];
    output_shape.extend(shape?);
    Some((output_shape, leaves))
}

pub(crate) fn row_to_column_major<T>(values: Vec<T>, shape: &[usize]) -> Result<Vec<T>, String> {
    let total = shape.iter().try_fold(1usize, |count, extent| {
        count
            .checked_mul(*extent)
            .ok_or_else(|| "JSON array shape exceeds addressable memory".to_string())
    })?;
    if total != values.len() || shape.contains(&0) {
        return Err("JSON array storage does not match its rectangular shape".into());
    }
    let mut row_strides = vec![1usize; shape.len()];
    for dimension in (0..shape.len().saturating_sub(1)).rev() {
        row_strides[dimension] = row_strides[dimension + 1]
            .checked_mul(shape[dimension + 1])
            .ok_or_else(|| "JSON array shape exceeds addressable memory".to_string())?;
    }
    let mut source = values.into_iter().map(Some).collect::<Vec<_>>();
    let mut output = Vec::with_capacity(total);
    for column_index in 0..total {
        let mut remainder = column_index;
        let mut row_index = 0usize;
        for (dimension, extent) in shape.iter().copied().enumerate() {
            let coordinate = remainder % extent;
            remainder /= extent;
            let offset = coordinate
                .checked_mul(row_strides[dimension])
                .ok_or_else(|| "JSON array index exceeds addressable memory".to_string())?;
            row_index = row_index
                .checked_add(offset)
                .ok_or_else(|| "JSON array index exceeds addressable memory".to_string())?;
        }
        output.push(
            source[row_index]
                .take()
                .ok_or_else(|| "JSON row-to-column mapping is not bijective".to_string())?,
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asymmetric_matrix_moves_from_row_to_column_major() {
        assert_eq!(
            row_to_column_major(vec![1, 2, 3, 4], &[2, 2]).unwrap(),
            vec![1, 3, 2, 4]
        );
    }

    #[test]
    fn rejects_shape_overflow_before_mapping() {
        assert!(row_to_column_major(Vec::<u8>::new(), &[usize::MAX, 2]).is_err());
    }
}
