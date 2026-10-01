use super::*;

pub(super) fn validate_integer_covariance_structure(
    name: &'static str,
    tensor: &Tensor,
) -> BuiltinResult<()> {
    if tensor.shape.len() > 2 || tensor.rows != tensor.cols {
        return Ok(());
    }
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    for col in 0..tensor.cols {
        for row in 0..col {
            if storage.value_at(row + col * tensor.rows)
                != storage.value_at(col + row * tensor.rows)
            {
                return Err(conversion_error(
                    name,
                    format!("{name}: covariance matrix must be symmetric"),
                ));
            }
        }
    }
    Ok(())
}
