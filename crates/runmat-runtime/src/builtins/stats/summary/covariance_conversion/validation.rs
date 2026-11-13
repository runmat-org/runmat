use super::*;

pub(super) fn validate_integer_covariance_structure(
    name: &'static str,
    tensor: &Tensor,
) -> BuiltinResult<()> {
    let (rows, cols) = super::matrix_validation::covariance_matrix_extents(tensor.shape());
    if tensor.shape().len() > 2 || rows != cols {
        return Ok(());
    }
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    for col in 0..cols {
        for row in 0..col {
            if storage.value_at(row + col * rows) != storage.value_at(col + row * rows) {
                return Err(conversion_error(
                    name,
                    format!("{name}: covariance matrix must be symmetric"),
                ));
            }
        }
    }
    Ok(())
}
