use crate::runtime_error::semantic_error as mex;
use crate::RuntimeError;

pub(super) fn checked_assignment_element_count(shape: &[usize]) -> Result<usize, RuntimeError> {
    shape.iter().try_fold(1usize, |length, &dimension| {
        length
            .checked_mul(dimension)
            .ok_or_else(|| mex("ShapeMismatch", "shape mismatch for slice assign"))
    })
}

/// Align the RHS column-major view with the selected destination dimensions.
/// Singleton dimensions do not change element order. Non-singleton dimensions
/// must match in order and length; only a scalar may expand.
pub(super) fn validated_assignment_rhs_shape(
    rhs_shape: &[usize],
    rhs_len: usize,
    selection_lengths: &[usize],
) -> Result<Vec<usize>, RuntimeError> {
    let dims = selection_lengths.len();
    if rhs_len == 1 {
        return Ok(vec![1; dims]);
    }
    let matches = if dims == 1 {
        rhs_len == selection_lengths[0]
    } else {
        rhs_shape
            .iter()
            .filter(|&&length| length != 1)
            .eq(selection_lengths.iter().filter(|&&length| length != 1))
    };
    let expected = checked_assignment_element_count(selection_lengths)?;
    if !matches || expected != rhs_len {
        return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
    }
    Ok(selection_lengths.to_vec())
}

#[cfg(test)]
mod tests;
