use crate::runtime_error::semantic_error as mex;
use crate::RuntimeError;

/// Only scalar replication or the original column-major order is permitted.
/// Singleton-axis alignment needs no shape/coordinate allocation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AssignmentRhsLayout {
    scalar: bool,
    count: usize,
}
impl AssignmentRhsLayout {
    pub(in crate::indexing) fn linear(
        source_len: usize,
        count: usize,
    ) -> Result<Self, RuntimeError> {
        if source_len != 1 && source_len != count {
            return Err(mex("ShapeMismatch", "shape mismatch for slice assign"));
        }
        Ok(Self {
            scalar: source_len == 1,
            count,
        })
    }
    pub(crate) fn multidimensional(
        source_shape: &[usize],
        source_len: usize,
        selection: &[usize],
    ) -> Result<Self, RuntimeError> {
        validate_assignment_rhs_shape(source_shape, source_len, selection)?;
        let count = checked_assignment_element_count(selection)?;
        Ok(Self {
            scalar: source_len == 1,
            count: if selection.is_empty() { 0 } else { count },
        })
    }
    pub(in crate::indexing) fn count(self) -> usize {
        self.count
    }
    pub(crate) fn source_index(self, ordinal: usize) -> usize {
        assert!(ordinal < self.count, "validated RHS output ordinal");
        if self.scalar {
            0
        } else {
            ordinal
        }
    }
}

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
    validate_assignment_rhs_shape(rhs_shape, rhs_len, selection_lengths)?;
    if rhs_len == 1 {
        Ok(vec![1; selection_lengths.len()])
    } else {
        Ok(selection_lengths.to_vec())
    }
}

fn validate_assignment_rhs_shape(
    rhs_shape: &[usize],
    rhs_len: usize,
    selection_lengths: &[usize],
) -> Result<(), RuntimeError> {
    if rhs_len == 1 {
        return Ok(());
    }
    let dims = selection_lengths.len();
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
    Ok(())
}

pub(super) fn validate_empty_selection_rhs_shape(
    rhs_shape: &[usize],
    rhs_len: usize,
    selection_lengths: &[usize],
) -> Result<(), RuntimeError> {
    validate_assignment_rhs_shape(rhs_shape, rhs_len, selection_lengths)
}

#[cfg(test)]
mod tests;
