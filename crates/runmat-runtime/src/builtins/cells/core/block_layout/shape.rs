pub(in crate::builtins::cells::core) fn extend_with_ones(
    shape: &[usize],
    rank: usize,
) -> Vec<usize> {
    let mut result = shape.to_vec();
    result.resize(result.len().max(rank), 1);
    result
}

pub(in crate::builtins::cells::core) fn checked_element_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
}

pub(in crate::builtins::cells::core) fn column_major_strides(
    shape: &[usize],
) -> Option<Vec<usize>> {
    let mut stride = 1usize;
    let mut strides = Vec::with_capacity(shape.len());
    for extent in shape {
        strides.push(stride);
        stride = stride.checked_mul((*extent).max(1))?;
    }
    Some(strides)
}

pub(in crate::builtins::cells::core) fn prefix_offsets(extents: &[usize]) -> Option<Vec<usize>> {
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(extents.len());
    for extent in extents {
        offsets.push(offset);
        offset = offset.checked_add(*extent)?;
    }
    Some(offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_arithmetic_rejects_overflow() {
        assert!(checked_element_count(&[usize::MAX, 2]).is_none());
        assert!(column_major_strides(&[usize::MAX, 2, 2]).is_none());
        assert!(prefix_offsets(&[usize::MAX, 1]).is_none());
    }
}
