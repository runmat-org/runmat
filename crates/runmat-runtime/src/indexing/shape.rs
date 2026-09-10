//! Shared shape rules for context-dependent indexing expressions.

use crate::{runtime_error::semantic_error, RuntimeError};

/// Returns the extent visible to one selector in an indexing expression.
///
/// With one selector MATLAB uses linear indexing. When fewer selectors than
/// dimensions are supplied, the final selector addresses the product of all
/// remaining dimensions. Missing trailing dimensions have extent one.
pub fn selector_extent(
    shape: &[usize],
    selector_count: usize,
    position: usize,
) -> Result<usize, RuntimeError> {
    if selector_count == 0 || position >= selector_count {
        return Err(semantic_error(
            "InvalidIndexSelectorPlan",
            "index selector position is out of bounds",
        ));
    }
    if selector_count == 1 {
        return checked_product(shape);
    }
    if position + 1 == selector_count {
        return checked_product(&shape[position.min(shape.len())..]);
    }
    Ok(shape.get(position).copied().unwrap_or(1))
}

pub(crate) fn column_major_strides(shape: &[usize]) -> Result<Vec<usize>, RuntimeError> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for extent in shape {
        strides.push(stride);
        stride = stride.checked_mul(*extent).ok_or_else(|| {
            semantic_error("IndexOutOfBounds", "index shape exceeds platform limits")
        })?;
    }
    Ok(strides)
}

fn checked_product(extents: &[usize]) -> Result<usize, RuntimeError> {
    extents.iter().try_fold(1usize, |length, extent| {
        length.checked_mul(*extent).ok_or_else(|| {
            semantic_error("IndexOutOfBounds", "index shape exceeds platform limits")
        })
    })
}

#[cfg(test)]
mod tests {
    use super::selector_extent;

    #[test]
    fn final_selector_collapses_remaining_dimensions() {
        let shape = [2, 3, 4];
        assert_eq!(selector_extent(&shape, 1, 0).unwrap(), 24);
        assert_eq!(selector_extent(&shape, 2, 0).unwrap(), 2);
        assert_eq!(selector_extent(&shape, 2, 1).unwrap(), 12);
        assert_eq!(selector_extent(&shape, 3, 2).unwrap(), 4);
        assert_eq!(selector_extent(&shape, 4, 3).unwrap(), 1);
    }
}
