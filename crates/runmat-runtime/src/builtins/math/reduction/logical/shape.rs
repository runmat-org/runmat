use crate::builtins::common::shape::is_scalar_shape;

use super::arguments::ReductionSpec;

pub(super) fn dimensions(spec: &ReductionSpec, shape: &[usize]) -> Vec<usize> {
    match spec {
        ReductionSpec::Default => vec![default_dimension(shape)],
        ReductionSpec::Dimension(dimension) => vec![*dimension],
        ReductionSpec::Dimensions(dimensions) => {
            let mut dimensions = dimensions.clone();
            dimensions.sort_unstable();
            dimensions.dedup();
            dimensions
        }
        ReductionSpec::All => {
            if is_scalar_shape(shape) {
                vec![1]
            } else {
                (1..=shape.len()).collect()
            }
        }
    }
}

pub(super) fn default_dimension(shape: &[usize]) -> usize {
    if is_scalar_shape(shape) {
        return 1;
    }
    shape
        .iter()
        .position(|extent| *extent != 1)
        .map_or(1, |index| index + 1)
}

pub(super) fn product(dimensions: &[usize]) -> usize {
    dimensions.iter().copied().fold(1, usize::saturating_mul)
}
