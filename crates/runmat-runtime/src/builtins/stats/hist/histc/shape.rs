use super::*;
pub(super) fn canonical_shape(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

pub(super) fn shape_with_dim(mut shape: Vec<usize>, dim: usize) -> Vec<usize> {
    if shape.len() <= dim {
        shape.resize(dim + 1, 1);
    }
    shape
}

pub(super) fn trim_trailing_added_dims(
    mut shape: Vec<usize>,
    original_rank: usize,
    dim: usize,
) -> Vec<usize> {
    while shape.len() > original_rank.max(dim + 1) && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

pub(super) fn is_vector_shape(shape: &[usize]) -> bool {
    shape.iter().filter(|&&dim| dim != 1).count() <= 1
}

pub(super) fn first_nonsingleton_dim(shape: &[usize]) -> usize {
    shape.iter().position(|&dim| dim != 1).unwrap_or(0)
}

pub(super) fn checked_product(dims: &[usize]) -> BuiltinResult<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| internal_error("histc: shape product overflow"))
}
