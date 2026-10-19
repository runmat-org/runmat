use super::*;

pub(super) fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape.push(1);
        shape.push(1);
    }
    if shape.len() == 1 {
        shape.push(1);
    }
    shape
}

pub(super) fn selected_dim(shape: &[usize], requested: Option<usize>) -> usize {
    if let Some(dim) = requested {
        return dim;
    }
    shape
        .iter()
        .position(|&len| len > 1)
        .map(|idx| idx + 1)
        .unwrap_or(1)
}

pub(super) fn output_shape_for_topk(
    kind: TopKKind,
    shape: &[usize],
    axis: usize,
    take: usize,
) -> BuiltinResult<Vec<usize>> {
    let mut output = shape.to_vec();
    if axis >= output.len() {
        let rank = axis
            .checked_add(1)
            .ok_or_else(|| topk_invalid_argument(kind, "dimension exceeds supported rank"))?;
        output
            .try_reserve_exact(rank - output.len())
            .map_err(|_| topk_invalid_argument(kind, "dimension exceeds supported rank"))?;
        output.resize(rank, 1);
    }
    output[axis] = take;
    Ok(output)
}

pub(super) fn checked_element_count(kind: TopKKind, shape: &[usize]) -> BuiltinResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| topk_internal(kind, "shape element count overflow"))
}

pub(super) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for idx in 1..shape.len() {
        strides[idx] = strides[idx - 1].saturating_mul(shape[idx - 1]);
    }
    strides
}

pub(super) fn map_linear_index(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .fold(0usize, |acc, (&coord, &stride)| {
            acc.saturating_add(coord.saturating_mul(stride))
        })
}

pub(super) fn increment_coords(coords: &mut [usize], shape: &[usize]) {
    for dim in 0..coords.len() {
        coords[dim] += 1;
        if coords[dim] < shape[dim] {
            break;
        }
        coords[dim] = 0;
    }
}
