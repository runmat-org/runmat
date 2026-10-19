use super::*;
use crate::builtins::common::shape::{is_scalar_shape, normalize_scalar_shape};

pub(super) fn resolve_output_shape(
    shape: &[usize],
    selection: &DimSelection,
    reduced_dims: &[usize],
) -> BuiltinResult<Vec<usize>> {
    if is_scalar_shape(shape) {
        return Ok(normalize_scalar_shape(shape));
    }
    let mut output = shape.to_vec();
    match selection {
        DimSelection::All => {
            output.fill(1);
        }
        _ => {
            for &dim in reduced_dims {
                if dim < output.len() {
                    output[dim] = 1;
                }
            }
        }
    }
    Ok(output)
}

pub(in crate::builtins::math::reduction::max) struct ResolvedDims {
    pub(in crate::builtins::math::reduction::max) output_shape: Vec<usize>,
    pub(in crate::builtins::math::reduction::max) reduced_dims: Vec<usize>,
    pub(in crate::builtins::math::reduction::max) reduce_all: bool,
    pub(super) dims_mask: Vec<bool>,
    pub(super) reduce_strides: Vec<usize>,
}

pub(in crate::builtins::math::reduction::max) fn resolve_reduction_dims(
    shape: &[usize],
    selection: &DimSelection,
) -> BuiltinResult<ResolvedDims> {
    if is_scalar_shape(shape) {
        return Ok(ResolvedDims {
            output_shape: normalize_scalar_shape(shape),
            reduced_dims: Vec::new(),
            reduce_all: true,
            dims_mask: Vec::new(),
            reduce_strides: Vec::new(),
        });
    }

    let mut reduced_dims = match selection {
        DimSelection::Auto => {
            let mut dim = None;
            for (index, &len) in shape.iter().enumerate() {
                if len > 1 {
                    dim = Some(index);
                    break;
                }
            }
            vec![dim.unwrap_or(0)]
        }
        DimSelection::Dim(dim) => {
            if *dim == 0 {
                return Err(max_invalid_argument("max: dimension must be >= 1"));
            }
            let index = dim.saturating_sub(1);
            if index >= shape.len() {
                Vec::new()
            } else {
                vec![index]
            }
        }
        DimSelection::Vec(dims) => {
            if dims.is_empty() {
                Vec::new()
            } else {
                dims.iter()
                    .filter_map(|dim| {
                        if *dim == 0 {
                            None
                        } else {
                            let idx = dim - 1;
                            if idx < shape.len() {
                                Some(idx)
                            } else {
                                None
                            }
                        }
                    })
                    .collect()
            }
        }
        DimSelection::All => (0..shape.len()).collect(),
    };

    reduced_dims.sort_unstable();
    reduced_dims.dedup();

    let reduce_all = !reduced_dims.is_empty()
        && reduced_dims.len() == shape.len()
        && reduced_dims.iter().enumerate().all(|(i, &d)| i == d);

    let output_shape = resolve_output_shape(shape, selection, &reduced_dims)?;
    let mut dims_mask = vec![false; shape.len()];
    for &dim in &reduced_dims {
        if dim < dims_mask.len() {
            dims_mask[dim] = true;
        }
    }
    let reduce_strides = compute_subspace_strides(shape, &reduced_dims);

    Ok(ResolvedDims {
        output_shape,
        reduced_dims,
        reduce_all,
        dims_mask,
        reduce_strides,
    })
}

pub(super) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &len in shape {
        strides.push(stride);
        stride = stride.saturating_mul(len.max(1));
    }
    strides
}

fn compute_subspace_strides(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let mut strides = Vec::with_capacity(dims.len());
    let mut accum = 1usize;
    for &dim in dims {
        let len = shape.get(dim).copied().unwrap_or(1).max(1);
        strides.push(accum);
        accum = accum.saturating_mul(len);
    }
    strides
}

pub(super) fn map_output_index(
    coords: &[usize],
    output_strides: &[usize],
    dims_mask: &[bool],
) -> usize {
    if coords.is_empty() {
        return 0;
    }
    let mut index = 0usize;
    for (dim, stride) in output_strides.iter().enumerate() {
        let coord = if *dims_mask.get(dim).unwrap_or(&false) {
            0
        } else {
            coords[dim]
        };
        index = index.saturating_add(coord.saturating_mul(*stride));
    }
    index
}

pub(super) fn map_reduce_index(
    coords: &[usize],
    reduced_dims: &[usize],
    reduce_strides: &[usize],
    reduce_all: bool,
) -> usize {
    if reduced_dims.is_empty() {
        return 0;
    }
    if reduce_all {
        // When all dimensions are reduced, the full index is used separately.
        return 0;
    }
    let mut index = 0usize;
    for (pos, &dim) in reduced_dims.iter().enumerate() {
        if let Some(coord) = coords.get(dim) {
            if let Some(stride) = reduce_strides.get(pos) {
                index = index.saturating_add(coord.saturating_mul(*stride));
            }
        }
    }
    index
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
        if shape[dim] == 0 {
            continue;
        }
        coords[dim] += 1;
        if coords[dim] < shape[dim] {
            break;
        }
        coords[dim] = 0;
    }
}

pub(super) fn default_dimension_from_shape(shape: &[usize]) -> usize {
    if is_scalar_shape(shape) {
        return 1;
    }
    for (i, &len) in shape.iter().enumerate() {
        if len > 1 {
            return i + 1;
        }
    }
    1
}
