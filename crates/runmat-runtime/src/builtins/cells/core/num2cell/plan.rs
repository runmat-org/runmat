use std::collections::HashSet;

use super::error;

pub(super) struct PartitionPlan {
    pub(super) output_shape: Vec<usize>,
    pub(super) slice_shape: Vec<usize>,
    source_shape: Vec<usize>,
    source_to_slice: Vec<Option<usize>>,
    output_count: usize,
    slice_count: usize,
}

impl PartitionPlan {
    pub(super) fn new(shape: &[usize], dimensions: &[usize]) -> crate::BuiltinResult<Self> {
        if dimensions.is_empty() {
            return Self::scalars(shape);
        }
        let axes = axes(dimensions, shape.len())?;
        let mut ordered_axes = axes.clone();
        ordered_axes.sort_unstable();
        let mut output_shape = shape.to_vec();
        let mut slice_shape = vec![1; shape.len()];
        let mut source_to_slice = vec![None; shape.len()];
        for (&source, &destination) in axes.iter().zip(ordered_axes.iter()) {
            output_shape[source] = 1;
            slice_shape[destination] = shape[source];
            source_to_slice[source] = Some(destination);
        }
        let output_count = element_count(&output_shape)?;
        let slice_count = element_count(&slice_shape)?;
        Ok(Self {
            output_shape,
            slice_shape,
            source_shape: shape.to_vec(),
            source_to_slice,
            output_count,
            slice_count,
        })
    }

    fn scalars(shape: &[usize]) -> crate::BuiltinResult<Self> {
        let count = element_count(shape)?;
        Ok(Self {
            output_shape: shape.to_vec(),
            slice_shape: vec![1, 1],
            source_shape: shape.to_vec(),
            source_to_slice: vec![None; shape.len()],
            output_count: count,
            slice_count: usize::from(count > 0),
        })
    }

    pub(super) fn groups(&self) -> impl Iterator<Item = crate::BuiltinResult<Vec<usize>>> + '_ {
        (0..self.output_count).map(|output_linear| self.source_indices(output_linear))
    }

    fn source_indices(&self, output_linear: usize) -> crate::BuiltinResult<Vec<usize>> {
        let output = row_major_coords(output_linear, &self.output_shape);
        (0..self.slice_count)
            .map(|slice_linear| {
                let slice = column_major_coords(slice_linear, &self.slice_shape);
                let source = (0..self.source_shape.len())
                    .map(|axis| self.source_to_slice[axis].map_or(output[axis], |to| slice[to]))
                    .collect::<Vec<_>>();
                column_major_linear(&source, &self.source_shape)
            })
            .collect()
    }
}

fn axes(dimensions: &[usize], rank: usize) -> crate::BuiltinResult<Vec<usize>> {
    let mut axes = Vec::with_capacity(dimensions.len());
    let mut seen = HashSet::with_capacity(dimensions.len());
    for &dimension in dimensions {
        let axis = dimension
            .checked_sub(1)
            .filter(|axis| *axis < rank)
            .ok_or_else(|| {
                error::invalid_input(format!("dimensions must be between 1 and {rank}"))
            })?;
        if !seen.insert(axis) {
            return Err(error::invalid_input(
                "dimensions must not contain duplicates",
            ));
        }
        axes.push(axis);
    }
    Ok(axes)
}

fn element_count(shape: &[usize]) -> crate::BuiltinResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| error::internal("shape exceeds addressable storage"))
}

fn row_major_coords(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        if shape[axis] != 0 {
            coords[axis] = linear % shape[axis];
            linear /= shape[axis];
        }
    }
    coords
}

pub(super) fn column_major_coords(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .map(|extent| {
            let coordinate = if *extent == 0 { 0 } else { linear % *extent };
            if *extent != 0 {
                linear /= *extent;
            }
            coordinate
        })
        .collect()
}

fn column_major_linear(coords: &[usize], shape: &[usize]) -> crate::BuiltinResult<usize> {
    let mut linear = 0usize;
    let mut stride = 1usize;
    for (&coordinate, &extent) in coords.iter().zip(shape) {
        linear = linear
            .checked_add(
                coordinate
                    .checked_mul(stride)
                    .ok_or_else(|| error::internal("index overflow"))?,
            )
            .ok_or_else(|| error::internal("index overflow"))?;
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| error::internal("index overflow"))?;
    }
    Ok(linear)
}
