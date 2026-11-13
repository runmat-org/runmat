use std::collections::HashSet;

use super::error;

mod managed;
pub(super) use managed::ManagedPartitionPlan;
mod index;
pub(super) use index::column_major_coords;
use index::{column_major_linear, row_major_coords, source_index};

pub(super) trait PartitionGeometry {
    fn output_shape(&self) -> &[usize];
    fn slice_shape(&self) -> &[usize];
    fn output_count(&self) -> usize;
    fn slice_count(&self) -> usize;
    fn source_index(&self, output: usize, slice: usize) -> crate::BuiltinResult<usize>;
}

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

    /// Address one source element without allocating the coordinate and index
    /// vectors used by the ordinary grouped iterator. An admitted output
    /// builder can use this after preparing its complete scratch and result
    /// owners, including when a callback is cancelled between elements.
    pub(super) fn source_index(
        &self,
        output_linear: usize,
        slice_linear: usize,
    ) -> crate::BuiltinResult<usize> {
        source_index(
            &self.output_shape,
            &self.slice_shape,
            &self.source_shape,
            self.output_count,
            self.slice_count,
            output_linear,
            slice_linear,
            |axis| self.source_to_slice[axis],
        )
    }

    #[cfg(test)]
    pub(super) fn output_count(&self) -> usize {
        self.output_count
    }

    #[cfg(test)]
    pub(super) fn slice_count(&self) -> usize {
        self.slice_count
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

impl PartitionGeometry for PartitionPlan {
    fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    fn slice_shape(&self) -> &[usize] {
        &self.slice_shape
    }
    fn output_count(&self) -> usize {
        self.output_count
    }
    fn slice_count(&self) -> usize {
        self.slice_count
    }
    fn source_index(&self, output: usize, slice: usize) -> crate::BuiltinResult<usize> {
        self.source_index(output, slice)
    }
}

pub(super) fn checked_element_count(shape: &[usize]) -> crate::BuiltinResult<usize> {
    element_count(shape)
}

pub(super) fn indexed_source(
    geometry: &impl PartitionGeometry,
    source_shape: &[usize],
    output: usize,
    slice: usize,
    destination: impl FnMut(usize) -> Option<usize>,
) -> crate::BuiltinResult<usize> {
    source_index(
        geometry.output_shape(),
        geometry.slice_shape(),
        source_shape,
        geometry.output_count(),
        geometry.slice_count(),
        output,
        slice,
        destination,
    )
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
