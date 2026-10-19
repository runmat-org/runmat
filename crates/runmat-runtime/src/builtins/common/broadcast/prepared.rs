//! Finite host broadcasting retains one exact, admitted metadata backing.
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use runmat_value::ExactHostNumericAllocationError;

pub(crate) struct PreparedBroadcastPlan {
    metadata: HostTypedWorkspace<usize>,
    rank: usize,
    len: usize,
}

const CHECKPOINT_CHUNK: usize = 4096;

fn invalid_shape() -> HostAllocationError {
    HostAllocationError::Numeric(ExactHostNumericAllocationError::ExceedsPermit)
}

fn extent(shape: &[usize], rank: usize, axis: usize) -> usize {
    if shape.len() == 1 && rank >= 2 {
        if axis == 1 {
            shape[0]
        } else {
            1
        }
    } else {
        shape.get(axis).copied().unwrap_or(1)
    }
}

fn validate_shapes(
    left: &[usize],
    right: &[usize],
    output: &[usize],
) -> Result<usize, HostAllocationError> {
    validate_shapes_with(left, right, output, || Ok(()))
}

fn validate_shapes_with(
    left: &[usize],
    right: &[usize],
    output: &[usize],
    mut checkpoint: impl FnMut() -> Result<(), HostAllocationError>,
) -> Result<usize, HostAllocationError> {
    let rank = output.len();
    if rank < left.len().max(right.len()) {
        return Err(invalid_shape());
    }
    let mut empty = false;
    for (axis, &actual) in output.iter().enumerate() {
        if axis % CHECKPOINT_CHUNK == 0 {
            checkpoint()?;
        }
        let (a, b) = (extent(left, rank, axis), extent(right, rank, axis));
        let expected = if a == b || b == 1 {
            a
        } else if a == 1 {
            b
        } else {
            return Err(invalid_shape());
        };
        if actual != expected {
            return Err(invalid_shape());
        }
        empty |= actual == 0;
    }
    if empty {
        Ok(0)
    } else {
        let mut len = 1usize;
        for (axis, &extent) in output.iter().enumerate() {
            if axis % CHECKPOINT_CHUNK == 0 {
                checkpoint()?;
            }
            len = len
                .checked_mul(extent)
                .ok_or(HostAllocationError::Overflow)?;
        }
        Ok(len)
    }
}

#[cfg(test)]
fn initialize(
    metadata: &mut [usize],
    left: &[usize],
    right: &[usize],
    output: &[usize],
) -> Result<usize, HostAllocationError> {
    initialize_with_checkpoint(metadata, left, right, output, || Ok(()))
}

fn initialize_with_checkpoint(
    metadata: &mut [usize],
    left: &[usize],
    right: &[usize],
    output: &[usize],
    mut checkpoint: impl FnMut() -> Result<(), HostAllocationError>,
) -> Result<usize, HostAllocationError> {
    let rank = output.len();
    if metadata.len() != rank.checked_mul(4).ok_or(HostAllocationError::Overflow)? {
        return Err(invalid_shape());
    }
    let len = validate_shapes_with(left, right, output, &mut checkpoint)?;
    for chunk in metadata.chunks_mut(CHECKPOINT_CHUNK) {
        checkpoint()?;
        chunk.fill(0);
    }
    for (target, source) in metadata[..rank]
        .chunks_mut(CHECKPOINT_CHUNK)
        .zip(output.chunks(CHECKPOINT_CHUNK))
    {
        checkpoint()?;
        target.copy_from_slice(source);
    }
    // An empty broadcast never reads an operand. Its zero extents also avoid
    // manufacturing an overflow from unreachable source-stride arithmetic.
    if len != 0 {
        let (mut stride_a, mut stride_b) = (1usize, 1usize);
        for axis in 0..rank {
            if axis % CHECKPOINT_CHUNK == 0 {
                checkpoint()?;
            }
            let (a, b) = (extent(left, rank, axis), extent(right, rank, axis));
            metadata[rank + axis] = if a > 1 { stride_a } else { 0 };
            metadata[2 * rank + axis] = if b > 1 { stride_b } else { 0 };
            stride_a = stride_a
                .checked_mul(a.max(1))
                .ok_or(HostAllocationError::Overflow)?;
            stride_b = stride_b
                .checked_mul(b.max(1))
                .ok_or(HostAllocationError::Overflow)?;
        }
    }
    Ok(len)
}

impl PreparedBroadcastPlan {
    pub(crate) fn quoted_bytes(rank: usize) -> Result<u64, HostAllocationError> {
        HostTypedWorkspace::<usize>::quoted_bytes(
            rank.checked_mul(4).ok_or(HostAllocationError::Overflow)?,
        )
    }

    pub(crate) fn quoted_for_shapes(
        left: &[usize],
        right: &[usize],
        output: &[usize],
    ) -> Result<u64, HostAllocationError> {
        validate_shapes(left, right, output)?;
        Self::quoted_bytes(output.len())
    }

    pub(crate) fn prepare(
        receipt: &HostAllocationReceipt,
        runtime: &RuntimeContext,
        left: &[usize],
        right: &[usize],
        output: &[usize],
    ) -> Result<Self, HostAllocationError> {
        let mut metadata = receipt.temporary_typed(
            runtime,
            output
                .len()
                .checked_mul(4)
                .ok_or(HostAllocationError::Overflow)?,
        )?;
        let len = initialize_with_checkpoint(metadata.as_mut_slice(), left, right, output, || {
            receipt.checkpoint(runtime)
        })?;
        receipt.checkpoint(runtime)?;
        Ok(Self {
            metadata,
            rank: output.len(),
            len,
        })
    }

    pub(crate) fn output_shape(&self) -> &[usize] {
        &self.metadata.as_slice()[..self.rank]
    }

    pub(crate) fn iter(&mut self) -> PreparedBroadcastIter<'_> {
        let (fixed, coords) = self.metadata.as_mut_slice().split_at_mut(3 * self.rank);
        coords.fill(0);
        let (shape, strides) = fixed.split_at(self.rank);
        let (advance_a, advance_b) = strides.split_at(self.rank);
        PreparedBroadcastIter {
            shape,
            advance_a,
            advance_b,
            coords,
            len: self.len,
            offset: 0,
            index_a: 0,
            index_b: 0,
        }
    }
}

pub(crate) struct PreparedBroadcastIter<'a> {
    shape: &'a [usize],
    advance_a: &'a [usize],
    advance_b: &'a [usize],
    coords: &'a mut [usize],
    len: usize,
    offset: usize,
    index_a: usize,
    index_b: usize,
}

impl Iterator for PreparedBroadcastIter<'_> {
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.len {
            return None;
        }
        let result = (self.offset, self.index_a, self.index_b);
        self.offset += 1;
        if self.offset != self.len {
            for axis in 0..self.shape.len() {
                self.coords[axis] += 1;
                if self.coords[axis] < self.shape[axis] {
                    self.index_a += self.advance_a[axis];
                    self.index_b += self.advance_b[axis];
                    break;
                }
                self.coords[axis] = 0;
                self.index_a -= (self.shape[axis] - 1) * self.advance_a[axis];
                self.index_b -= (self.shape[axis] - 1) * self.advance_b[axis];
            }
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests;
