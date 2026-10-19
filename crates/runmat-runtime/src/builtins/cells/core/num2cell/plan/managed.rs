//! Claimed partition metadata for the Managed string path. The ordinary
//! planner and this owner share the source-indexing geometry in `plan`.

use runmat_value::{SharedShape, Value};

use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
    HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

use super::{checked_element_count, indexed_source, PartitionGeometry};
use crate::builtins::cells::core::num2cell::{dimension, error};

pub(in crate::builtins::cells::core::num2cell) struct ManagedPartitionPlan {
    source_shape: SharedShape,
    metadata: HostTypedWorkspace<usize>,
    rank: usize,
    slice_rank: usize,
    output_count: usize,
    slice_count: usize,
    _receipt: HostAllocationReceipt,
}

fn admission(runtime: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        Some(runtime),
        "RunMat:num2cell:HostAllocation",
        "num2cell: partition admission failed",
        cause,
    )
}

impl ManagedPartitionPlan {
    pub(in crate::builtins::cells::core::num2cell) fn prepare(
        source_shape: &SharedShape,
        dimensions: Option<&Value>,
        runtime: &RuntimeContext,
    ) -> BuiltinResult<Self> {
        let mut dimension_count = 0usize;
        if let Some(dimensions) = dimensions {
            dimension::visit(dimensions, |_| {
                dimension_count = dimension_count
                    .checked_add(1)
                    .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
                Ok(())
            })?;
        }
        let scalar_mode = dimension_count == 0;
        let rank = source_shape.len();
        let slice_rank = if scalar_mode { rank.max(2) } else { rank };
        let count = rank
            .checked_mul(2)
            .and_then(|value| value.checked_add(slice_rank))
            .ok_or_else(|| admission(runtime, HostAllocationError::Overflow))?;
        let scratch = HostTypedWorkspace::<usize>::quoted_bytes(count)
            .map_err(|cause| admission(runtime, cause))?;
        let allocation =
            HostAllocationPlan::new(scratch, 0).map_err(|cause| admission(runtime, cause))?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), allocation)
            .map_err(|cause| admission(runtime, cause))?
            .ok_or_else(|| admission(runtime, HostAllocationError::MissingManagedOwner))?;
        let mut metadata = receipt
            .temporary_typed::<usize>(runtime, count)
            .map_err(|cause| admission(runtime, cause))?;
        let (output, rest) = metadata.as_mut_slice().split_at_mut(rank);
        let (slice, mapping) = rest.split_at_mut(slice_rank);
        output.copy_from_slice(source_shape.as_slice());
        slice.fill(1);
        if !scalar_mode {
            let dimensions = dimensions.expect("nonempty dimension control has a value");
            dimension::visit(dimensions, |dimension| {
                receipt
                    .checkpoint(runtime)
                    .map_err(|cause| admission(runtime, cause))?;
                let source = dimension
                    .checked_sub(1)
                    .filter(|axis| *axis < rank)
                    .ok_or_else(|| {
                        error::invalid_input(format!("dimensions must be between 1 and {rank}"))
                    })?;
                if mapping[source] != 0 {
                    return Err(error::invalid_input(
                        "dimensions must not contain duplicates",
                    ));
                }
                mapping[source] = 1;
                output[source] = 1;
                Ok(())
            })?;
            let mut ordinal = 0usize;
            dimension::visit(dimensions, |dimension| {
                receipt
                    .checkpoint(runtime)
                    .map_err(|cause| admission(runtime, cause))?;
                let source = dimension - 1;
                let destination = (0..rank)
                    .filter(|&axis| mapping[axis] != 0)
                    .nth(ordinal)
                    .ok_or_else(|| error::internal("partition destination is absent"))?;
                slice[destination] = source_shape[source];
                mapping[source] = destination + 1;
                ordinal += 1;
                Ok(())
            })?;
        }
        let output_count = checked_element_count(output)?;
        let slice_count = if scalar_mode {
            usize::from(output_count != 0)
        } else {
            checked_element_count(slice)?
        };
        receipt
            .checkpoint(runtime)
            .map_err(|cause| admission(runtime, cause))?;
        Ok(Self {
            source_shape: source_shape.clone(),
            metadata,
            rank,
            slice_rank,
            output_count,
            slice_count,
            _receipt: receipt,
        })
    }

    fn mapping(&self) -> &[usize] {
        &self.metadata.as_slice()[self.rank + self.slice_rank..]
    }
}

impl PartitionGeometry for ManagedPartitionPlan {
    fn output_shape(&self) -> &[usize] {
        &self.metadata.as_slice()[..self.rank]
    }

    fn slice_shape(&self) -> &[usize] {
        &self.metadata.as_slice()[self.rank..self.rank + self.slice_rank]
    }

    fn output_count(&self) -> usize {
        self.output_count
    }
    fn slice_count(&self) -> usize {
        self.slice_count
    }

    fn source_index(&self, output: usize, slice: usize) -> BuiltinResult<usize> {
        indexed_source(self, self.source_shape.as_slice(), output, slice, |axis| {
            self.mapping()[axis].checked_sub(1)
        })
    }
}
