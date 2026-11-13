//! Exact retained outputs for finite host numerical permutations.

use super::permute_error;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
    HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_value::{
    ComplexStorage, ComplexTensor, ExactHostComplexTensorAllocation,
    ExactHostIntegerComplexTensorAllocation, ExactHostLogicalTensorAllocation,
    ExactHostNumericAllocationError, ExactHostTensorAllocation, LogicalArray, NumericDType, Tensor,
};

const CHECKPOINT_CHUNK: usize = 4096;

mod geometry;
mod output;
use geometry::{initialize_metadata, validate_order_range, validate_unique_order};
pub(super) use output::{complex, logical, tensor};

fn allocation_error(
    context: &RuntimeContext,
    builtin: &'static str,
    cause: HostAllocationError,
) -> RuntimeError {
    build_runtime_error(format!("{builtin}: host permutation allocation: {cause}"))
        .with_builtin(builtin)
        .with_identifier(format!(
            "{}:{}",
            context.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

fn numeric_error(error: ExactHostNumericAllocationError) -> HostAllocationError {
    HostAllocationError::Numeric(error)
}

struct PreparedPermutation {
    builtin: &'static str,
    receipt: HostAllocationReceipt,
    metadata: HostTypedWorkspace<usize>,
    rank: usize,
    len: usize,
}

impl PreparedPermutation {
    fn prepare(
        context: &RuntimeContext,
        builtin: &'static str,
        source_shape: &[usize],
        source_len: usize,
        order: &[usize],
        retained_bytes: usize,
        retained_claims: usize,
    ) -> BuiltinResult<Option<Self>> {
        if HostAllocationReceipt::policy(Some(context))
            .map_err(|cause| allocation_error(context, builtin, cause))?
            != HostAllocationPolicy::Managed
        {
            return Ok(None);
        }
        let rank = order.len();
        if source_shape.len() > rank {
            return Err(permute_error(
                builtin,
                format!(
                    "{builtin}: order length ({rank}) must be at least ndims(A) ({})",
                    source_shape.len()
                ),
            ));
        }
        validate_order_range(context, builtin, order)?;
        let metadata_len = rank.checked_mul(3).ok_or_else(|| {
            permute_error(
                builtin,
                format!("{builtin}: permutation rank exceeds platform limits"),
            )
        })?;
        let scratch = HostTypedWorkspace::<usize>::quoted_bytes(metadata_len)
            .map_err(|cause| allocation_error(context, builtin, cause))?;
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(retained_bytes).map_err(|_| {
                permute_error(
                    builtin,
                    format!("{builtin}: output size exceeds platform limits"),
                )
            })?,
            retained_claims,
        )
        .map_err(|cause| allocation_error(context, builtin, cause))?;
        let plan = HostAllocationPlan::new(scratch, retained)
            .map_err(|cause| allocation_error(context, builtin, cause))?;
        let receipt = HostAllocationReceipt::admit(Some(context), plan)
            .map_err(|cause| allocation_error(context, builtin, cause))?
            .ok_or_else(|| {
                allocation_error(context, builtin, HostAllocationError::MissingManagedOwner)
            })?;
        let mut metadata = receipt
            .temporary_typed::<usize>(context, metadata_len)
            .map_err(|cause| allocation_error(context, builtin, cause))?;
        validate_unique_order(
            context,
            builtin,
            &mut metadata.as_mut_slice()[..rank],
            order,
            &receipt,
        )?;
        initialize_metadata(
            context,
            builtin,
            metadata.as_mut_slice(),
            source_shape,
            source_len,
            order,
            &receipt,
        )?;
        Ok(Some(Self {
            builtin,
            receipt,
            metadata,
            rank,
            len: source_len,
        }))
    }

    fn output_shape(&self) -> &[usize] {
        &self.metadata.as_slice()[..self.rank]
    }

    fn source_index(&self, destination_index: usize) -> Result<usize, RuntimeError> {
        let metadata = self.metadata.as_slice();
        let shape = &metadata[..self.rank];
        let strides = &metadata[2 * self.rank..3 * self.rank];
        let mut remainder = destination_index;
        let mut source_index = 0usize;
        for (destination_dimension, &extent) in shape.iter().enumerate() {
            let coordinate = if extent == 0 { 0 } else { remainder % extent };
            if extent != 0 {
                remainder /= extent;
            }
            source_index = source_index
                .checked_add(
                    coordinate
                        .checked_mul(strides[destination_dimension])
                        .ok_or_else(|| {
                            permute_error(self.builtin, "permutation index exceeds platform limits")
                        })?,
                )
                .ok_or_else(|| {
                    permute_error(self.builtin, "permutation index exceeds platform limits")
                })?;
        }
        Ok(source_index)
    }
}

fn managed_policy(context: &RuntimeContext, builtin: &'static str) -> BuiltinResult<bool> {
    HostAllocationReceipt::policy(Some(context))
        .map(|policy| policy == HostAllocationPolicy::Managed)
        .map_err(|cause| allocation_error(context, builtin, cause))
}

#[cfg(test)]
mod tests;
