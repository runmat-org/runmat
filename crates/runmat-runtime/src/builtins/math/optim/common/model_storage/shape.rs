//! Fill the final canonical shared shape, without an unmanaged shape mirror.
use super::*;
use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
use runmat_value::SharedShape;

pub(in crate::builtins::math::optim) async fn shape(
    rank: usize,
    mut dimension: impl FnMut(usize) -> BuiltinResult<usize>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<SharedShape> {
    cooperation.check_allocation_epoch()?;
    let runtime = cooperation.runtime().cloned();
    let context = runtime.as_ref();
    let raw = SharedShape::requested_bytes_for_rank(rank)
        .map_err(|source| allocation(context, source.into()))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| allocation(context, HostAllocationError::Overflow))?,
        1,
    )
    .map_err(|source| allocation(context, source))?;
    let receipt = HostAllocationReceipt::admit(
        context,
        HostAllocationPlan::new(0, retained).map_err(|source| allocation(context, source))?,
    )
    .map_err(|source| allocation(context, source))?;
    cooperation.check_allocation_epoch()?;
    match receipt {
        Some(receipt) => {
            let context = context.expect("managed shape has original context");
            let mut shape = receipt
                .with_output_allocation(context, |permit| SharedShape::begin_exact(rank, permit))
                .map_err(|source| allocation(Some(context), source))?;
            for index in 0..rank {
                receipt
                    .checkpoint(context)
                    .map_err(|source| allocation(Some(context), source))?;
                shape
                    .try_push(dimension(index)?)
                    .map_err(|_| allocation(Some(context), HostAllocationError::Overflow))?;
                cooperation.completed(1).await?;
            }
            receipt
                .checkpoint(context)
                .map_err(|source| allocation(Some(context), source))?;
            cooperation.check_allocation_epoch()?;
            shape
                .try_finish()
                .map_err(|_| allocation(Some(context), HostAllocationError::Overflow))
        }
        None => {
            let mut dimensions = runmat_value::exact_capacity_vec(rank)
                .map_err(|source| allocation(context, source.into()))?;
            for index in 0..rank {
                dimensions.push(dimension(index)?);
                cooperation.completed(1).await?;
            }
            cooperation.check_allocation_epoch()?;
            Ok(SharedShape::new_unmanaged(dimensions))
        }
    }
}
