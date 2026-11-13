//! Final primitive storage; no payload layout or conversion mirrors.
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::indexing::plan::admission::Owner;
use crate::RuntimeError;
use runmat_value::{ExactHostLogicalTensorAllocation, LogicalArray, SharedShape};

pub(in crate::indexing) fn allocation(
    context: Option<&RuntimeContext>,
    cause: HostAllocationError,
) -> RuntimeError {
    host_allocation_runtime_error(
        context,
        "RunMat:Indexing:PrimitiveAllocation",
        "indexed primitive output",
        cause,
    )
}

pub(super) fn shape(
    context: Option<&RuntimeContext>,
    source: &SharedShape,
) -> Result<SharedShape, RuntimeError> {
    let owner = Owner::new(context)?;
    let shape = if source.is_admitted()
        || HostAllocationReceipt::policy(context).map_err(|e| allocation(context, e))?
            == HostAllocationPolicy::Unbounded
    {
        source.clone()
    } else {
        crate::indexing::plan::admission::shape(context, source.len(), |axis| Ok(source[axis]))?
    };
    owner.check()?;
    Ok(shape)
}

pub(in crate::indexing) fn logical(
    context: Option<&RuntimeContext>,
    shape: &[usize],
    fill: impl FnOnce(&mut [u8]) -> Result<(), RuntimeError>,
) -> Result<LogicalArray, RuntimeError> {
    let owner = Owner::new(context)?;
    let bytes = HostAllocationPlan::retained_logical_tensor_bytes(shape)
        .map_err(|e| allocation(context, e))?;
    let receipt = HostAllocationReceipt::admit(
        context,
        HostAllocationPlan::new(0, bytes).map_err(|e| allocation(context, e))?,
    )
    .map_err(|e| allocation(context, e))?;
    let result = if let Some(receipt) = &receipt {
        let runtime = context.expect("admitted primitive has explicit context");
        receipt
            .with_output_construction(runtime, |permit| {
                let mut output = ExactHostLogicalTensorAllocation::new(shape, permit)
                    .map_err(|e| allocation(context, e.into()))?;
                fill(output.as_mut_slice())?;
                owner.check()?;
                output
                    .into_array()
                    .map_err(|e| allocation(context, e.into()))
            })
            .map_err(|e| allocation(context, e))??
    } else {
        let count = super::scalar::checked_shape_len(shape)?;
        let mut bytes = vec![0; count];
        fill(&mut bytes)?;
        LogicalArray::new(bytes, shape.to_vec()).map_err(super::scalar::shape_error)?
    };
    owner.check()?;
    Ok(result)
}
