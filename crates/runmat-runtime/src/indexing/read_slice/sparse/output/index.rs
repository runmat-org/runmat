use super::*;

pub(in crate::indexing) fn indices(
    context: Option<&RuntimeContext>,
    count: usize,
    mut value: impl FnMut(usize) -> Result<usize, RuntimeError>,
) -> Result<runmat_value::HostIndexBuffer, RuntimeError> {
    use runmat_value::ExactHostIndexBufferAllocation;
    let owner = Owner::new(context)?;
    let mut work = 0usize;
    let raw = ExactHostIndexBufferAllocation::requested_bytes(count)
        .map_err(|cause| allocation(context, cause.into()))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(raw).map_err(|_| allocation(context, HostAllocationError::Overflow))?,
        ExactHostIndexBufferAllocation::retained_claim_count(),
    )
    .map_err(|cause| allocation(context, cause))?;
    let receipt = HostAllocationReceipt::admit(
        context,
        HostAllocationPlan::new(0, retained).map_err(|cause| allocation(context, cause))?,
    )
    .map_err(|cause| allocation(context, cause))?;
    let result = if let Some(receipt) = &receipt {
        receipt
            .with_output_construction(
                context.expect("managed index output has an owner"),
                |permit| {
                    let mut output = ExactHostIndexBufferAllocation::new(count, permit)
                        .map_err(|cause| allocation(context, cause.into()))?;
                    for (index, slot) in output.as_mut_slice().iter_mut().enumerate() {
                        super::step(&owner, &mut work)?;
                        *slot = value(index)?;
                    }
                    owner.check()?;
                    output
                        .into_buffer_fallible()
                        .map_err(|cause| allocation(context, cause.into()))
                },
            )
            .map_err(|cause| allocation(context, cause))??
    } else {
        let mut output = runmat_value::exact_capacity_vec(count)
            .map_err(|cause| allocation(context, cause.into()))?;
        for index in 0..count {
            super::step(&owner, &mut work)?;
            output.push(value(index)?);
        }
        output.into()
    };
    owner.check()?;
    Ok(result)
}
