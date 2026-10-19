use super::*;

pub(super) fn validate_order_range(
    context: &RuntimeContext,
    builtin: &'static str,
    order: &[usize],
) -> BuiltinResult<()> {
    for (position, &dimension) in order.iter().enumerate() {
        if position.is_multiple_of(CHECKPOINT_CHUNK) && context.is_cancelled() {
            return Err(allocation_error(
                context,
                builtin,
                HostAllocationError::Cancelled,
            ));
        }
        if dimension == 0 || dimension > order.len() {
            return Err(permute_error(
                builtin,
                format!("{builtin}: invalid permutation dimension {dimension}"),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_unique_order(
    context: &RuntimeContext,
    builtin: &'static str,
    seen: &mut [usize],
    order: &[usize],
    receipt: &HostAllocationReceipt,
) -> BuiltinResult<()> {
    // `HostTypedWorkspace` initializes its complete exact backing with
    // `usize::default`, and this runs before geometry overwrites that prefix.
    for (position, &dimension) in order.iter().enumerate() {
        if position.is_multiple_of(CHECKPOINT_CHUNK) {
            receipt
                .checkpoint(context)
                .map_err(|cause| allocation_error(context, builtin, cause))?;
        }
        if seen[dimension - 1] != 0 {
            return Err(permute_error(
                builtin,
                format!("{builtin}: duplicate dimension index {dimension}"),
            ));
        }
        seen[dimension - 1] = 1;
    }
    Ok(())
}

pub(super) fn initialize_metadata(
    context: &RuntimeContext,
    builtin: &'static str,
    metadata: &mut [usize],
    source_shape: &[usize],
    source_len: usize,
    order: &[usize],
    receipt: &HostAllocationReceipt,
) -> BuiltinResult<()> {
    let rank = order.len();
    let (output_shape, strides) = metadata.split_at_mut(rank);
    let (source_strides, selected_strides) = strides.split_at_mut(rank);
    let mut checkpoint = |position: usize| {
        if position.is_multiple_of(CHECKPOINT_CHUNK) {
            receipt
                .checkpoint(context)
                .map_err(|cause| allocation_error(context, builtin, cause))?;
        }
        Ok(())
    };
    if source_len == 0 {
        if !source_shape.contains(&0) {
            return Err(permute_error(
                builtin,
                format!("{builtin}: input data length does not match shape product"),
            ));
        }
        source_strides.fill(0);
        select_output_geometry(
            source_shape,
            order,
            output_shape,
            selected_strides,
            source_strides,
            &mut checkpoint,
        )?;
        receipt
            .checkpoint(context)
            .map_err(|cause| allocation_error(context, builtin, cause))?;
        return Ok(());
    }
    let mut source_stride = 1usize;
    let mut product = 1usize;
    for source_dimension in 0..rank {
        if source_dimension.is_multiple_of(CHECKPOINT_CHUNK) {
            receipt
                .checkpoint(context)
                .map_err(|cause| allocation_error(context, builtin, cause))?;
        }
        let extent = source_shape.get(source_dimension).copied().unwrap_or(1);
        product = product.checked_mul(extent).ok_or_else(|| {
            permute_error(
                builtin,
                format!("{builtin}: input shape exceeds platform limits"),
            )
        })?;
        source_strides[source_dimension] = source_stride;
        source_stride = source_stride.checked_mul(extent).ok_or_else(|| {
            permute_error(
                builtin,
                format!("{builtin}: input strides exceed platform limits"),
            )
        })?;
    }
    if product != source_len {
        return Err(permute_error(
            builtin,
            format!("{builtin}: input data length does not match shape product"),
        ));
    }
    select_output_geometry(
        source_shape,
        order,
        output_shape,
        selected_strides,
        source_strides,
        checkpoint,
    )
}

fn select_output_geometry(
    source_shape: &[usize],
    order: &[usize],
    output_shape: &mut [usize],
    selected_strides: &mut [usize],
    source_strides: &[usize],
    mut checkpoint: impl FnMut(usize) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    for (destination_dimension, &source_dimension) in order.iter().enumerate() {
        checkpoint(destination_dimension)?;
        selected_strides[destination_dimension] = source_strides[source_dimension - 1];
        output_shape[destination_dimension] =
            source_shape.get(source_dimension - 1).copied().unwrap_or(1);
    }
    Ok(())
}
