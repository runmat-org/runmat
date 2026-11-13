//! Pure native-integer binary geometry shared by prospective and live admission.

use super::*;

pub(super) struct IntegerBinaryGeometry {
    pub(super) broadcast: IntegerBroadcastPlan,
    pub(super) output_words: usize,
    pub(super) output_bytes: u64,
    pub(super) launch: FrozenLinearLaunch,
    pub(super) chunk_capacity: usize,
}

#[derive(Clone, Copy)]
pub(super) struct IntegerBinaryInputGeometry<'a> {
    pub(super) shape: &'a [usize],
    pub(super) len: usize,
    pub(super) binding_bytes: u64,
}

pub(super) fn plan(
    limits: &wgpu::Limits,
    launch: FrozenLinearLaunch,
    operation_name: &str,
    integer_type: IntegerElementType,
    lhs: IntegerBinaryInputGeometry<'_>,
    rhs: IntegerBinaryInputGeometry<'_>,
) -> Result<IntegerBinaryGeometry> {
    let broadcast = integer_broadcast_plan(operation_name, lhs.shape, lhs.len, rhs.shape, rhs.len)?;
    let len = broadcast.len;
    if len > u32::MAX as usize {
        return Err(unsupported_provider_operation(
            gpu_dispatch_length_limit_error(operation_name, len).to_string(),
        ));
    }
    let output_words = integer_word_count(integer_type, len)?;
    let output_bytes = crate::backend::wgpu_storage::raw_word_requested_bytes(output_words as u64)
        .ok_or_else(|| anyhow!("{operation_name}: output backing extent overflow"))?;
    let chunk_capacity = if len == 0 {
        1
    } else {
        super::super::super::elementwise::check_kernel_limits(
            operation_name,
            limits,
            3,
            4,
            std::mem::size_of::<IntegerBinaryParams>() as u64,
            &[lhs.binding_bytes, rhs.binding_bytes, output_bytes],
            launch,
        )
        .map_err(|error| unsupported_provider_operation(error.to_string()))?
    };
    Ok(IntegerBinaryGeometry {
        broadcast,
        output_words,
        output_bytes,
        launch,
        chunk_capacity,
    })
}
