use super::super::*;

pub(super) fn request(
    request: &FusionExecutionRequest<'_>,
    reduce_len: usize,
    num_slices: usize,
    output_shape: &[usize],
) -> ExecutionResult<()> {
    if std::env::var("RUNMAT_DISABLE_FUSED_REDUCTION").is_ok() {
        return Err(anyhow!("fused reduction disabled by env").into());
    }
    crate::ensure_residency_hooks();
    if !request.plan.group.kind.is_reduction() {
        return Err(anyhow!("unsupported fusion kind").into());
    }
    if !request.plan.kernel.supported {
        return Err(anyhow!("fusion kernel not supported for this plan").into());
    }
    if request.inputs.len() != request.plan.inputs.len() {
        return Err(anyhow!(
            "fusion input mismatch: expected {}, got {}",
            request.plan.inputs.len(),
            request.inputs.len()
        )
        .into());
    }
    reject_native_integer_fusion_inputs(request)?;
    let len = reduce_len
        .checked_mul(num_slices)
        .ok_or_else(|| anyhow!("fusion reduction length overflows"))?;
    if len == 0 {
        return Err(anyhow!("fusion: zero-length execution not supported").into());
    }
    let output_len = output_shape
        .iter()
        .try_fold(1usize, |length, extent| length.checked_mul(*extent))
        .ok_or_else(|| anyhow!("fusion reduction output size overflows"))?;
    if output_len != num_slices {
        return Err(anyhow!("fusion reduction output shape disagrees with selected slices").into());
    }
    Ok(())
}
