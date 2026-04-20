use crate::execution::ExecutionProfiling;

#[cfg(not(target_arch = "wasm32"))]
use runmat_accelerate_api::provider as accel_provider;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn reset_provider_telemetry() {
    if let Some(provider) = accel_provider() {
        provider.reset_telemetry();
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn reset_provider_telemetry() {}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn gather_profiling(execution_time_ms: u64) -> Option<ExecutionProfiling> {
    let provider = accel_provider()?;
    let telemetry = provider.telemetry_snapshot();
    let gpu_ns = telemetry.fused_elementwise.total_wall_time_ns
        + telemetry.fused_reduction.total_wall_time_ns
        + telemetry.matmul.total_wall_time_ns;
    let gpu_ms = gpu_ns.saturating_div(1_000_000);
    Some(ExecutionProfiling {
        total_ms: execution_time_ms,
        cpu_ms: Some(execution_time_ms.saturating_sub(gpu_ms)),
        gpu_ms: Some(gpu_ms),
        kernel_count: Some(
            (telemetry.fused_elementwise.count
                + telemetry.fused_reduction.count
                + telemetry.matmul.count
                + telemetry.kernel_launches.len() as u64)
                .min(u32::MAX as u64) as u32,
        ),
    })
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn gather_profiling(execution_time_ms: u64) -> Option<ExecutionProfiling> {
    Some(ExecutionProfiling {
        total_ms: execution_time_ms,
        ..ExecutionProfiling::default()
    })
}
