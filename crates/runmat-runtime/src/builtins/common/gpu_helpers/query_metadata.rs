//! Checked request-owned feasibility metadata before its Vec/String storage
//! is constructed. The receipt covers only the query/preparation lifetime;
//! provider records and published outputs have separate retained owners.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::ProviderQueryMetadataFootprint;

pub(crate) fn requested_bytes(
    label: &'static str,
    footprint: &ProviderQueryMetadataFootprint<'_>,
) -> BuiltinResult<u64> {
    footprint.requested_bytes().ok_or_else(|| {
        build_runtime_error(format!("{label}: query metadata extent overflow"))
            .with_builtin(label)
            .build()
    })
}

pub(crate) fn admit(
    runtime: Option<&RuntimeContext>,
    bytes: u64,
) -> Result<Option<HostAllocationReceipt>, HostAllocationError> {
    HostAllocationPlan::new(bytes, 0).and_then(|plan| HostAllocationReceipt::admit(runtime, plan))
}

pub(crate) fn admission_error(
    label: &'static str,
    runtime: Option<&RuntimeContext>,
    source: HostAllocationError,
) -> crate::RuntimeError {
    let suffix = source.runtime_identifier_suffix();
    let namespace = runtime
        .map(RuntimeContext::error_namespace)
        .unwrap_or_else(|| "RunMat".to_string());
    build_runtime_error(format!("{label}: query metadata admission: {source}"))
        .with_builtin(label)
        .with_identifier(format!("{namespace}:{suffix}"))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
}
