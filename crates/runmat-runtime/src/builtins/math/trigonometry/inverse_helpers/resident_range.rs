//! Inverse-trigonometric domain checks over the shared exact-owner probe.
use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(crate) enum RealDomain {
    ClosedUnitInterval,
    AtLeastOne,
}

/// `None` is no conclusive real-domain proof. Ordinary public builtins retain
/// their host/complex fallback when a provider cannot prove this route.
pub(crate) async fn requires_complex(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    domain: RealDomain,
    builtin: gpu_helpers::RealProbeDiagnostic,
) -> BuiltinResult<Option<bool>> {
    let Some(range) = gpu_helpers::probe_real_extrema(
        provider,
        input,
        matches!(domain, RealDomain::ClosedUnitInterval),
        builtin,
    )
    .await?
    else {
        return Ok(None);
    };
    Ok(Some(match domain {
        RealDomain::AtLeastOne => range.minimum < 1.0,
        RealDomain::ClosedUnitInterval => {
            range.minimum < -1.0 || range.maximum.is_some_and(|maximum| maximum > 1.0)
        }
    }))
}
