//! Normalize only constructor payloads at the host-owned distributed boundary.

use runmat_runtime::{execution::RootedValueLease, RuntimeError};
use runmat_value::Value;

/// Automatic placement is an implementation detail, not a change to the
/// compiler-owned Dense storage contract. Preserve explicit device intent and
/// use the canonical exact typed gather, including its admission/cancellation
/// rules. The returned root spans all subsequent coordination and publication.
pub(super) async fn prepare(input: Value) -> Result<(Value, RootedValueLease), RuntimeError> {
    if runmat_runtime::dispatcher::value_contains_explicit_gpu(&input) {
        return Err(crate::interpreter::errors::mex(
            "DistributedExplicitDeviceIntent",
            "distributed construction requires host data; explicitly gather gpuArray input first",
        ));
    }
    let input = runmat_runtime::dispatcher::gather_if_needed_owned_async(input, None).await?;
    let roots = RootedValueLease::for_values(std::iter::once(&input))
        .map_err(|error| error.into_runtime_error())?;
    Ok((input, roots))
}

#[cfg(all(test, feature = "native-accel"))]
mod tests;
