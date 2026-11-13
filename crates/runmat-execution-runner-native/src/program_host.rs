//! Scheduler host requirements are derived from the compiled program, never invented.
use std::collections::BTreeSet;

use runmat_execution::host::ExecutionHostRequirement;
use runmat_execution::security::ExecutionTrustTier;
use runmat_execution_artifact::ProgramExecutionRequest;

use crate::{NativeExecutionError, NativeExecutionResult};

/// Trust admitted by an in-process or local-process session.
pub(crate) fn local_trust() -> BTreeSet<ExecutionTrustTier> {
    BTreeSet::from([ExecutionTrustTier::CustomerTrusted])
}

pub(crate) fn program_host(
    request: &ProgramExecutionRequest,
    permitted_trust_tiers: BTreeSet<ExecutionTrustTier>,
) -> NativeExecutionResult<ExecutionHostRequirement> {
    request
        .artifact
        .execution_host_requirement(&request.recipe, permitted_trust_tiers)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))
}
