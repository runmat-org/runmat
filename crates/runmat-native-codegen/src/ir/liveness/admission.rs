//! The real compilation owner's authority, never inferred from ambient TLS.
use runmat_runtime::context::{
    host_allocation::{HostAllocationError, HostControlAllocationOwner},
    RuntimeContext,
};
use runmat_value::ExactHostAllocationClaim;

/// Every bind/compile caller chooses explicitly. Managed callers capture the
/// original context before background dispatch; no context crosses threads.
#[derive(Clone, Debug)]
pub enum NativeLivenessAdmission {
    AuthorizedUnbounded,
    Managed(HostControlAllocationOwner),
}

impl NativeLivenessAdmission {
    pub fn capture(runtime: &RuntimeContext) -> Result<Self, HostAllocationError> {
        Ok(match HostControlAllocationOwner::capture(runtime)? {
            Some(owner) => Self::Managed(owner),
            None => Self::AuthorizedUnbounded,
        })
    }

    pub(super) fn check_current(&self) -> Result<(), crate::NativeCodegenError> {
        match self {
            Self::AuthorizedUnbounded => Ok(()),
            Self::Managed(owner) => owner.check_current().map_err(|error| {
                crate::NativeCodegenError::new(
                    "native.liveness.admission",
                    "native liveness owner is no longer current",
                )
                .with_allocation(error)
            }),
        }
    }

    pub(super) fn claim(
        &self,
        bytes: usize,
    ) -> Result<Option<ExactHostAllocationClaim>, crate::NativeCodegenError> {
        match self {
            Self::AuthorizedUnbounded => Ok(None),
            Self::Managed(owner) => owner.claim(bytes).map(Some).map_err(|error| {
                crate::NativeCodegenError::new(
                    "native.liveness.admission",
                    format!("native liveness metadata admission failed: {error}"),
                )
                .with_allocation(error)
            }),
        }
    }
}
