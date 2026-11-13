use super::{AllocationIdentity, AllocationOwnerHandle, AllocationRetirementOutcome};

/// Both terminal phases are attempted once. A view failure never suppresses
/// physical release, and a physical failure retains its exact owner.
#[derive(Debug)]
pub struct GpuHandleReleaseError {
    pub residency: Option<crate::ResidencyCleanupError>,
    pub physical: Result<AllocationReleaseOutcome, AllocationReleaseError>,
}
impl std::fmt::Display for GpuHandleReleaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU handle release failed: residency {:?}, physical {:?}",
            self.residency, self.physical
        )
    }
}
impl std::error::Error for GpuHandleReleaseError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationReleaseOutcome {
    /// This reference was released. Concurrent owners may already have changed;
    /// no assertion about current alias count or physical completion is made.
    ReferenceReleased,
    Retired,
}

/// Failed final retirement retains the exact owner and its persisted record.
/// It cannot reacquire a live lease or blindly retry an uncertain driver free.
pub struct AllocationReleaseError {
    identity: AllocationIdentity,
    outcome: AllocationRetirementOutcome,
    owner: AllocationOwnerHandle,
}
impl AllocationReleaseError {
    pub(super) fn new(
        identity: AllocationIdentity,
        outcome: AllocationRetirementOutcome,
        owner: AllocationOwnerHandle,
    ) -> Self {
        Self {
            identity,
            outcome,
            owner,
        }
    }
    pub fn identity(&self) -> AllocationIdentity {
        self.identity
    }
    pub fn outcome(&self) -> &AllocationRetirementOutcome {
        &self.outcome
    }
    /// Retain the same state while the embedding owner reports/drains failures.
    /// This is not a second allocation registry or a renewed allocation lease.
    pub fn retained_owner(&self) -> &AllocationOwnerHandle {
        &self.owner
    }
}
impl std::fmt::Debug for AllocationReleaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AllocationReleaseError")
            .field("identity", &self.identity)
            .field("outcome", &self.outcome)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Display for AllocationReleaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "allocation {:?} retirement failed: {:?}",
            self.identity, self.outcome
        )
    }
}
impl std::error::Error for AllocationReleaseError {}
