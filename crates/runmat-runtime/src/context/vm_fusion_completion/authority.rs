use runmat_execution::{Digest, PlacementRevision};
use std::sync::Arc;

/// Stable identity of one exact, VM-validated executable fusion region.
///
/// This is process-local input to a nonpersistable placement-session authority.
/// It is not an execution permit and deliberately has no serde representation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VmFusionSemanticIdentity(Digest);

impl VmFusionSemanticIdentity {
    /// Trusted Rust-host boundary. The normal VM supplies only its canonical
    /// explicit-field fingerprint after reconciling the executable plan.
    pub fn from_validated_digest(digest: Digest) -> Self {
        Self(digest)
    }

    pub fn digest(self) -> Digest {
        self.0
    }
}

/// Session-owned, revocable authority for one installation epoch. Consumers
/// can observe retirement but cannot re-enable or clone the concrete owner.
pub trait RuntimeVmFusionGenerationPermit: Send + Sync {
    fn is_current(&self) -> bool;
}

/// Stable learning revision paired with one exact installation-epoch permit.
pub struct VmFusionSessionRegistration {
    revision: PlacementRevision,
    generation: Arc<dyn RuntimeVmFusionGenerationPermit>,
}

impl VmFusionSessionRegistration {
    pub fn new(
        revision: PlacementRevision,
        generation: Arc<dyn RuntimeVmFusionGenerationPermit>,
    ) -> Self {
        Self {
            revision,
            generation,
        }
    }

    pub fn revision(&self) -> &PlacementRevision {
        &self.revision
    }

    pub(super) fn is_current(&self) -> bool {
        self.generation.is_current()
    }

    pub(super) fn generation_permit(&self) -> Arc<dyn RuntimeVmFusionGenerationPermit> {
        Arc::clone(&self.generation)
    }

    pub(super) fn same_generation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.generation, &other.generation)
    }

    pub(super) fn feedback_lease(&self, liveness: Arc<()>) -> VmFusionFeedbackLease {
        VmFusionFeedbackLease {
            generation: Arc::clone(&self.generation),
            _liveness: liveness,
        }
    }
}

/// One already-published VM call's authority to contribute its pending receipt.
/// It keeps resources alive but does not authorize another execution.
pub struct VmFusionFeedbackLease {
    generation: Arc<dyn RuntimeVmFusionGenerationPermit>,
    _liveness: Arc<()>,
}

impl VmFusionFeedbackLease {
    pub fn is_current(&self) -> bool {
        self.generation.is_current()
    }
}
