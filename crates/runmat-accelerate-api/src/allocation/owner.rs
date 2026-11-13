use super::{AllocationIdentity, GpuPhysicalDescriptor, WeakAllocationLease};

/// Exact provider allocation authority: worker-local in browsers, thread-safe
/// on native targets. It is captured by leases and failed-retirement records.
#[cfg(not(target_arch = "wasm32"))]
pub type AllocationOwnerHandle = std::sync::Arc<dyn AllocationOwner>;
#[cfg(target_arch = "wasm32")]
pub type AllocationOwnerHandle = std::rc::Rc<dyn AllocationOwner>;

/// Registration fails without granting a second lifetime authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationRegistrationError {
    UnknownAllocation,
    WrongOwner,
    StaleGeneration,
    AlreadyRegistered,
    NotLive,
}

impl std::fmt::Display for AllocationRegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::UnknownAllocation => "allocation is unknown",
            Self::WrongOwner => "allocation belongs to another owner",
            Self::StaleGeneration => "allocation generation is stale",
            Self::AlreadyRegistered => "allocation already has a registered lease",
            Self::NotLive => "allocation is not live",
        })
    }
}

impl std::error::Error for AllocationRegistrationError {}

/// Physical-retirement result recorded by the owning allocation state.
/// This is not a provider execution-completion observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocationRetirementOutcome {
    /// Physical retirement completed; metadata/cache cleanup may complete.
    Retired,
    /// Payload is known retained, but no ordinary retry transition is granted.
    /// The existing owner retains failure bookkeeping until its explicit drain
    /// or teardown. This is neither an uncertain free nor device-loss proof.
    RetainedFailure { detail: String },
    /// Backend proves no release occurred and an explicit later retry is safe.
    /// The owner keeps the payload and failure record reachable.
    RetryableFailure { detail: String },
    /// Backend cannot prove whether release occurred. Never retry blindly,
    /// reacquire the allocation, or return it to a reusable pool.
    Uncertain { detail: String },
    /// Device/context loss makes the allocation unusable. This does not claim
    /// that a physical free succeeded; backend teardown owns remaining work.
    DeviceLost { detail: String },
}

/// Captured authority for the provider's existing allocation map.
///
/// Implementors must retain failed-retirement records through their existing
/// provider-state management independently of leases. The map holds only weak
/// leases; a state/lease strong-reference cycle is forbidden. Registered product
/// providers are process-retained; custom fallible owners must provide an
/// equally real drain/teardown lifetime, not an otherwise unreachable record.
/// Owner destructors and panic-recording implementations must not unwind.
pub trait AllocationOwner: crate::AcceleratorThreadBound + 'static {
    /// Under the existing map's lock, validate identity/generation/live payload
    /// and install exactly one weak lease. Reject a duplicate even if its old
    /// weak lease is no longer acquirable. No retired entry may be resurrected.
    /// On Err or unwind leave the previous record and payload unchanged. Perform
    /// all validation/fallible work first, then install the association and
    /// immediately return Ok. Do not run fallible work or external callbacks
    /// after installation. If a future backend needs such work, its existing
    /// owner must supply rollback before it can implement this contract. An
    /// unacquirable weak entry is not permission to strand the payload or reject
    /// every future registration attempt. Poison recovery retains this invariant.
    /// Do not hold a map lock beyond this call.
    ///
    /// The new weak lease cannot acquire until registration returns Ok. This
    /// prevents publishing partially admitted ownership to another caller.
    /// Return the validated immutable physical descriptor from the payload
    /// record. The lease installs it before making the weak association live;
    /// callers cannot supply replacement physical facts through a tensor view.
    fn register_lease(
        &self,
        identity: AllocationIdentity,
        lease: WeakAllocationLease,
    ) -> Result<GpuPhysicalDescriptor, AllocationRegistrationError>;

    /// Validate the existing payload phase and owner availability. This cannot
    /// grant a new lease, revive retirement, or promise future device health.
    fn validate_live(
        &self,
        identity: AllocationIdentity,
    ) -> Result<(), super::AllocationAcquisitionError>;

    /// Admit the requested interpretation using the existing allocation/view
    /// record and trusted producer facts. Structural capacity validation alone
    /// does not establish an admitted logical type. A numeric allocation cannot
    /// acquire logical classification from a caller or deserialized flag;
    /// only an already admitted producer result (or lawful transformation of
    /// that result) supplies that evidence. Native ABI uninitialized logical
    /// allocations carry class intent, not proof of initialized 0/1 values.
    /// Do not scan or download payloads
    /// here. Reject interpretations lacking existing evidence.
    ///
    /// The lease stays strong throughout this call. As with live validation,
    /// release any map lock before returning or invoking external code.
    fn validate_view(
        &self,
        identity: AllocationIdentity,
        view: &super::GpuTensorView,
    ) -> Result<(), super::AllocationAcquisitionError>;

    /// Called exactly once by the final strong lease, without a primitive lock.
    /// Before backend work, mark the existing entry retiring/unacquirable.
    /// Release its map lock before driver/cache/destructor callbacks. Persist
    /// the resulting disposition in that same owner before returning; errors
    /// are not silently discarded just because Drop cannot return them.
    /// Retryable/uncertain/lost records must retain their required resources.
    /// Never route through ambient selection or another provider's free hook.
    fn retire(&self, identity: AllocationIdentity) -> AllocationRetirementOutcome;

    /// Recover bookkeeping after an unexpected unwind from retire. Record an
    /// uncertain disposition without reissuing physical release. Must not
    /// panic; on panic-abort targets no unwind recovery can be promised.
    /// This separate method is required, not a default that loses the failure.
    fn record_retirement_panic(&self, identity: AllocationIdentity);
}
