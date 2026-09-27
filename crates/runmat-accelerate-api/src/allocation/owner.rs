use super::{AllocationIdentity, WeakAllocationLease};

/// Native leases may cross threads. Browser-only owners need not assert
/// thread safety for JS-backed resources; their leases remain non-Send.
#[cfg(not(target_arch = "wasm32"))]
pub trait AllocationOwnerThreadBound: Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync + ?Sized> AllocationOwnerThreadBound for T {}

#[cfg(target_arch = "wasm32")]
pub trait AllocationOwnerThreadBound {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> AllocationOwnerThreadBound for T {}

/// Registration fails without granting a second lifetime authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationRegistrationError {
    UnknownAllocation,
    WrongOwner,
    StaleGeneration,
    AlreadyRegistered,
    NotLive,
}

/// Physical-retirement result recorded by the owning allocation state.
/// This is not a provider execution-completion observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocationRetirementOutcome {
    /// Physical retirement completed; metadata/cache cleanup may complete.
    Retired,
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
pub trait AllocationOwner: AllocationOwnerThreadBound + 'static {
    /// Under the existing map's lock, validate identity/generation/live payload
    /// and install exactly one weak lease. Reject a duplicate even if its old
    /// weak lease is no longer acquirable. No retired entry may be resurrected.
    /// On Err leave the previous record unchanged. Do not invoke external
    /// callbacks or hold a map lock beyond this call.
    ///
    /// The new weak lease cannot acquire until registration returns Ok. This
    /// prevents publishing partially admitted ownership to another caller.
    fn register_lease(
        &self,
        identity: AllocationIdentity,
        lease: WeakAllocationLease,
    ) -> Result<(), AllocationRegistrationError>;

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
