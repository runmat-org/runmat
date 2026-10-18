//! Provider-owned allocations that must outlive an emitted submission.
//!
//! The concrete owner stays opaque to dispatch.  The operation that creates
//! it remains responsible for retaining the exact allocation leases, buffer
//! owners, and admitted host claims.  Dispatch only transfers the owner to
//! the canonical exact-submission completion path.

use super::{current_provider_observation, prepare, ProviderAdmission};
mod retirement;
use retirement::{SubmittedAllocationRetirement, SubmittedAllocationRetirementNotifier};

/// Exact provider-observation completion reserved before an operation creates
/// any allocation that must survive submission cancellation.
pub(crate) struct PreparedSubmittedAllocationRetention {
    pub(super) admission: ProviderAdmission,
    pub(super) retirement: SubmittedAllocationRetirementNotifier,
}

impl PreparedSubmittedAllocationRetention {
    pub(crate) fn reserve() -> anyhow::Result<(Self, SubmittedAllocationRetirement)> {
        let recorder = current_provider_observation().ok_or_else(|| {
            anyhow::anyhow!("submitted allocation retention requires a selected provider owner")
        })?;
        let admission = prepare(recorder);
        anyhow::ensure!(
            matches!(admission, ProviderAdmission::Prepared(_)),
            "exact provider submission completion is unavailable"
        );
        let (retirement, notifier) = retirement::retirement_pair();
        Ok((
            Self {
                admission,
                retirement: notifier,
            },
            retirement,
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) trait RetainedSubmittedAllocations: Send + 'static {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + 'static> RetainedSubmittedAllocations for T {}

#[cfg(target_arch = "wasm32")]
pub(crate) trait RetainedSubmittedAllocations: 'static {}
#[cfg(target_arch = "wasm32")]
impl<T: 'static> RetainedSubmittedAllocations for T {}

/// One opaque allocation owner transferred into exactly one physical
/// submission completion.  Dropping it before submit is safe; after submit it
/// is released only after that exact submission has settled.
pub(crate) struct SubmittedAllocationRetention {
    owner: Option<Box<dyn RetainedSubmittedAllocations>>,
    retirement: Option<SubmittedAllocationRetirementNotifier>,
}

impl SubmittedAllocationRetention {
    pub(crate) fn new<T>(owner: T) -> Self
    where
        T: RetainedSubmittedAllocations,
    {
        Self {
            owner: Some(Box::new(owner)),
            retirement: None,
        }
    }

    /// Exact request-owned heap allocation for the opaque concrete owner.
    /// The retirement notification state is included here. The concrete owner
    /// separately accounts for any allocations it owns.
    pub(crate) fn host_upper_bytes<T>() -> Option<u64> {
        let owner: u64 = std::mem::size_of::<T>()
            .checked_add(std::mem::align_of::<T>())?
            .try_into()
            .ok()?;
        owner.checked_add(retirement::retirement_host_upper_bytes()?)
    }

    pub(super) fn with_retirement(
        mut self,
        retirement: SubmittedAllocationRetirementNotifier,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.retirement.is_none(),
            "submission allocation retirement is already attached"
        );
        self.retirement = Some(retirement);
        Ok(self)
    }

    /// Release the concrete owners before acknowledging exact queue
    /// retirement.  A failed completion still resolves the waiter, but never
    /// grants permission to continue a bounded sequence.
    pub(super) fn retire(mut self, succeeded: bool) {
        drop(self.owner.take());
        if let Some(retirement) = self.retirement.take() {
            retirement.finish(succeeded);
        }
    }
}

impl std::fmt::Debug for SubmittedAllocationRetention {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("SubmittedAllocationRetention")
            .field(
                &self
                    .owner
                    .as_ref()
                    .map(|owner| std::any::type_name_of_val(owner.as_ref())),
            )
            .finish()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "submitted_retention/tests.rs"]
mod tests;
