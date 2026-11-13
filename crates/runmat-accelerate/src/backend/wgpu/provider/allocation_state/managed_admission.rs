//! Selected WGPU registration-slot admission, separate from ordinary output
//! publication. The table backing and every slot are reserved pre-effect.

use super::*;
use runmat_execution::resource::{
    ProviderHostMetadataAdmission, ProviderHostMetadataError, ProviderRegistrationSlot,
    ProviderRegistrationSlotPool, RetainedHostAllocationClaim, RetainedHostViewGrant,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct WgpuManagedAdmission {
    provider_owner: usize,
    allocations: super::super::owner_shared::OwnerShared<AllocationState>,
    underlying: Box<dyn ProviderHostMetadataAdmission>,
    outstanding: Arc<AtomicUsize>,
    remaining: AtomicUsize,
    reusable: Option<ProviderRegistrationSlotPool>,
}

impl std::fmt::Debug for WgpuManagedAdmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuManagedAdmission")
            .field("provider_owner", &self.provider_owner)
            .field("remaining", &self.remaining.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl Drop for WgpuManagedAdmission {
    fn drop(&mut self) {
        drop(self.reusable.take());
        let remaining = self.remaining.swap(0, Ordering::AcqRel);
        self.outstanding.fetch_sub(remaining, Ordering::AcqRel);
        self.allocations.release_empty_managed();
    }
}

impl ProviderHostMetadataAdmission for WgpuManagedAdmission {
    fn claim_registration_slot(
        &self,
        provider_owner: usize,
    ) -> Result<Option<ProviderRegistrationSlot>, ProviderHostMetadataError> {
        if let Some(pool) = &self.reusable {
            pool.acquire(provider_owner).map(Some)
        } else {
            self.commit_registration_slot(provider_owner)?;
            Ok(None)
        }
    }

    fn claim_view(
        &self,
        view_bytes: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, ProviderHostMetadataError> {
        self.underlying.claim_view(view_bytes)
    }

    fn claim_record(
        &self,
        record_bytes: u64,
    ) -> Result<Arc<dyn RetainedHostAllocationClaim>, ProviderHostMetadataError> {
        self.underlying.claim_record(record_bytes)
    }

    fn commit_registration_slot(
        &self,
        provider_owner: usize,
    ) -> Result<(), ProviderHostMetadataError> {
        if self.provider_owner != provider_owner || self.reusable.is_some() {
            return Err(ProviderHostMetadataError::OwnerFailure);
        }
        self.remaining
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_sub(1)
            })
            .map_err(|_| ProviderHostMetadataError::OwnerFailure)?;
        self.outstanding.fetch_sub(1, Ordering::AcqRel);
        Ok(())
    }
}

impl AllocationState {
    pub(in super::super) fn managed_preparation_host_bytes() -> Option<u64> {
        let bytes = std::mem::size_of::<WgpuManagedAdmission>()
            .checked_add(std::mem::align_of::<WgpuManagedAdmission>())?;
        u64::try_from(bytes).ok()
    }

    fn release_empty_managed(&self) {
        let detached = self
            .lock_records()
            .ok()
            .and_then(|mut table| table.detach_empty_managed());
        drop(detached);
    }

    pub(in super::super) fn managed_table_upper_bytes(&self, count: usize) -> Option<u64> {
        self.lock_records().ok()?.managed_backing_upper(count)
    }

    pub(in super::super) fn prepare_managed_metadata(
        self: &super::super::owner_shared::OwnerShared<Self>,
        underlying: Box<dyn ProviderHostMetadataAdmission>,
        count: usize,
    ) -> Result<Box<dyn ProviderHostMetadataAdmission>> {
        self.prepare_metadata(underlying, count, RegistrationLifetime::OneShot)
    }

    pub(in super::super) fn prepare_reusable_managed_metadata(
        self: &super::super::owner_shared::OwnerShared<Self>,
        underlying: Box<dyn ProviderHostMetadataAdmission>,
        count: usize,
    ) -> Result<Box<dyn ProviderHostMetadataAdmission>> {
        self.prepare_metadata(underlying, count, RegistrationLifetime::Concurrent)
    }

    fn prepare_metadata(
        self: &super::super::owner_shared::OwnerShared<Self>,
        underlying: Box<dyn ProviderHostMetadataAdmission>,
        count: usize,
        lifetime: RegistrationLifetime,
    ) -> Result<Box<dyn ProviderHostMetadataAdmission>> {
        if count == 0 || self.is_lost() {
            return Err(AllocationStateError::Capacity.into());
        }
        let pool_funding = match lifetime {
            RegistrationLifetime::OneShot => None,
            RegistrationLifetime::Concurrent => Some(
                underlying.claim_record(
                    ProviderRegistrationSlotPool::requested_host_bytes()
                        .ok_or(AllocationStateError::Capacity)?,
                )?,
            ),
        };
        let upper = self
            .managed_table_upper_bytes(count)
            .ok_or(AllocationStateError::Capacity)?;
        // The bound is acquired before hashbrown can allocate a new table.
        let backing_claim = if upper == 0 {
            None
        } else {
            Some(underlying.claim_record(upper)?)
        };
        let (outstanding, retired_claim) =
            self.lock_records()?
                .reserve_managed(count, upper, &backing_claim)?;
        drop(backing_claim);
        drop(retired_claim);
        let owner = (&**self as *const Self) as usize;
        let reusable = match pool_funding {
            Some(funding) => match ProviderRegistrationSlotPool::from_reserved_capacity(
                owner,
                count,
                outstanding.clone(),
                funding,
            ) {
                Ok(pool) => Some(pool),
                Err(error) => {
                    outstanding.fetch_sub(count, Ordering::AcqRel);
                    self.release_empty_managed();
                    return Err(error.into());
                }
            },
            None => None,
        };
        Ok(Box::new(WgpuManagedAdmission {
            provider_owner: owner,
            allocations: self.clone(),
            underlying,
            outstanding,
            remaining: AtomicUsize::new(if reusable.is_some() { 0 } else { count }),
            reusable,
        }))
    }
}

#[derive(Clone, Copy)]
enum RegistrationLifetime {
    OneShot,
    Concurrent,
}
