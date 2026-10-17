//! Owning tensor views. Neither diagnostic identities nor serde mint ownership.
use super::shared::{Shared, WeakShared};
use super::*;
use crate::{GpuHandleProvenance, GpuTensorDescriptor};

/// Serializable diagnostic identity only; it cannot reconstruct a live view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct TensorViewIdentity {
    allocation: AllocationIdentity,
    view: u64,
}
impl TensorViewIdentity {
    pub fn allocation(self) -> AllocationIdentity {
        self.allocation
    }
    pub fn view(self) -> u64 {
        self.view
    }
}

struct OwnedTensorView {
    cleanup: crate::residency::ViewCleanup,
    allocation: AllocationLease,
    identity: TensorViewIdentity,
    view: GpuTensorView,
    // Declared last: refund follows the owned view and its shape storage.
    metadata_grant: Option<Box<dyn runmat_execution::resource::RetainedHostViewGrant>>,
}

// Constructor errors must drop the proposed view (including its shape Vec)
// before refunding a supplied or newly reserved metadata grant. Function
// arguments alone have the opposite drop order for these two values.
struct PendingTensorView {
    view: GpuTensorView,
    metadata_grant: Option<Box<dyn runmat_execution::resource::RetainedHostViewGrant>>,
}

/// Clone retains one shared view without copying shape or tensor payloads.
/// A new validated view retains the same physical allocation lease.
///
/// Live ownership cannot be deserialized:
/// ```compile_fail
/// let _: runmat_accelerate_api::GpuTensorHandle = serde_json::from_str("{}").unwrap();
/// ```
/// Shared views cannot be mutated through legacy public fields:
/// ```compile_fail
/// fn mutate(handle: &mut runmat_accelerate_api::GpuTensorHandle) {
///     handle.shape = vec![9];
/// }
/// ```
#[derive(Clone)]
pub struct GpuTensorHandle {
    inner: Shared<OwnedTensorView>,
}

/// Non-owning cache reference to an existing admitted view. No new view or
/// physical lease is minted on upgrade; unavailable owners still reject use.
#[derive(Clone)]
pub struct WeakGpuTensorHandle {
    inner: WeakShared<OwnedTensorView>,
    // The view grant's reservation follows the backing through the last Weak.
    _backing_claim:
        Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
}
impl WeakGpuTensorHandle {
    pub fn acquire(&self) -> Result<GpuTensorHandle, AllocationAcquisitionError> {
        let inner = self
            .inner
            .upgrade()
            .ok_or(AllocationAcquisitionError::NotLive)?;
        inner.allocation.validate_live()?;
        Ok(GpuTensorHandle { inner })
    }
}

impl GpuTensorHandle {
    /// Conservative host metadata request for one newly owned tensor view.
    /// This excludes the provider's physical allocation lease and payload,
    /// but includes the ref-counted view record and its rank-sized shape.
    /// Cloning an existing handle does not request another view record.
    pub fn view_host_upper_bytes(shape_capacity: usize) -> Option<u64> {
        use std::mem::{align_of, size_of};

        let counters = size_of::<usize>().checked_mul(2)?;
        let record = counters
            .checked_add(size_of::<OwnedTensorView>())?
            .checked_add(align_of::<OwnedTensorView>())?;
        let shape = shape_capacity.checked_mul(size_of::<usize>())?;
        u64::try_from(record.checked_add(shape)?).ok()
    }

    pub fn downgrade(&self) -> WeakGpuTensorHandle {
        WeakGpuTensorHandle {
            inner: Shared::downgrade(&self.inner),
            _backing_claim: self
                .inner
                .metadata_grant
                .as_ref()
                .map(|grant| grant.backing_claim()),
        }
    }
    /// Only an owner-admitted lease can construct an operational handle. There
    /// is deliberately no constructor taking only device/buffer identifiers.
    pub fn from_allocation(
        allocation: AllocationLease,
        view: GpuTensorView,
    ) -> Result<Self, AllocationAcquisitionError> {
        Self::from_allocation_with_grant(allocation, view, None)
    }

    /// The physical provider may attach one consumed host-view grant while
    /// publishing a newly registered allocation. The lease remains the only
    /// authority to construct an operational handle.
    pub fn from_allocation_with_retained_grant(
        allocation: AllocationLease,
        view: GpuTensorView,
        grant: Box<dyn runmat_execution::resource::RetainedHostViewGrant>,
    ) -> Result<Self, AllocationAcquisitionError> {
        Self::from_allocation_with_grant(allocation, view, Some(grant))
    }

    fn from_allocation_with_grant(
        allocation: AllocationLease,
        view: GpuTensorView,
        metadata_grant: Option<Box<dyn runmat_execution::resource::RetainedHostViewGrant>>,
    ) -> Result<Self, AllocationAcquisitionError> {
        let mut pending = PendingTensorView {
            view,
            metadata_grant,
        };
        allocation.validate_live()?;
        pending
            .view
            .validate(allocation.physical())
            .map_err(AllocationAcquisitionError::InvalidView)?;
        allocation.validate_view(&pending.view)?;
        if let Some(issuer) = allocation.view_issuer() {
            if let Some(grant) = pending.metadata_grant.as_ref() {
                if !std::sync::Arc::ptr_eq(issuer, &grant.issuer()) {
                    return Err(AllocationAcquisitionError::WrongOwner);
                }
            } else {
                let bytes = Self::view_host_upper_bytes(pending.view.shape_capacity()).ok_or(
                    AllocationAcquisitionError::RetainedHostView(
                        runmat_execution::resource::RetainedHostViewError::Overflow,
                    ),
                )?;
                pending.metadata_grant = Some(
                    issuer
                        .clone()
                        .reserve_view(bytes)
                        .map_err(AllocationAcquisitionError::RetainedHostView)?,
                );
            }
        }
        if let Some(grant) = &pending.metadata_grant {
            let required = Self::view_host_upper_bytes(pending.view.shape_capacity()).ok_or(
                AllocationAcquisitionError::RetainedHostView(
                    runmat_execution::resource::RetainedHostViewError::Overflow,
                ),
            )?;
            if grant.view_bytes() < required {
                return Err(AllocationAcquisitionError::RetainedHostView(
                    runmat_execution::resource::RetainedHostViewError::ExceedsCapacity,
                ));
            }
        }
        let identity = TensorViewIdentity {
            allocation: allocation.identity(),
            view: allocation
                .next_view_identity()
                .map_err(AllocationAcquisitionError::InvalidView)?,
        };
        let PendingTensorView {
            view,
            metadata_grant,
        } = pending;
        Ok(Self {
            inner: Shared::new(OwnedTensorView {
                cleanup: crate::residency::ViewCleanup::new(identity),
                allocation,
                identity,
                view,
                metadata_grant,
            }),
        })
    }
    pub fn allocation_identity(&self) -> AllocationIdentity {
        self.inner.allocation.identity()
    }
    pub fn view_identity(&self) -> TensorViewIdentity {
        self.inner.identity
    }
    pub fn device_id(&self) -> u32 {
        self.allocation_identity().device_id()
    }
    pub fn buffer_id(&self) -> u64 {
        self.allocation_identity().allocation_id()
    }
    pub fn shape(&self) -> &[usize] {
        self.inner.view.shape()
    }
    pub fn physical(&self) -> &GpuPhysicalDescriptor {
        self.inner.allocation.physical()
    }
    pub fn view(&self) -> &GpuTensorView {
        &self.inner.view
    }
    /// Read-only compatibility projection; it is never allocation authority.
    pub fn descriptor(&self) -> GpuTensorDescriptor {
        GpuTensorDescriptor {
            element_type: Some(self.physical().element_type()),
            storage: Some(self.physical().storage()),
            provenance: self.view().provenance(),
        }
    }
    pub fn with_view(&self, view: GpuTensorView) -> Result<Self, AllocationAcquisitionError> {
        // A derived view has its own exact claim; the parent's remains with
        // the old record. An unbounded handle retains its ordinary behavior.
        self.inner.allocation.validate_live()?;
        view.validate(self.physical())
            .map_err(AllocationAcquisitionError::InvalidView)?;
        self.inner.allocation.validate_view(&view)?;
        let grant = self
            .inner
            .metadata_grant
            .as_ref()
            .map(|parent| {
                let bytes = Self::view_host_upper_bytes(view.shape_capacity()).ok_or(
                    AllocationAcquisitionError::RetainedHostView(
                        runmat_execution::resource::RetainedHostViewError::Overflow,
                    ),
                )?;
                parent
                    .issuer()
                    .reserve_view(bytes)
                    .map_err(AllocationAcquisitionError::RetainedHostView)
            })
            .transpose()?;
        Self::from_allocation_with_grant(self.inner.allocation.clone(), view, grant)
    }
    /// Construct a new validated view whose metadata reservation remains live
    /// for exactly as long as the resulting shared view does.
    pub fn with_view_retained_grant(
        &self,
        view: GpuTensorView,
        grant: Box<dyn runmat_execution::resource::RetainedHostViewGrant>,
    ) -> Result<Self, AllocationAcquisitionError> {
        Self::from_allocation_with_grant(self.inner.allocation.clone(), view, Some(grant))
    }
    pub fn with_provenance(
        &self,
        provenance: GpuHandleProvenance,
    ) -> Result<Self, AllocationAcquisitionError> {
        if self.view().provenance() == Some(provenance) {
            self.allocation_lease().validate_live()?;
            return Ok(self.clone());
        }
        // This API constructs the derived shape itself. Reserve its metadata
        // before cloning that shape; with_view cannot do so because callers
        // have already constructed their proposed view.
        self.inner.allocation.validate_live()?;
        let grant = self
            .inner
            .metadata_grant
            .as_ref()
            .map(|parent| {
                let bytes = Self::view_host_upper_bytes(self.view().shape_capacity()).ok_or(
                    AllocationAcquisitionError::RetainedHostView(
                        runmat_execution::resource::RetainedHostViewError::Overflow,
                    ),
                )?;
                parent
                    .issuer()
                    .reserve_view(bytes)
                    .map_err(AllocationAcquisitionError::RetainedHostView)
            })
            .transpose()?;
        let view = self.view().clone().with_provenance(provenance);
        Self::from_allocation_with_grant(self.inner.allocation.clone(), view, grant)
    }
    pub fn shares_allocation_with(&self, other: &Self) -> bool {
        self.inner
            .allocation
            .shares_allocation_with(&other.inner.allocation)
    }
    pub fn allocation_lease(&self) -> &AllocationLease {
        &self.inner.allocation
    }
    pub fn wire_descriptor(&self) -> GpuTensorWireDescriptor {
        GpuTensorWireDescriptor {
            allocation: self.allocation_identity(),
            physical: *self.physical(),
            view: self.view().clone(),
        }
    }
    pub fn release(self) -> Result<AllocationReleaseOutcome, GpuHandleReleaseError> {
        match Shared::try_unwrap(self.inner) {
            Err(view) => {
                drop(view);
                Ok(AllocationReleaseOutcome::ReferenceReleased)
            }
            Ok(view) => {
                let OwnedTensorView {
                    mut cleanup,
                    allocation,
                    ..
                } = view;
                let residency = cleanup.finish().err();
                let physical = allocation.release();
                match (residency, physical) {
                    (None, Ok(outcome)) => Ok(outcome),
                    (residency, physical) => Err(GpuHandleReleaseError {
                        residency,
                        physical,
                    }),
                }
            }
        }
    }
}
impl PartialEq for GpuTensorHandle {
    fn eq(&self, other: &Self) -> bool {
        self.shares_allocation_with(other) && self.view() == other.view()
    }
}
impl Eq for GpuTensorHandle {}
impl std::fmt::Debug for GpuTensorHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTensorHandle")
            .field("identity", &self.view_identity())
            .field("physical", self.physical())
            .field("view", self.view())
            .finish()
    }
}
impl serde::Serialize for GpuTensorHandle {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&self.wire_descriptor(), serializer)
    }
}
