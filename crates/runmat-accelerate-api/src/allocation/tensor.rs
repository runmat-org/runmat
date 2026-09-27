//! Owning tensor views. Neither diagnostic identities nor serde mint ownership.
use super::*;
use crate::{GpuHandleProvenance, GpuTensorDescriptor};
use std::sync::{Arc, Weak};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    inner: Arc<OwnedTensorView>,
}

/// Non-owning cache reference to an existing admitted view. No new view or
/// physical lease is minted on upgrade; unavailable owners still reject use.
#[derive(Clone)]
pub struct WeakGpuTensorHandle {
    inner: Weak<OwnedTensorView>,
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
    pub fn downgrade(&self) -> WeakGpuTensorHandle {
        WeakGpuTensorHandle {
            inner: Arc::downgrade(&self.inner),
        }
    }
    /// Only an owner-admitted lease can construct an operational handle. There
    /// is deliberately no constructor taking only device/buffer identifiers.
    pub fn from_allocation(
        allocation: AllocationLease,
        view: GpuTensorView,
    ) -> Result<Self, AllocationAcquisitionError> {
        allocation.validate_live()?;
        view.validate(allocation.physical())
            .map_err(AllocationAcquisitionError::InvalidView)?;
        allocation.validate_view(&view)?;
        let identity = TensorViewIdentity {
            allocation: allocation.identity(),
            view: allocation
                .next_view_identity()
                .map_err(AllocationAcquisitionError::InvalidView)?,
        };
        Ok(Self {
            inner: Arc::new(OwnedTensorView {
                cleanup: crate::residency::ViewCleanup::new(identity),
                allocation,
                identity,
                view,
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
        Self::from_allocation(self.inner.allocation.clone(), view)
    }
    pub fn with_provenance(
        &self,
        provenance: GpuHandleProvenance,
    ) -> Result<Self, AllocationAcquisitionError> {
        self.with_view(self.view().clone().with_provenance(provenance))
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
        match Arc::try_unwrap(self.inner) {
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
