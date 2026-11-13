//! Operation-bound allocation authority for a VM frame's mutable collections.
//!
//! The frame owns its table semantics. This issuer only reuses the canonical
//! collection allocator and fences each growth to the selected host operation.

use super::{
    collection_allocator::InstalledCollectionPermit, hashbrown_table_layout,
    AdmittedCollectionAllocator, HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
    OperationCollectionAllocator, OperationPermitGate, OperationPermitGateError,
};
use crate::context::RuntimeContext;
use std::alloc::Layout;
use std::hash::Hash;

pub fn collection_block_retained_bytes(layout: Layout) -> Result<u64, HostAllocationError> {
    let block = AdmittedCollectionAllocator::requested_bytes(layout)
        .map_err(|_| HostAllocationError::Overflow)?;
    HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(block).map_err(|_| HostAllocationError::Overflow)?,
        AdmittedCollectionAllocator::retained_claim_count(),
    )
}

/// Exact physical table block plus its retained allocator claim. The layout
/// is probed from the same pinned hashbrown implementation used by the VM.
pub fn frame_hash_table_retained_bytes<K: Eq + Hash, V>(
    capacity: usize,
) -> Result<u64, HostAllocationError> {
    let Some(layout) = hashbrown_table_layout::<K, V>(capacity)? else {
        return Ok(0);
    };
    collection_block_retained_bytes(layout)
}

#[derive(Clone, Debug)]
pub struct FrameCollectionGate {
    inner: OperationPermitGate,
}

pub struct FrameCollectionPermit {
    _installed: InstalledCollectionPermit,
}

impl HostAllocationReceipt {
    /// Fixed collection blocks can allocate directly from this operation's
    /// fenced output headroom; their own headers retain the physical claims.
    pub fn collection_allocator(
        &self,
        context: &RuntimeContext,
    ) -> Result<AdmittedCollectionAllocator, HostAllocationError> {
        Ok(AdmittedCollectionAllocator::new(
            self.output_permit(context)?,
        ))
    }
}

impl FrameCollectionGate {
    pub fn retained_bytes() -> Result<u64, HostAllocationError> {
        OperationPermitGate::retained_bytes()
    }

    /// The gate itself is a retained, exactly claimed frame allocation.
    pub fn prepare(
        context: &RuntimeContext,
        receipt: &HostAllocationReceipt,
    ) -> Result<Self, HostAllocationError> {
        let permit = receipt.output_permit(context)?;
        let inner = OperationPermitGate::prepare(&permit).map_err(HostAllocationError::Numeric)?;
        receipt.checkpoint(context)?;
        Ok(Self { inner })
    }

    /// Existing unbounded contexts retain their normal allocation policy.
    pub fn unbounded() -> Self {
        Self {
            inner: OperationPermitGate::unbounded(),
        }
    }

    pub fn allocator(&self) -> OperationCollectionAllocator {
        OperationCollectionAllocator::new(self.inner.clone())
    }

    pub fn install(
        &self,
        context: &RuntimeContext,
        receipt: &HostAllocationReceipt,
    ) -> Result<FrameCollectionPermit, HostAllocationError> {
        let permit = receipt.output_permit(context)?;
        let installed = self.inner.install(permit).map_err(gate_error)?;
        receipt.checkpoint(context)?;
        Ok(FrameCollectionPermit {
            _installed: installed,
        })
    }

    pub fn install_unbounded(&self) -> Result<FrameCollectionPermit, HostAllocationError> {
        self.inner
            .install_unbounded()
            .map(|installed| FrameCollectionPermit {
                _installed: installed,
            })
            .map_err(gate_error)
    }
}

fn gate_error(_: OperationPermitGateError) -> HostAllocationError {
    // A nested or outstanding mutable frame operation is a revoked operation
    // boundary, not a capacity denial that may be retried after publication.
    HostAllocationError::StaleAuthority
}
