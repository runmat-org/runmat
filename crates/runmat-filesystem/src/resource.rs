//! Narrow filesystem resource claims. A direct file handle certifies only
//! payload staging in the provider, never the memory used by its caller.

use crate::FsProvider;
use sha2::{Digest as _, Sha256};
use std::fmt::{Display, Formatter};
use std::sync::Arc;

mod bounded_input;
pub use bounded_input::FsBoundedInput;

/// Requested Rust-managed heap bytes for one provider-owned opened input,
/// excluding caller buffers, path capture, OS/kernel allocations, and RSS.
/// Providers may advertise this only when every allocation in their direct
/// opened-handle route has a proven bound and revisioned implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct FsManagedOpenedInputCost {
    pub requested_heap_bytes: u64,
}

/// Pure pre-effect bound for allocations made by this provider's bounded
/// input route. The first amount covers captured lease construction; the
/// second covers opening the exact handle. Both are requested Rust-managed
/// heap bytes, not allocator overhead, OS memory, caller path/codec buffers,
/// or a complete-call resource certificate. The revision must match the
/// retained [`FsResourceLease`] and opened handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct FsManagedReadContract {
    pub revision: FsResourceRevision,
    /// Ownership of source bytes that remain live behind the direct handle.
    /// This is distinct from the per-open handle's requested allocation.
    pub backing: FsManagedReadBacking,
    pub capture_requested_heap_bytes: u64,
    pub opened_requested_heap_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum FsManagedReadBacking {
    /// File contents live outside the process's Rust-managed heap. This does
    /// not make OS caches, allocator overhead, or total RSS finite evidence.
    ExternalToRustManagedHeap,
    /// A provider-owned managed backing survives the opened handle. Callers
    /// need a same-ledger retention and mutation-copy certificate before a
    /// complete managed-byte admission; handle size alone is insufficient.
    RetainedManagedPayloadUnaccounted,
}

/// Revision of the implementation and resource behavior being certified.
/// A changed claim must use a new revision before placement profiles can be
/// reused; no archive codec or HTTP revision is implied by this value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FsResourceRevision([u8; 32]);

impl FsResourceRevision {
    /// `authority` identifies the provider implementation/configuration owner;
    /// `revision_material` changes whenever its resource claim changes. The
    /// length-delimited, domain-separated encoding prevents ambiguous pairs.
    pub fn for_provider(authority: &str, revision_material: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"runmat.filesystem.resource-revision.v1\0");
        hasher.update((authority.len() as u64).to_le_bytes());
        hasher.update(authority.as_bytes());
        hasher.update((revision_material.len() as u64).to_le_bytes());
        hasher.update(revision_material);
        Self(hasher.finalize().into())
    }

    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Display for FsResourceRevision {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("sha256:")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl serde::Serialize for FsResourceRevision {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

/// The provider reads/writes the supplied slice directly, without retaining
/// or constructing a payload-sized user-space copy. Fixed handle/OS overhead,
/// path processing, codecs, and caller-owned buffers remain outside this
/// claim and require separate bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum FsFilePayloadContract {
    CallerOwnedBuffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct FsResourceContract {
    pub revision: FsResourceRevision,
    pub opened_input_handle: Option<FsFilePayloadContract>,
    pub opened_output_handle: Option<FsFilePayloadContract>,
}

/// An opt-in provider may retain a configuration guard here. `is_valid` must
/// become false when its claim can no longer govern calls on that provider.
/// A mutable provider must synchronize this check with its own operations;
/// a point-in-time revision check alone is not a lease.
pub trait FsResourceLease: Send + Sync + 'static {
    fn contract(&self) -> FsResourceContract;
    fn is_valid(&self) -> bool;

    /// An absent certificate is not a zero-cost opened handle. This bound is
    /// for the selected provider's managed heap route only, not a process
    /// physical-memory or codec certificate.
    fn managed_opened_input_cost(&self) -> Option<FsManagedOpenedInputCost> {
        None
    }
}

/// A provider may expose an immutable static lease without allocating an Arc
/// for every opened source. Mutable authorities retain their owned Arc.
#[derive(Clone)]
pub enum FsResourceLeaseHandle {
    Static(&'static dyn FsResourceLease),
    Owned(Arc<dyn FsResourceLease>),
}

impl FsResourceLeaseHandle {
    pub fn lease(&self) -> &dyn FsResourceLease {
        match self {
            Self::Static(lease) => *lease,
            Self::Owned(lease) => lease.as_ref(),
        }
    }
}

/// Keeps the precise provider and its resource authority together across
/// suspension. Callers must perform I/O through `provider`, never the ambient
/// process-wide provider, and recheck `is_valid` before effect boundaries.
pub struct CapturedFsResourceLease {
    provider: Arc<dyn FsProvider>,
    lease: FsResourceLeaseHandle,
    contract: FsResourceContract,
}

impl CapturedFsResourceLease {
    pub fn provider(&self) -> &Arc<dyn FsProvider> {
        &self.provider
    }

    pub fn contract(&self) -> FsResourceContract {
        self.contract
    }

    pub fn managed_opened_input_cost(&self) -> Option<FsManagedOpenedInputCost> {
        self.is_valid()
            .then(|| self.lease.lease().managed_opened_input_cost())
            .flatten()
    }

    pub fn is_valid(&self) -> bool {
        self.lease.lease().is_valid() && self.lease.lease().contract() == self.contract
    }
}

pub fn capture_resource_lease(provider: Arc<dyn FsProvider>) -> Option<CapturedFsResourceLease> {
    let lease = provider.resource_lease_handle()?;
    let contract = lease.lease().contract();
    if !lease.lease().is_valid() {
        return None;
    }
    Some(CapturedFsResourceLease {
        provider,
        lease,
        contract,
    })
}

pub fn capture_resource_lease_for_effect(
    provider: Arc<dyn FsProvider>,
) -> std::io::Result<Option<CapturedFsResourceLease>> {
    let Some(lease) = provider.resource_lease_handle() else {
        return Ok(None);
    };
    let contract = lease.lease().contract();
    if !lease.lease().is_valid() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "filesystem resource authority was revoked",
        ));
    }
    Ok(Some(CapturedFsResourceLease {
        provider,
        lease,
        contract,
    }))
}

#[cfg(test)]
mod tests;
