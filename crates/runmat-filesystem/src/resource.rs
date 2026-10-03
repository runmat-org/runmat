//! Narrow filesystem resource claims. A direct file handle certifies only
//! payload staging in the provider, never the memory used by its caller.

use crate::FsProvider;
use std::sync::Arc;

/// Revision of the implementation and resource behavior being certified.
/// A changed claim must use a new revision before placement profiles can be
/// reused; no archive codec or HTTP revision is implied by this value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Serialize)]
pub enum FsResourceRevision {
    NativeDirectFileHandlesV1,
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
}

/// Keeps the precise provider and its resource authority together across
/// suspension. Callers must perform I/O through `provider`, never the ambient
/// process-wide provider, and recheck `is_valid` before effect boundaries.
pub struct CapturedFsResourceLease {
    provider: Arc<dyn FsProvider>,
    lease: Arc<dyn FsResourceLease>,
    contract: FsResourceContract,
}

impl CapturedFsResourceLease {
    pub fn provider(&self) -> &Arc<dyn FsProvider> {
        &self.provider
    }

    pub fn contract(&self) -> FsResourceContract {
        self.contract
    }

    pub fn is_valid(&self) -> bool {
        self.lease.is_valid() && self.lease.contract() == self.contract
    }
}

pub fn capture_resource_lease(provider: Arc<dyn FsProvider>) -> Option<CapturedFsResourceLease> {
    let lease = provider.resource_lease()?;
    let contract = lease.contract();
    if !lease.is_valid() {
        return None;
    }
    Some(CapturedFsResourceLease {
        provider,
        lease,
        contract,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryFsProvider;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::NativeFsProvider;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn memory_provider_does_not_inherit_native_payload_claim() {
        let provider: Arc<dyn FsProvider> = Arc::new(MemoryFsProvider::new());
        assert!(capture_resource_lease(provider).is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_claim_retains_exact_provider_without_claiming_codec_memory() {
        let provider: Arc<dyn FsProvider> = Arc::new(NativeFsProvider);
        let captured = capture_resource_lease(provider.clone()).expect("native direct handles");
        assert!(Arc::ptr_eq(captured.provider(), &provider));
        drop(provider); // The exact provider remains alive with the lease.
        assert!(captured.is_valid());
        assert_eq!(
            captured.contract(),
            FsResourceContract {
                revision: FsResourceRevision::NativeDirectFileHandlesV1,
                opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
                opened_output_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
            }
        );
    }

    struct RevocableLease(Arc<AtomicBool>);

    impl FsResourceLease for RevocableLease {
        fn contract(&self) -> FsResourceContract {
            FsResourceContract {
                revision: FsResourceRevision::NativeDirectFileHandlesV1,
                opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
                opened_output_handle: None,
            }
        }

        fn is_valid(&self) -> bool {
            self.0.load(Ordering::Acquire)
        }
    }

    #[test]
    fn captured_lease_detects_revocation_after_suspension() {
        let valid = Arc::new(AtomicBool::new(true));
        let provider: Arc<dyn FsProvider> = Arc::new(MemoryFsProvider::new());
        let lease: Arc<dyn FsResourceLease> = Arc::new(RevocableLease(valid.clone()));
        let captured = CapturedFsResourceLease {
            provider,
            contract: lease.contract(),
            lease,
        };
        assert!(captured.is_valid());
        valid.store(false, Ordering::Release);
        assert!(!captured.is_valid());
    }
}
