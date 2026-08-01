//! Portable transactional package-cache policy for RunMat.

pub mod backend;
pub mod config;
pub mod error;
pub mod gc;
pub mod layout;
pub mod lease;
pub mod materialize;
pub mod object;
pub mod state;

pub use backend::{
    BackendCommit, BackendSnapshot, CacheBackend, CacheClock, CacheTransaction, CommitOutcome,
    ObjectWrite,
};
pub use config::CacheConfig;
pub use error::{BackendError, CacheError};
pub use gc::{GcPlan, GcPolicy};
pub use layout::{CacheNamespace, StorageKey};
pub use lease::{Lease, LeaseId, LeaseOwner};
pub use materialize::{MaterializationRecord, MaterializationState, MountDescriptor};
pub use object::{
    BlobMetadata, CacheObject, CacheObjectKind, Pin, PinId, SourceIndexMetadata, TreeEntry,
    TreeEntryKind, TreeManifest,
};
pub use state::{
    AccessRecord, CacheState, CorruptionRecord, QuotaPressure, QuotaRecord, CACHE_SCHEMA_VERSION,
};
