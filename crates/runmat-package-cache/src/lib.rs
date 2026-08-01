//! Portable transactional package-cache policy for RunMat.

pub mod archive;
pub mod backend;
pub mod config;
pub mod error;
pub mod gc;
pub mod layout;
pub mod lease;
pub mod materialize;
pub mod object;
pub mod source;
pub mod state;

pub use archive::{
    normalize_link_for_entry, validate_archive, ArchiveEntryHeader, ArchiveEntryKind, ArchiveError,
    ArchiveLimits, ValidatedArchive, ValidatedArchiveEntry,
};
pub use backend::{
    BackendCommit, BackendSnapshot, CacheBackend, CacheClock, CacheTransaction, CommitOutcome,
    ObjectWrite,
};
pub use config::CacheConfig;
pub use error::{BackendError, CacheError};
pub use gc::{execute_gc, GcPlan, GcPolicy};
pub use layout::{CacheNamespace, StorageKey};
pub use lease::{acquire_lease, release_lease, renew_lease, Lease, LeaseId, LeaseOwner};
pub use materialize::{MaterializationRecord, MaterializationState, MountDescriptor};
pub use object::{
    BlobMetadata, CacheObject, CacheObjectKind, Pin, PinId, SourceIndexMetadata, TreeEntry,
    TreeEntryKind, TreeManifest,
};
pub use source::{
    cache_git_snapshot, cache_server_project_snapshot, cache_source_inventory, load_git_snapshot,
    load_server_project_snapshot, load_source_inventory, publish_source_inventory,
    GitInventoryEntry, GitInventoryEntryKind, GitSnapshot, GitTreeInventory, ServerProjectSnapshot,
    ServerProjectTreeInventory, SnapshotBlob, TreeInventoryEntry, TreeInventoryEntryKind,
};
pub use state::{
    AccessRecord, CacheState, CacheStatus, CorruptionRecord, QuotaPressure, QuotaRecord,
    CACHE_SCHEMA_VERSION,
};
