mod canonical;
mod compatibility;
mod diff;
mod model;
mod validate;

pub use canonical::{decode_lock, encode_lock};
pub use compatibility::LockCompatibility;
pub use diff::{diff_locks, LockDiff};
pub use model::{
    LockSelection, LockedEdge, LockedPackage, PackageLock, RootLock, LOCK_SCHEMA_VERSION,
    RESOLVER_FORMAT_VERSION,
};
