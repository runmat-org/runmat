//! The reference and naming system (synthesis §6).
//!
//! - [`name`]: structured names (generation names, pattern copies, CAD-file names).
//! - [`reference`]: references as written: names, relational forms, published names, the
//!   explicit geometric fallback.
//! - [`history`]: the history record contract every kernel operation emits.
//! - [`naming`]: names carried forward through one operation's history.
//! - [`resolve`]: reference + state → entity set, or a specific diagnosed reason.
//! - [`publish`]: publish/consume, the catalog seam, auto-publish.
//! - [`lock`]: stable item ids and rename aliases.
//! - [`merge`]: re-resolution after merge (redirected and broken references).

pub mod history;
pub mod lock;
pub mod merge;
pub mod name;
pub mod naming;
pub mod publish;
pub mod reference;
pub mod resolve;

pub use history::*;
pub use lock::{ItemPath, LockEntry, LockError, LockSection, SeededMinter, UuidMinter};
pub use name::*;
pub use naming::{edge_name, BodyNames, NamingContext};
pub use publish::{
    apply_publish, apply_publish_locked, auto_publish, rename_publish, AutoPublishEdit,
    AutoPublishError, ForeignEntity, ForeignPick, ForeignTarget, MemoryCatalog, PublishRenameError,
    PublishTable, PublishedCatalog, XrefProblem,
};
pub use reference::*;
pub use resolve::{
    EntityHandle, FeatureEntry, FeatureTable, MakerStatus, RefProblem, Resolution, ResolutionClass,
    ResolutionPolicy, ResolutionStats, ResolveNote, ResolveOutcome, Resolved, Resolver, WaitCause,
    WaitReason,
};
