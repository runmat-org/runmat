mod collisions;
mod limits;
mod paths;
mod reader;
mod validate;

pub use limits::ArchiveLimits;
pub use paths::relative_link_target;
pub use reader::{ArchiveEntryHeader, ArchiveEntryKind, ArchiveHeaderReader};
pub use validate::{
    normalize_link_for_entry, validate_archive, ArchiveError, ValidatedArchive,
    ValidatedArchiveEntry,
};
