mod embedded;
mod manifest;
mod product;

pub use embedded::embedded_runtime_archive;
pub use manifest::{
    RuntimeArchiveEncoding, RuntimeArchiveManifest, RUNTIME_ARCHIVE_SCHEMA_VERSION,
};
pub(crate) use manifest::{MAX_RUNTIME_ARCHIVE_BYTES, MAX_RUNTIME_PAYLOAD_BYTES};
pub use product::{build_runtime_archive, RuntimeArchive};
