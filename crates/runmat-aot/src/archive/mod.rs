mod embedded;
mod manifest;
#[cfg(target_os = "windows")]
mod msvc;
mod product;

pub use embedded::embedded_runtime_archive;
pub use manifest::{
    RuntimeArchiveCapabilities, RuntimeArchiveEncoding, RuntimeArchiveManifest,
    RUNTIME_ARCHIVE_SCHEMA_VERSION,
};
pub(crate) use manifest::{MAX_RUNTIME_ARCHIVE_BYTES, MAX_RUNTIME_PAYLOAD_BYTES};
#[cfg(target_os = "windows")]
pub use msvc::{prepare_msvc_runtime_archive, PreparedMsvcRuntimeArchive};
pub use product::{build_runtime_archive, RuntimeArchive};
