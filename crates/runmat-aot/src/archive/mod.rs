mod accelerator_profile;
mod embedded;
mod manifest;
#[cfg(target_os = "windows")]
mod msvc;
mod product;
mod probe;
mod selected;
#[cfg(test)]
pub(crate) mod test_archive;

pub use accelerator_profile::read_native_accelerator_profile;
pub use embedded::embedded_runtime_archive;
pub use manifest::{
    RuntimeArchiveCapabilities, RuntimeArchiveEncoding, RuntimeArchiveManifest,
    RUNTIME_ARCHIVE_SCHEMA_VERSION,
};
pub(crate) use manifest::{MAX_RUNTIME_ARCHIVE_BYTES, MAX_RUNTIME_PAYLOAD_BYTES};
#[cfg(target_os = "windows")]
pub use msvc::{prepare_msvc_runtime_archive, MsvcImportLibrary, PreparedMsvcRuntimeArchive};
pub(crate) use probe::member_matches_probe_target;
pub use probe::read_native_abi_probe;
pub use product::{build_runtime_archive, RuntimeArchive};
pub use selected::build_selected_runtime_archive;
