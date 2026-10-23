//! Durable physical native installation, not execution or service authority.
//!
//! A receipt is issued only after exact bytes are materialized and published.
//! Loader observations below are retained obligations for subsequent target
//! execution qualification, not an assertion that an OS loader resolved them.
mod admission;
mod assemble;
mod layout;
mod loader;
mod materialize;

pub use crate::tree::{HostedNativeTree, NativeStorageGrant};
pub use admission::{
    issue_completed_installation_admission, ExternalNativeInstallationAdmission,
    ExternalNativeInstallationIntent, NativeInstallationAdmissionRecord,
    NATIVE_INSTALLATION_SCHEMA_VERSION,
};
pub use assemble::{
    prepare_native_installation, NativeInstallationBundle, NativeInstallationManifest,
};
pub use layout::{NativeInstallationLayout, NativeInstalledComponent};
pub use loader::{NativeLoaderImage, NativeLoaderReference, NativeLoaderReferenceKind};
pub use materialize::{
    install_native_bundle, open_native_installation, InstalledNativeArtifact,
    NativeInstallationRequest,
};

fn invalid(error: impl std::fmt::Display) -> crate::AotError {
    crate::AotError::contract("aot.installation", error.to_string())
}
