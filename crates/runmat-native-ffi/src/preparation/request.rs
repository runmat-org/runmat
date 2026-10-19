use std::path::PathBuf;

use crate::NativeInterfaceArtifactManifest;

/// Complete host-side request for deriving a prepared native interface from
/// one already-built shared library and its public C declarations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeInterfacePreparation {
    pub interface_name: String,
    pub library_name: String,
    pub library_path: PathBuf,
    pub primary_header: PathBuf,
    pub additional_headers: Vec<PathBuf>,
    pub include_directories: Vec<PathBuf>,
    pub definitions: Vec<String>,
    pub compiler_frontend: PathBuf,
}

/// Canonical interface artifact and compiler diagnostics produced by one
/// preparation request. Publication remains an explicit caller decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedNativeInterface {
    pub manifest: NativeInterfaceArtifactManifest,
    pub warnings: String,
}
