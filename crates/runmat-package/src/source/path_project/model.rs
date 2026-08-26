use runmat_config::project::{ProjectManifest, ProjectSourceFile};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub(super) struct LoadedPathProject {
    pub root_manifest: PathBuf,
    pub workspace_root: PathBuf,
    pub packages: BTreeMap<PathBuf, LoadedPathPackage>,
}

pub(super) struct LoadedPathPackage {
    pub manifest_path: PathBuf,
    pub project_root: PathBuf,
    pub manifest: ProjectManifest,
    pub sources: Vec<LoadedSource>,
    pub native_interfaces: Vec<LoadedNativeInterface>,
    pub dependencies: BTreeMap<String, PathBuf>,
}

pub(super) struct LoadedSource {
    pub descriptor: ProjectSourceFile,
    pub bytes: Vec<u8>,
}

pub(super) struct LoadedNativeInterface {
    pub name: String,
    pub manifest_path: PathBuf,
    pub manifest_bytes: Vec<u8>,
    pub library_path: PathBuf,
    pub library_bytes: Vec<u8>,
}
