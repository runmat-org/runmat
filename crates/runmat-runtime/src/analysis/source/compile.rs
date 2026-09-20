use std::path::{Path, PathBuf};

use runmat_analysis_language::{
    CheckRequest, CheckedFeaProgram, FeaCompiler, LogicalSourcePath, ModuleGraphRequest,
};

use super::FilesystemFeaHost;

pub fn check_fea_path(path: &Path) -> Result<CheckedFeaProgram, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    check_fea_path_in_root(parent, path)
}

pub fn check_fea_path_in_root(
    source_root: &Path,
    path: &Path,
) -> Result<CheckedFeaProgram, String> {
    let host = FilesystemFeaHost::new(source_root)?;
    check_with_host(&host, source_root, path)
}

/// Check the supplied source as one module in a filesystem-backed project graph.
///
/// The supplied bytes replace only `path`; imports and referenced geometry continue to resolve
/// below `source_root`. This is the canonical entrypoint for editor buffers that have not been
/// saved yet.
pub fn check_fea_source_in_root(
    source_root: &Path,
    path: &Path,
    source: impl Into<Vec<u8>>,
) -> Result<CheckedFeaProgram, String> {
    let physical_path = entry_path(source_root, path);
    let host = FilesystemFeaHost::new(source_root)?.with_source_overlay(&physical_path, source)?;
    check_with_host(&host, source_root, &physical_path)
}

fn check_with_host(
    host: &FilesystemFeaHost,
    source_root: &Path,
    path: &Path,
) -> Result<CheckedFeaProgram, String> {
    let physical_path = entry_path(source_root, path);
    let root = host.load_physical_module(&physical_path)?;
    Ok(FeaCompiler::new(host).check(CheckRequest {
        modules: ModuleGraphRequest {
            source_root: LogicalSourcePath::new(host.logical_root().to_string())
                .map_err(|error| error.to_string())?,
            root,
        },
    }))
}

fn entry_path(source_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() || path.starts_with(source_root) {
        path.to_path_buf()
    } else {
        source_root.join(path)
    }
}
