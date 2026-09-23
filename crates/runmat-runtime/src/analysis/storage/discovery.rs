use std::path::Path;

use super::{fs_read, fs_read_dir, owned_directory_exists, validate_run_id_token, FsFileType};

/// Discovers supported canonical artifacts without treating unrelated or retired JSON as a run.
///
/// Direct lookup remains strict and reports malformed content. Directory-wide consumers such as
/// retention and trend aggregation operate only on the supported canonical inventory, so a
/// schema-less or unknown-schema file is not allowed to poison every valid run in the directory.
pub(super) fn list_supported_artifacts<T>(
    root: &Path,
    mut decode: impl FnMut(&[u8], &str) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    let runs_dir = root.join("runs");
    if !owned_directory_exists(root, "analysis artifact root")?
        || !owned_directory_exists(&runs_dir, "analysis artifact runs directory")?
    {
        return Ok(Vec::new());
    }
    let mut artifacts = Vec::new();
    for entry in
        fs_read_dir(&runs_dir).map_err(|error| format!("failed to scan artifacts: {error}"))?
    {
        let path = entry.path().to_path_buf();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        if entry.file_type() != FsFileType::File {
            return Err(format!(
                "analysis run artifact must be a regular file: {}",
                path.display()
            ));
        }
        let run_id = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| {
                format!(
                    "analysis run artifact filename is invalid: {}",
                    path.display()
                )
            })?;
        validate_run_id_token(run_id)?;
        let bytes =
            fs_read(&path).map_err(|error| format!("failed to read run artifact: {error}"))?;
        if let Ok(artifact) = decode(&bytes, run_id) {
            artifacts.push(artifact);
        }
    }
    Ok(artifacts)
}
