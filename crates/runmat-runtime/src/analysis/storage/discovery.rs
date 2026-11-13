use super::{
    access::FilesystemAccess, validate_run_id_token, AnalysisPublicationError, FsFileType,
};

/// Discover supported canonical artifacts without treating unrelated or retired JSON as runs.
/// Unknown schemas are skipped; consumer failures remain fatal.
/// Decode and consume one original artifact at a time; callers need not retain fields.
#[cfg(test)]
pub(super) fn visit_supported_artifacts<T>(
    store: &FilesystemAccess<'_>,
    decode: impl FnMut(&[u8], &str) -> Result<T, String>,
    mut visit: impl FnMut(T) -> Result<(), String>,
) -> Result<(), String> {
    visit_supported_artifacts_admitted(
        store,
        |_| Ok(()),
        decode,
        |value| visit(value).map_err(Into::into),
    )
    .map_err(|e| e.to_string())
}

/// Admission failures are fatal; only schema decoding may skip retired artifacts.
/// The admitted guard remains alive through decoding and consumer execution.
pub(super) fn visit_supported_artifacts_admitted<T, G>(
    store: &FilesystemAccess<'_>,
    mut admit: impl FnMut(&[u8]) -> Result<G, AnalysisPublicationError>,
    mut decode: impl FnMut(&[u8], &str) -> Result<T, String>,
    mut visit: impl FnMut(T) -> Result<(), AnalysisPublicationError>,
) -> Result<(), AnalysisPublicationError> {
    let root = &store.root;
    let runs_dir = root.join("runs");
    if !store.owned_directory_exists(root, "analysis artifact root")?
        || !store.owned_directory_exists(&runs_dir, "analysis artifact runs directory")?
    {
        return Ok(());
    }
    let entries = store.fs_read_dir(&runs_dir).map_err(|error| {
        AnalysisPublicationError::from_io(error).context("failed to scan artifacts")
    })?;
    let _directory_reservation = store.retain_directory(&entries)?;
    for entry in entries {
        let file_type = entry.file_type();
        let path = entry.into_path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        if file_type != FsFileType::File {
            return Err(format!(
                "analysis run artifact must be a regular file: {}",
                path.display()
            )
            .into());
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
        let bytes = store.fs_read(&path).map_err(|error| {
            AnalysisPublicationError::from_io(error).context("failed to read run artifact")
        })?;
        let _record_reservation = admit(&bytes)?;
        if let Ok(artifact) = decode(&bytes, run_id) {
            visit(artifact)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
