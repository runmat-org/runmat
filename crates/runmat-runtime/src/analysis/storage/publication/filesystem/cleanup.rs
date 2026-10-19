//! Reclaim hidden private attempts only after shared journal recovery under the write fence.
use super::*;

#[cfg(test)]
mod tests;

pub(super) fn uncommitted_roots(
    store: &access::FilesystemAccess<'_>,
    incoming: Option<Digest>,
) -> Result<(), AnalysisPublicationError> {
    let roots = transaction::roots(store)?;
    for root in roots.iter() {
        let name = root
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("publication directory name invalid")?;
        let identity = format!("sha256:{name}")
            .parse::<Digest>()
            .map_err(|e| e.to_string())?;
        // Preserve the exact retry's immutable private records; never guess producer liveness.
        if Some(identity) == incoming || descriptor(store, identity, 1 << 20)?.is_some() {
            continue;
        }
        let mut files = inventory::Paths::new(store)?;
        let mut directories = inventory::Paths::new(store)?;
        let entries = store
            .fs_read_dir(root)
            .map_err(AnalysisPublicationError::from_io)?;
        let _entries_reservation = store.retain_directory(&entries)?;
        for entry in entries {
            let name = entry
                .file_name()
                .to_str()
                .ok_or("publication filename invalid")?;
            match name {
                "runs" | "observations" if entry.file_type() == FsFileType::Directory => {
                    let children = store
                        .fs_read_dir(entry.path())
                        .map_err(AnalysisPublicationError::from_io)?;
                    let _children_reservation = store.retain_directory(&children)?;
                    for child in children {
                        if child.file_type() != FsFileType::File
                            || child.path().extension().and_then(|s| s.to_str()) != Some("json")
                        {
                            return Err(
                                "private publication record must be a regular JSON file".into()
                            );
                        }
                        validate_run_id_token(
                            child
                                .path()
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .ok_or("private run filename invalid")?,
                        )?;
                        files.push(child.path())?;
                    }
                    directories.push(entry.path())?;
                }
                "manifest.json" if entry.file_type() == FsFileType::File => {
                    files.push(entry.path())?
                }
                _ => return Err("uncommitted publication contains an unowned path".into()),
            }
        }
        store.visit_bindings_for(identity, |path| files.push(path))?;
        // Every path has been validated before deleting anything. A stopped cleanup is
        // retryable: no marker exists, and none of these files can become a legacy run.
        for file in files.iter() {
            store
                .fs_remove_file(file)
                .map_err(AnalysisPublicationError::from_io)?;
        }
        for directory in directories.iter() {
            store
                .fs_remove_dir(directory)
                .map_err(AnalysisPublicationError::from_io)?;
        }
        store
            .fs_remove_dir(root)
            .map_err(AnalysisPublicationError::from_io)?;
    }
    Ok(())
}
