//! Metadata-only inventory admission. No original content buffer is allocated here.
use super::*;

#[derive(serde::Serialize)]
pub(crate) struct FilePlan {
    pub path: PathBuf,
    pub length: u64,
    pub readonly: bool,
}
#[derive(serde::Serialize)]
pub(crate) struct CapturePlan {
    pub excluded: Vec<PathBuf>,
    pub directories: Vec<(PathBuf, bool)>,
    pub files: Vec<FilePlan>,
}

pub(super) fn inspect(root: &Path, request: WorkspaceCaptureRequest) -> io::Result<CapturePlan> {
    inspect_admitted(root, request, u64::MAX)
}

pub(crate) fn inspect_admitted(
    root: &Path,
    request: WorkspaceCaptureRequest,
    maximum: u64,
) -> io::Result<CapturePlan> {
    if !matches!(request.root.as_path(), path if path == Path::new("/") || path == Path::new(".")) {
        return Err(invalid(
            "native workspace capture requires the authorized mount root",
        ));
    }
    let mut excluded = request.excluded_subtrees;
    for path in &excluded {
        confined(root, path)?;
    }
    excluded.push(PathBuf::from(JOURNAL));
    excluded.sort();
    excluded.dedup();
    let mut metadata_bytes = (excluded.capacity() * std::mem::size_of::<PathBuf>()) as u64;
    for path in &excluded {
        metadata_bytes = metadata_bytes
            .checked_add(path.capacity() as u64)
            .ok_or_else(|| invalid("workspace inspection metadata overflow"))?;
    }
    charge(
        &mut metadata_bytes,
        std::mem::size_of::<PathBuf>() as u64,
        maximum,
    )?;
    let mut pending = vec![PathBuf::new()];
    let mut files = Vec::new();
    let mut directories = Vec::new();
    let mut size = 0u64;
    let mut visited = 0usize;
    while let Some(relative) = pending.pop() {
        let directory = root.join(&relative);
        ensure_no_symlink(root, &directory)?;
        charge(
            &mut metadata_bytes,
            (2 * std::mem::size_of::<(PathBuf, bool)>()) as u64 + relative.as_os_str().len() as u64,
            maximum,
        )?;
        directories.push((
            relative.clone(),
            fs::metadata(&directory)?.permissions().readonly(),
        ));
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let path = relative.join(entry.file_name());
            if excluded.iter().any(|ignored| path.starts_with(ignored)) {
                continue;
            }
            visited += 1;
            if visited > request.max_files as usize {
                return Err(invalid("workspace capture exceeds its inventory budget"));
            }
            let metadata = fs::symlink_metadata(entry.path())?;
            if metadata.file_type().is_symlink() {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "coherent native capture does not admit symbolic links",
                ));
            }
            if metadata.is_dir() {
                charge(
                    &mut metadata_bytes,
                    (2 * std::mem::size_of::<PathBuf>()) as u64 + path.capacity() as u64,
                    maximum,
                )?;
                pending.push(path);
                continue;
            }
            if !metadata.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "workspace contains a non-file input",
                ));
            }
            let path_text = path
                .to_str()
                .ok_or_else(|| invalid("workspace paths must be UTF-8"))?;
            if path_text.len() > 1024 || metadata.len() > request.max_bytes.saturating_sub(size) {
                return Err(invalid("workspace capture exceeds its bounded budget"));
            }
            size = size
                .checked_add(metadata.len())
                .ok_or_else(|| invalid("workspace capture size overflow"))?;
            charge(
                &mut metadata_bytes,
                (2 * std::mem::size_of::<FilePlan>()) as u64 + path.capacity() as u64,
                maximum,
            )?;
            files.push(FilePlan {
                path,
                length: metadata.len(),
                readonly: metadata.permissions().readonly(),
            });
        }
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    directories.sort();
    Ok(CapturePlan {
        excluded,
        directories,
        files,
    })
}

fn charge(used: &mut u64, bytes: u64, maximum: u64) -> io::Result<()> {
    let next = used
        .checked_add(bytes)
        .ok_or_else(|| invalid("workspace inspection metadata overflow"))?;
    if next > maximum {
        return Err(crate::workspace_access::resource_error(next, maximum));
    }
    *used = next;
    Ok(())
}
