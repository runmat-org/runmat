use super::super::record::Record;
use super::authority::DirFilesystem;
use crate::builtins::common::fs::path_to_string;
use crate::builtins::io::repl_fs::path_root::is_rooted_path;
use crate::BuiltinResult;
use std::path::Path;

pub(super) async fn directory(
    fs: &DirFilesystem,
    folder: &Path,
    include_special: bool,
) -> BuiltinResult<Vec<Record>> {
    let absolute = absolute_folder(fs, folder).await?;
    let metadata = fs.metadata(folder).await.ok();
    let mut records = Vec::new();
    if include_special {
        records.push(Record::special(".", &absolute, metadata.as_ref()));
        records.push(Record::special("..", &absolute, metadata.as_ref()));
    }
    let entries = fs.directory(folder).await.map_err(|error| {
        super::super::error::io(
            format!("dir: unable to access '{absolute}' ({error})"),
            error,
        )
    })?;
    records.extend(entries.into_iter().map(|entry| {
        let name = crate::builtins::common::fs::enumeration::filename(&entry.path)
            .to_string_lossy()
            .into_owned();
        Record::from_metadata(absolute.clone(), name, &entry.metadata)
    }));
    Ok(records)
}

pub(super) async fn absolute_folder(fs: &DirFilesystem, path: &Path) -> BuiltinResult<String> {
    let joined = if is_rooted_path(path) {
        path.to_path_buf()
    } else {
        fs.current_directory().join(path)
    };
    let normalized = fs.canonicalize(&joined).await.unwrap_or(joined);
    Ok(path_to_string(&normalized))
}

#[cfg(test)]
mod tests {
    use super::super::selection::from_text;
    use super::*;
    use runmat_filesystem::{self as vfs, MemoryFsProvider};
    use std::sync::Arc;

    #[test]
    fn captured_virtual_provider_survives_global_provider_swap() {
        let _lock = runmat_filesystem::provider_override_lock();
        let memory = Arc::new(MemoryFsProvider::with_current_dir("/workspace"));
        memory
            .write_project_path("/workspace/virtual.txt", b"owned")
            .expect("virtual file");
        let captured = vfs::with_provider_override(memory, || {
            DirFilesystem::capture().expect("capture selected provider and cwd")
        });
        let records = futures::executor::block_on(from_text(&captured, "."))
            .expect("list through retained provider");
        assert!(records.iter().any(|record| record.name == "virtual.txt"));
    }
}
