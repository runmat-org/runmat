use runmat_filesystem as vfs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::builtins::common::fs::{
    compare_names, contains_wildcards, expand_user_path, path_to_string,
};
use crate::builtins::io::repl_fs::path_root::is_rooted_path;
use crate::BuiltinResult;

use super::input::Input;
use super::record::Record;

pub(super) async fn execute(input: Input) -> BuiltinResult<Vec<Record>> {
    let fs = DirFilesystem::capture()?;
    let mut records = match input {
        Input::Current => directory(&fs, fs.current_directory(), true).await?,
        Input::Name(name) => from_text(&fs, &name).await?,
        Input::FolderPattern { folder, pattern } => folder_pattern(&fs, &folder, &pattern).await?,
    };
    fs.check().map_err(super::error::operation)?;
    records.sort_by(|left, right| compare_names(&left.name, &right.name));
    Ok(records)
}

struct DirFilesystem {
    provider: Arc<dyn vfs::FsProvider>,
    cwd: PathBuf,
    lease: Option<vfs::CapturedFsResourceLease>,
}

impl DirFilesystem {
    fn capture() -> BuiltinResult<Self> {
        let (provider, cwd) = vfs::current_provider_with_current_dir().map_err(|error| {
            super::error::operation(format!(
                "dir: unable to determine current directory ({error})"
            ))
        })?;
        let lease = vfs::capture_resource_lease(provider.clone());
        Ok(Self {
            provider,
            cwd,
            lease,
        })
    }

    fn check(&self) -> std::io::Result<()> {
        if self.lease.as_ref().is_some_and(|lease| !lease.is_valid()) {
            Err(std::io::Error::other(
                "filesystem resource authority changed during directory listing",
            ))
        } else {
            Ok(())
        }
    }

    fn current_directory(&self) -> &Path {
        &self.cwd
    }
    fn resolved(&self, path: &Path) -> PathBuf {
        if is_rooted_path(path) {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }

    async fn metadata(&self, path: &Path) -> std::io::Result<vfs::FsMetadata> {
        self.check()?;
        let result = self.provider.metadata(&self.resolved(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    async fn canonicalize(&self, path: &Path) -> std::io::Result<PathBuf> {
        self.check()?;
        let result = self.provider.canonicalize(&self.resolved(path)).await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    async fn directory(
        &self,
        path: &Path,
    ) -> std::io::Result<Vec<crate::builtins::common::fs::enumeration::ListedEntry>> {
        self.check()?;
        let result = crate::builtins::common::fs::enumeration::directory_in(
            self.provider.as_ref(),
            &self.cwd,
            path,
        )
        .await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }

    async fn matching(
        &self,
        path: &Path,
    ) -> std::io::Result<Vec<crate::builtins::common::fs::enumeration::ListedEntry>> {
        self.check()?;
        let result = crate::builtins::common::fs::enumeration::matching_in(
            self.provider.as_ref(),
            &self.cwd,
            path,
        )
        .await;
        if result.is_ok() {
            self.check()?;
        }
        result
    }
}

async fn from_text(fs: &DirFilesystem, text: &str) -> BuiltinResult<Vec<Record>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return directory(fs, fs.current_directory(), true).await;
    }
    let expanded = expand_user_path(trimmed, "dir").map_err(super::error::operation)?;
    if contains_wildcards(&expanded) {
        pattern(fs, Path::new(&expanded), trimmed).await
    } else {
        path(fs, Path::new(&expanded), trimmed).await
    }
}

async fn folder_pattern(
    fs: &DirFilesystem,
    folder: &str,
    pattern_text: &str,
) -> BuiltinResult<Vec<Record>> {
    let expanded = expand_user_path(folder.trim(), "dir").map_err(super::error::operation)?;
    if contains_wildcards(&expanded) {
        return Err(super::error::contract(
            &runmat_builtins::DIR_ERROR_FOLDER_WILDCARD,
        ));
    }
    let folder = PathBuf::from(expanded);
    let pattern_text = pattern_text.trim();
    if pattern_text.is_empty() {
        return directory(fs, &folder, true).await;
    }
    let target = folder.join(pattern_text);
    if contains_wildcards(pattern_text) {
        pattern(fs, &target, pattern_text).await
    } else {
        path(fs, &target, pattern_text).await
    }
}

async fn path(fs: &DirFilesystem, path: &Path, display: &str) -> BuiltinResult<Vec<Record>> {
    match fs.metadata(path).await {
        Ok(metadata) if metadata.is_dir() => directory(fs, path, true).await,
        Ok(metadata) => {
            let folder_path = path.parent().unwrap_or_else(|| Path::new("."));
            let folder = absolute_folder(fs, folder_path).await?;
            let name = crate::builtins::common::fs::enumeration::filename(path)
                .to_string_lossy()
                .into_owned();
            Ok(vec![Record::from_metadata(folder, name, &metadata)])
        }
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(super::error::operation(format!(
            "dir: unable to access '{display}' ({error})"
        ))),
    }
}

async fn pattern(fs: &DirFilesystem, pattern: &Path, display: &str) -> BuiltinResult<Vec<Record>> {
    let entries = fs.matching(pattern).await.map_err(|error| {
        super::error::operation(format!(
            "dir: unable to enumerate matches for '{display}' ({error})"
        ))
    })?;
    let mut records = Vec::with_capacity(entries.len());
    for entry in entries {
        let folder =
            absolute_folder(fs, entry.path.parent().unwrap_or_else(|| Path::new("."))).await?;
        let name = crate::builtins::common::fs::enumeration::filename(&entry.path)
            .to_string_lossy()
            .into_owned();
        records.push(Record::from_metadata(folder, name, &entry.metadata));
    }
    Ok(records)
}

async fn directory(
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
        super::error::operation(format!("dir: unable to access '{absolute}' ({error})"))
    })?;
    records.extend(entries.into_iter().map(|entry| {
        let name = crate::builtins::common::fs::enumeration::filename(&entry.path)
            .to_string_lossy()
            .into_owned();
        Record::from_metadata(absolute.clone(), name, &entry.metadata)
    }));
    Ok(records)
}

async fn absolute_folder(fs: &DirFilesystem, path: &Path) -> BuiltinResult<String> {
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
    use super::*;
    use runmat_filesystem::MemoryFsProvider;

    #[test]
    fn captured_virtual_provider_survives_global_provider_swap() {
        let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
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
