use async_trait::async_trait;
use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags, ReadManyEntry};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

use super::bindings::JsFsFuncs;
use super::handle::{JsFileHandle, JsFileState};

pub(crate) fn install_js_fs_provider(bindings: &JsValue) -> Result<(), JsValue> {
    let funcs = JsFsFuncs::new(bindings)?;
    let provider: Arc<dyn FsProvider> = Arc::new(JsFsProvider { funcs });
    runmat_filesystem::set_provider(provider);
    Ok(())
}

pub(crate) struct JsFsProvider {
    funcs: JsFsFuncs,
}

unsafe impl Send for JsFsProvider {}
unsafe impl Sync for JsFsProvider {}

#[async_trait(?Send)]
impl FsProvider for JsFsProvider {
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let read_result = if should_load_initial(flags) {
            Some(self.funcs.read_file(path))
        } else {
            None
        };
        self.open_with_initial(path, flags, read_result)
    }

    async fn open_async(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let read_result = if should_load_initial(flags) {
            Some(self.funcs.read_file_async(path).await)
        } else {
            None
        };
        self.open_with_initial(path, flags, read_result)
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.funcs.read_file_async(path).await
    }

    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.funcs.write_file_async(path, data).await
    }

    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.funcs.remove_file_async(path).await
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.funcs.metadata_async(path).await
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.funcs.symlink_metadata_async(path).await
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.funcs.read_dir_async(path).await
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.funcs.canonicalize_async(path).await
    }

    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.funcs.create_dir_async(path).await
    }

    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.funcs.create_dir_all_async(path).await
    }

    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.funcs.remove_dir_async(path).await
    }

    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.funcs.remove_dir_all_async(path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.funcs.rename_async(from, to).await
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.funcs.set_readonly_async(path, readonly).await
    }

    async fn read_many(&self, paths: &[PathBuf]) -> io::Result<Vec<ReadManyEntry>> {
        self.funcs.read_many(paths).await
    }
}

impl JsFsProvider {
    fn open_with_initial(
        &self,
        path: &Path,
        flags: &OpenFlags,
        read_result: Option<io::Result<Vec<u8>>>,
    ) -> io::Result<Box<dyn FileHandle>> {
        let mut initial = Vec::new();
        let mut exists = false;
        let mut dirty = false;

        if let Some(read_result) = read_result {
            match read_result {
                Ok(bytes) => {
                    initial = bytes;
                    exists = true;
                }
                Err(err) if err.kind() == ErrorKind::NotFound => {
                    exists = false;
                }
                Err(err) => return Err(err),
            }
        }

        if flags.create_new && exists {
            return Err(io::Error::new(
                ErrorKind::AlreadyExists,
                format!("File already exists: {}", path.display()),
            ));
        }

        if (flags.create || flags.create_new) && !exists {
            exists = true;
            dirty = true;
        }

        if !exists && !flags.create {
            return Err(io::Error::new(
                ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            ));
        }

        if flags.truncate {
            initial.clear();
            dirty = true;
        }

        let cursor = if flags.append { initial.len() } else { 0 };

        let inner = JsFileState {
            funcs: self.funcs.clone(),
            path: path.to_path_buf(),
            buffer: initial,
            cursor,
            can_read: flags.read,
            can_write: flags.write || flags.append,
            append: flags.append,
            dirty,
        };
        #[allow(clippy::arc_with_non_send_sync)]
        let inner = Arc::new(Mutex::new(inner));
        Ok(Box::new(JsFileHandle { inner }))
    }
}

fn should_load_initial(flags: &OpenFlags) -> bool {
    !(flags.truncate && flags.create && !flags.read && !flags.append && !flags.create_new)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags(configure: impl FnOnce(&mut OpenFlags)) -> OpenFlags {
        let mut flags = OpenFlags::default();
        configure(&mut flags);
        flags
    }

    #[test]
    fn create_new_loads_initial_to_check_existence() {
        let flags = flags(|flags| {
            flags.write = true;
            flags.create_new = true;
        });

        assert!(should_load_initial(&flags));
    }

    #[test]
    fn create_without_truncate_loads_initial_to_preserve_existing_data() {
        let flags = flags(|flags| {
            flags.write = true;
            flags.create = true;
        });

        assert!(should_load_initial(&flags));
    }

    #[test]
    fn create_with_truncate_can_skip_initial_when_not_reading_or_appending() {
        let flags = flags(|flags| {
            flags.write = true;
            flags.create = true;
            flags.truncate = true;
        });

        assert!(!should_load_initial(&flags));
    }

    #[test]
    fn read_or_append_still_loads_initial_with_create_and_truncate() {
        let read_flags = flags(|flags| {
            flags.read = true;
            flags.create = true;
            flags.truncate = true;
        });
        let append_flags = flags(|flags| {
            flags.append = true;
            flags.create = true;
            flags.truncate = true;
        });

        assert!(should_load_initial(&read_flags));
        assert!(should_load_initial(&append_flags));
    }
}
