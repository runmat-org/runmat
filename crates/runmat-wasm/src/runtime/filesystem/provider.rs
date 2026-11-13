use async_trait::async_trait;
use runmat_filesystem::{
    DirEntry, DirectoryDialogRequest, DirectoryDialogSelection, FileHandle, FsMetadata, FsProvider,
    OpenFileDialogRequest, OpenFileDialogSelection, OpenFlags, ReadManyEntry,
    SaveFileDialogRequest, SaveFileDialogSelection,
};
use runmat_runtime::context::{HostInteraction, RuntimeHostService, RuntimeServiceFuture};
use runmat_runtime::{interaction::InteractionResponse, RuntimeError};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

use super::bindings::JsFsFuncs;
use super::handle::{JsFileHandle, JsFileState};

pub(crate) fn install_js_fs_provider(
    bindings: &JsValue,
) -> Result<Rc<dyn RuntimeHostService>, JsValue> {
    let funcs = JsFsFuncs::new(bindings)?;
    let provider = Arc::new(JsFsProvider { funcs });
    runmat_filesystem::set_provider(provider.clone());
    Ok(Rc::new(JsDialogHost { provider }))
}

pub(crate) struct JsFsProvider {
    funcs: JsFsFuncs,
}

unsafe impl Send for JsFsProvider {}
unsafe impl Sync for JsFsProvider {}

#[async_trait(?Send)]
impl FsProvider for JsFsProvider {
    fn supports_workspace_transactions(&self) -> bool {
        self.funcs.supports_workspace_transactions()
    }
    async fn capture_workspace(
        &self,
        request: runmat_filesystem::workspace::WorkspaceCaptureRequest,
    ) -> io::Result<runmat_filesystem::workspace::WorkspaceSnapshot> {
        self.funcs.capture_workspace(&request).await
    }

    async fn lookup_workspace_publication(
        &self,
        request: &runmat_filesystem::workspace::WorkspacePublicationRequest,
    ) -> io::Result<Option<runmat_filesystem::workspace::WorkspacePublicationOutcome>> {
        self.funcs.lookup_workspace_publication(request).await
    }

    async fn publish_workspace(
        &self,
        request: runmat_filesystem::workspace::WorkspacePublicationRequest,
    ) -> io::Result<runmat_filesystem::workspace::WorkspacePublicationOutcome> {
        self.funcs.publish_workspace(&request).await
    }
    fn supports_read(&self) -> bool {
        self.funcs.supports_read()
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/"))
    }

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
    async fn read_version(
        &self,
        path: &Path,
        version: &str,
        size: u64,
        snapshot: &str,
    ) -> io::Result<Vec<u8>> {
        self.funcs
            .read_version_async(path, version, size, snapshot)
            .await
    }

    async fn publish_output(
        &self,
        path: &Path,
        data: &[u8],
        expected: Option<&str>,
    ) -> io::Result<String> {
        self.funcs.publish_output_async(path, data, expected).await
    }

    async fn remove_output(&self, path: &Path, expected: &str) -> io::Result<()> {
        self.funcs.remove_output_async(path, expected).await
    }

    async fn atomic_create_new(
        &self,
        path: &Path,
        data: &[u8],
    ) -> io::Result<runmat_filesystem::AtomicCreateResult> {
        self.funcs.atomic_create_new_async(path, data).await
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

    async fn rename_replace(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.funcs.rename_replace_async(from, to).await
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.funcs.set_readonly_async(path, readonly).await
    }

    async fn read_many(&self, paths: &[PathBuf]) -> io::Result<Vec<ReadManyEntry>> {
        self.funcs.read_many(paths).await
    }
}

struct JsDialogHost {
    provider: Arc<JsFsProvider>,
}

impl RuntimeHostService for JsDialogHost {
    fn console(
        &self,
        stream: runmat_runtime::console::ConsoleStream,
        text: runmat_runtime::console::ConsoleText,
    ) {
        runmat_runtime::console::forward_recorded_console_output(stream, text);
    }

    fn warning(&self, _: runmat_runtime::warning_store::RuntimeWarning) {}

    fn interact(
        &self,
        request: HostInteraction,
    ) -> RuntimeServiceFuture<Result<InteractionResponse, RuntimeError>> {
        runmat_runtime::interaction::invoke_installed_async_handler(request).unwrap_or_else(|| {
            Box::pin(async {
                Err(runmat_runtime::build_runtime_error(
                    "The browser host does not provide interactive input",
                )
                .with_identifier("RunMat:host:InteractionUnavailable")
                .build())
            })
        })
    }

    fn select_file_open(
        &self,
        request: OpenFileDialogRequest,
    ) -> RuntimeServiceFuture<Result<Option<OpenFileDialogSelection>, RuntimeError>> {
        let provider = self.provider.clone();
        Box::pin(async move {
            provider
                .funcs
                .select_file_open_async(&request)
                .await
                .map_err(dialog_error)
        })
    }

    fn select_file_save(
        &self,
        request: SaveFileDialogRequest,
    ) -> RuntimeServiceFuture<Result<Option<SaveFileDialogSelection>, RuntimeError>> {
        let provider = self.provider.clone();
        Box::pin(async move {
            provider
                .funcs
                .select_file_save_async(&request)
                .await
                .map_err(dialog_error)
        })
    }

    fn select_directory(
        &self,
        _request: DirectoryDialogRequest,
    ) -> RuntimeServiceFuture<Result<Option<DirectoryDialogSelection>, RuntimeError>> {
        Box::pin(async { Ok(None) })
    }
}

fn dialog_error(source: io::Error) -> RuntimeError {
    runmat_runtime::build_runtime_error(source.to_string())
        .with_identifier("RunMat:host:DialogFailed")
        .with_source(source)
        .build()
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
