use super::bindings::JsFsFuncs;
use super::handle::{JsFileHandle, JsFileState};
use crate::{
    DirEntry, FileHandle, FsMetadata, FsProvider, OpenFileDialogRequest, OpenFileDialogSelection,
    OpenFlags, ReadManyEntry, SaveFileDialogRequest, SaveFileDialogSelection,
};
use async_trait::async_trait;
use std::cell::RefCell;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use wasm_bindgen::prelude::*;

pub struct JsFsProvider {
    funcs: JsFsFuncs,
}

// SAFETY: this module rejects shared-memory/atomics builds. A provider and its
// callback table remain in one Wasm instance on one JS agent; Rust threads
// cannot transfer or concurrently access these JS references in this build.
unsafe impl Send for JsFsProvider {}
unsafe impl Sync for JsFsProvider {}

#[async_trait(?Send)]
impl FsProvider for JsFsProvider {
    fn supports_workspace_transactions(&self) -> bool {
        self.funcs.supports_workspace_transactions()
    }
    async fn capture_workspace(
        &self,
        request: crate::workspace::WorkspaceCaptureRequest,
    ) -> io::Result<crate::workspace::WorkspaceSnapshot> {
        self.funcs.capture_workspace(&request).await
    }

    async fn lookup_workspace_publication(
        &self,
        request: &crate::workspace::WorkspacePublicationRequest,
    ) -> io::Result<Option<crate::workspace::WorkspacePublicationOutcome>> {
        self.funcs.lookup_workspace_publication(request).await
    }

    async fn publish_workspace(
        &self,
        request: crate::workspace::WorkspacePublicationRequest,
    ) -> io::Result<crate::workspace::WorkspacePublicationOutcome> {
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
    ) -> io::Result<crate::AtomicCreateResult> {
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

impl JsFsProvider {
    pub fn new(bindings: &JsValue) -> Result<Self, JsValue> {
        Ok(Self {
            funcs: JsFsFuncs::new(bindings)?,
        })
    }

    pub async fn select_file_open(
        &self,
        request: &OpenFileDialogRequest,
    ) -> io::Result<Option<OpenFileDialogSelection>> {
        self.funcs.select_file_open_async(request).await
    }

    pub async fn select_file_save(
        &self,
        request: &SaveFileDialogRequest,
    ) -> io::Result<Option<SaveFileDialogSelection>> {
        self.funcs.select_file_save_async(request).await
    }

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
        let inner = Rc::new(RefCell::new(inner));
        Ok(Box::new(JsFileHandle { inner }))
    }
}

fn should_load_initial(flags: &OpenFlags) -> bool {
    !(flags.truncate && flags.create && !flags.read && !flags.append && !flags.create_new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[wasm_bindgen_test::wasm_bindgen_test]
    async fn shared_handle_flushes_through_its_own_async_host_callback() {
        let bindings = js_sys::Object::new();
        let write = js_sys::Function::new_with_args(
            "path, bytes",
            "this.path = path; this.bytes = Array.from(bytes); return Promise.resolve();",
        );
        js_sys::Reflect::set(&bindings, &JsValue::from_str("writeFile"), &write).unwrap();
        let provider = JsFsProvider::new(&bindings.clone().into()).unwrap();
        let mut file = provider
            .open_async(
                Path::new("/project/output.m"),
                &OpenFlags {
                    write: true,
                    create: true,
                    truncate: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        file.write_all(&[1, 2, 3]).unwrap();
        file.flush_async().await.unwrap();
        assert_eq!(
            js_sys::Reflect::get(&bindings, &JsValue::from_str("path"))
                .unwrap()
                .as_string()
                .as_deref(),
            Some("/project/output.m")
        );
        let bytes: Vec<u8> = serde_wasm_bindgen::from_value(
            js_sys::Reflect::get(&bindings, &JsValue::from_str("bytes")).unwrap(),
        )
        .unwrap();
        assert_eq!(bytes, [1, 2, 3]);
    }

    #[wasm_bindgen_test::wasm_bindgen_test]
    async fn shared_provider_preserves_async_host_callback_receiver_and_path() {
        let bindings = js_sys::Object::new();
        let read = js_sys::Function::new_with_args(
            "path",
            "this.requested = path; return Promise.resolve(new Uint8Array([42]));",
        );
        js_sys::Reflect::set(&bindings, &JsValue::from_str("readFile"), &read).unwrap();
        let provider = JsFsProvider::new(&bindings.clone().into()).unwrap();
        assert_eq!(
            provider.read(Path::new("/project/helper.m")).await.unwrap(),
            [42]
        );
        assert_eq!(
            js_sys::Reflect::get(&bindings, &JsValue::from_str("requested"))
                .unwrap()
                .as_string()
                .as_deref(),
            Some("/project/helper.m"),
        );
    }

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
