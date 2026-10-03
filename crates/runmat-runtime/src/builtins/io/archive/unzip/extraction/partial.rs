use super::*;
use std::sync::Arc;

/// An entry is not published until its complete body is synced and renamed.
/// Closing the owned handle precedes cleanup, including on a dropped future.
pub(super) struct PartialEntry {
    path: PathBuf,
    file: Option<Box<dyn vfs::FileHandle>>,
    provider: Arc<dyn vfs::FsProvider>,
    committed: bool,
}

impl PartialEntry {
    pub(super) async fn create(
        path: PathBuf,
        provider: Arc<dyn vfs::FsProvider>,
    ) -> BuiltinResult<Self> {
        let flags = vfs::OpenFlags {
            write: true,
            create_new: true,
            ..vfs::OpenFlags::default()
        };
        let file = provider.open_async(&path, &flags).await.map_err(|err| {
            unzip_error_with_source(
                &UNZIP_ERROR_IO,
                format!(
                    "unzip: unable to create temporary output '{}': {err}",
                    path.display()
                ),
                err,
            )
        })?;
        Ok(Self {
            path,
            file: Some(file),
            provider,
            committed: false,
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.file
            .as_mut()
            .expect("open partial output")
            .write_all(bytes)
    }

    pub(super) async fn flush(&mut self) -> std::io::Result<()> {
        self.file
            .as_mut()
            .expect("open partial output")
            .flush_async()
            .await
    }

    pub(super) async fn sync_all(&mut self) -> std::io::Result<()> {
        self.file
            .as_mut()
            .expect("open partial output")
            .sync_all_async()
            .await
    }

    pub(super) fn close(&mut self) {
        self.file.take();
    }

    pub(super) async fn commit_to(mut self, output: &Path) -> BuiltinResult<()> {
        self.close();
        self.provider
            .rename(&self.path, output)
            .await
            .map_err(|err| {
                unzip_error_with_source(
                    &UNZIP_ERROR_IO,
                    format!("unzip: unable to create '{}': {err}", output.display()),
                    err,
                )
            })?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for PartialEntry {
    fn drop(&mut self) {
        self.close();
        if self.committed {
            return;
        }
        let path = self.path.clone();
        let provider = self.provider.clone();
        #[cfg(not(target_arch = "wasm32"))]
        let _ = vfs::remove_file_in_provider(provider, &path);
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            let _ = provider.remove_file(&path).await;
        });
    }
}
