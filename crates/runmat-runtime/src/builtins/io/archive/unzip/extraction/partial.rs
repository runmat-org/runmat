use super::*;
use std::sync::Arc;

#[cfg(target_arch = "wasm32")]
mod cleanup;

/// An entry is not published until its complete body is synced and renamed.
/// Closing the owned handle precedes cleanup, including on a dropped future.
pub(super) struct PartialEntry {
    path: PathBuf,
    file: Option<Box<dyn vfs::FileHandle>>,
    provider: Arc<dyn vfs::FsProvider>,
    committed: bool,
    #[cfg(target_arch = "wasm32")]
    command_origin: Option<runmat_execution::profiling::CommandProfilingReceipt>,
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
            #[cfg(target_arch = "wasm32")]
            command_origin: None,
        })
    }

    /// This private staging entry retains only the extraction invocation's
    /// candidate origin. A lease is admitted later, only if rollback detaches.
    #[cfg(target_arch = "wasm32")]
    pub(super) async fn create_with_origin(
        path: PathBuf,
        provider: Arc<dyn vfs::FsProvider>,
        command_origin: Option<runmat_execution::profiling::CommandProfilingReceipt>,
    ) -> BuiltinResult<Self> {
        let mut partial = Self::create(path, provider).await?;
        partial.command_origin = command_origin;
        Ok(partial)
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
        {
            let token = self
                .command_origin
                .take()
                .and_then(|origin| origin.begin_deferred_work());
            cleanup::spawn(provider, path, token);
        }
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "partial/tests.rs"]
mod tests;
