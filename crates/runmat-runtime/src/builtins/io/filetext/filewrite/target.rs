use super::{filewrite_error_with_source, FILEWRITE_ERROR_IO};
use crate::BuiltinResult;
use runmat_filesystem as fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(super) struct OutputTarget {
    provider: Arc<dyn fs::FsProvider>,
    path: PathBuf,
}

impl OutputTarget {
    pub(super) fn capture(path: PathBuf) -> BuiltinResult<Self> {
        if path.is_absolute() {
            return Ok(Self {
                provider: fs::current_provider(),
                path,
            });
        }
        let (provider, current_dir) =
            fs::current_provider_with_current_dir().map_err(|source| {
                filewrite_error_with_source(
                    format!(
                        "{}: unable to resolve current folder: {source}",
                        FILEWRITE_ERROR_IO.message
                    ),
                    &FILEWRITE_ERROR_IO,
                    source,
                )
            })?;
        Ok(Self {
            provider,
            path: current_dir.join(path),
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) async fn open(&self, options: &fs::OpenOptions) -> std::io::Result<fs::File> {
        options
            .open_async_on(self.provider.as_ref(), &self.path)
            .await
    }
}
