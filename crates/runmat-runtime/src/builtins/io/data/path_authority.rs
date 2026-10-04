//! One filesystem and working-directory authority for a public data path call.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_filesystem::{self as fs, FsProvider};

use crate::data::data_error;
use crate::BuiltinResult;

pub(super) struct DataPathAuthority {
    provider: Arc<dyn FsProvider>,
    cwd: PathBuf,
}

impl DataPathAuthority {
    pub(super) fn capture() -> BuiltinResult<Self> {
        let (provider, cwd) = fs::current_provider_with_current_dir().map_err(|error| {
            data_error(format!(
                "data: failed to capture current directory: {error}"
            ))
        })?;
        Ok(Self { provider, cwd })
    }

    pub(super) fn resolve(&self, path: impl AsRef<Path>) -> PathBuf {
        let path = path.as_ref();
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }

    pub(super) fn provider(&self) -> &dyn FsProvider {
        self.provider.as_ref()
    }
}
