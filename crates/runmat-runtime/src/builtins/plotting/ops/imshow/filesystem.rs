//! One captured filesystem owner for an image read and uploaded-name lookup.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_filesystem::{self as fs, CapturedFsResourceLease, FsProvider};

use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

pub(super) struct ImageFile {
    provider: Arc<dyn FsProvider>,
    path: PathBuf,
    lease: Option<CapturedFsResourceLease>,
    #[cfg(target_arch = "wasm32")]
    uploaded_name: Option<String>,
}

impl ImageFile {
    pub(super) fn capture(path: &str) -> BuiltinResult<Self> {
        let (provider, captured) = fs::capture_provider_path(Path::new(path)).map_err(error)?;
        let lease = fs::capture_resource_lease_for_effect(provider.clone()).map_err(error)?;
        let selected = Self {
            provider,
            path: captured,
            lease,
            #[cfg(target_arch = "wasm32")]
            uploaded_name: {
                let path_value = Path::new(path);
                (!path_value.is_absolute()
                    && path_value.components().count() == 1
                    && path_value
                        .components()
                        .all(|part| matches!(part, std::path::Component::Normal(_))))
                .then(|| path.to_owned())
            },
        };
        selected.check()?;
        Ok(selected)
    }

    pub(super) fn check(&self) -> BuiltinResult<()> {
        if self.lease.as_ref().is_some_and(|lease| !lease.is_valid()) {
            return Err(error(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "captured image filesystem authority was revoked",
            )));
        }
        Ok(())
    }

    pub(super) async fn read(&self, runtime: Option<&RuntimeContext>) -> BuiltinResult<Vec<u8>> {
        self.check()?;
        let result = crate::execution::staged_file::await_cancellable(
            runtime,
            super::BUILTIN_NAME,
            self.provider.read(&self.path),
        )
        .await?;
        self.check()?;
        match result {
            Ok(bytes) => Ok(bytes),
            Err(source) => {
                #[cfg(target_arch = "wasm32")]
                if source.kind() == std::io::ErrorKind::NotFound {
                    if let Some(name) = &self.uploaded_name {
                        if let Some(path) = self.unique_uploaded_path(runtime, name).await? {
                            self.check()?;
                            let bytes = crate::execution::staged_file::await_cancellable(
                                runtime,
                                super::BUILTIN_NAME,
                                self.provider.read(&path),
                            )
                            .await?;
                            self.check()?;
                            return bytes.map_err(error);
                        }
                    }
                }
                Err(error(source))
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    async fn unique_uploaded_path(
        &self,
        runtime: Option<&RuntimeContext>,
        name: &str,
    ) -> BuiltinResult<Option<PathBuf>> {
        let mut matches = Vec::new();
        let mut stack = vec![PathBuf::from("/")];
        let mut visited = 0usize;
        while let Some(dir) = stack.pop() {
            visited += 1;
            if visited > 4096 {
                return Err(super::imshow_error(
                    "imshow: virtual filesystem search exceeded directory limit",
                ));
            }
            self.check()?;
            let entries = crate::execution::staged_file::await_cancellable(
                runtime,
                super::BUILTIN_NAME,
                self.provider.read_dir(&dir),
            )
            .await?;
            self.check()?;
            // Unreadable directories are not an alternate provider or retry of
            // the requested file; continue the established uploaded-file lookup.
            for entry in entries.unwrap_or_default() {
                if entry.is_dir() {
                    stack.push(entry.path().to_path_buf());
                } else if entry.file_name().to_string_lossy() == name {
                    matches.push(entry.path().to_path_buf());
                }
            }
        }
        match matches.len() {
            0 => Ok(None),
            1 => Ok(matches.pop()),
            _ => Err(super::imshow_error(
                "imshow: uploaded image filename is ambiguous",
            )),
        }
    }
}

fn error(source: std::io::Error) -> RuntimeError {
    crate::build_runtime_error(format!("imshow: image filesystem access failed: {source}"))
        .with_builtin(super::BUILTIN_NAME)
        .with_identifier(super::IMSHOW_ERROR_INTERNAL.identifier.unwrap())
        .with_source(source)
        .build()
}
