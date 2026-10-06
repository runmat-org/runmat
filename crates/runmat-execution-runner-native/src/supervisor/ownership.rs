//! Physical durable-provider ownership, independent of reusable process IDs.
use std::path::{Path, PathBuf};

use runmat_execution_runner::AttemptRequest;
use runmat_filesystem::ExclusiveFileLock;

use super::filesystem::io_error;
use crate::{NativeExecutionError, NativeExecutionResult};

/// Construction requires the original persisted assignment and its kernel lock.
/// This guard is neither cloneable nor serializable. Hold it through publication
/// of the original terminal report; PID metadata is only a display hint.
pub struct DurableAttemptLease {
    request: AttemptRequest,
    directory: PathBuf,
    _guard: ExclusiveFileLock,
}

impl std::fmt::Debug for DurableAttemptLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DurableAttemptLease")
            .field("attempt_id", &self.request.id)
            .finish_non_exhaustive()
    }
}

impl DurableAttemptLease {
    pub(super) fn acquire(
        directory: &Path,
        request: &AttemptRequest,
    ) -> NativeExecutionResult<Self> {
        let started = std::time::Instant::now();
        let claimed = super::driver_files::marker_path(directory, Some(request.id));
        let guard = loop {
            // An attempt gets one physical execution. A released old lease is
            // not authority to replay the same assignment after a provider loss.
            if claimed.try_exists().map_err(io_error)? {
                return Err(NativeExecutionError::Configuration(
                    "original durable attempt already claimed its physical provider".into(),
                ));
            }
            if let Some(guard) =
                ExclusiveFileLock::try_acquire(&path(directory, request)).map_err(io_error)?
            {
                // A competing provider may have published before releasing.
                if claimed.try_exists().map_err(io_error)? {
                    return Err(NativeExecutionError::Configuration(
                        "original durable provider claim prevents attempt replay".into(),
                    ));
                }
                break guard;
            }
            // A supervisor's lock probe is transient. Do not confuse it with
            // a second provider; wait only within the existing launch grace.
            if started.elapsed().as_millis()
                >= u128::from(super::service::STARTING_RECOVERY_GRACE_MILLIS)
            {
                return Err(NativeExecutionError::Configuration(
                    "original durable provider ownership cannot be acquired".into(),
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        Ok(Self {
            request: request.clone(),
            directory: directory.to_path_buf(),
            _guard: guard,
        })
    }

    pub(super) fn validate(
        &self,
        directory: &Path,
        request: &AttemptRequest,
    ) -> NativeExecutionResult<()> {
        if self.directory != directory || self.request != *request {
            return Err(NativeExecutionError::Configuration(
                "physical durable lease differs from the original assignment".into(),
            ));
        }
        Ok(())
    }
}

/// Probe the original persistent inode without replacing or deleting it.
/// `false` means the provider guard is absent; it does not prove its descendants
/// exited, recover missing consumption, or establish terminal result authority.
pub(super) fn is_owned(directory: &Path, request: &AttemptRequest) -> NativeExecutionResult<bool> {
    Ok(ExclusiveFileLock::try_acquire(&path(directory, request))
        .map_err(io_error)?
        .is_none())
}

fn path(directory: &Path, request: &AttemptRequest) -> PathBuf {
    directory.join(format!("{}.provider.lock", request.id))
}

#[cfg(all(test, unix))]
mod tests;
