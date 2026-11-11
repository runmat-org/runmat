//! Frozen native execution profile passed to the actual detached authority.
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{filesystem, LocalSupervisorConfig, SupervisorPaths};
use crate::{NativeExecutionConfig, NativeExecutionError, NativeExecutionResult};

mod default_profile;
pub(super) use default_profile::resolve_default_profile;

pub(super) const PROFILE_ENV: &str = "RUNMAT_EXECUTION_SUPERVISOR_PROFILE";
pub(super) const PROFILE_MAX_BYTES: u64 = 64 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile<T> {
    schema_version: u16,
    execution: T,
}

pub(super) fn publish_profile(config: &LocalSupervisorConfig) -> NativeExecutionResult<PathBuf> {
    config.validate()?;
    let path = config.paths.root.join("supervisor-profile.json");
    if path.to_str().is_none() {
        return Err(invalid(
            "supervisor bootstrap path must be representable in its host environment",
        ));
    }
    let profile = Profile {
        schema_version: 1,
        execution: &config.execution,
    };
    runmat_canonical_codec::profile_json_bytes(&profile, PROFILE_MAX_BYTES)?;
    filesystem::immutable_json(&path, &profile)?;
    Ok(path)
}

pub(super) fn read_profile(path: &Path) -> NativeExecutionResult<LocalSupervisorConfig> {
    let bytes = runmat_filesystem::NativeFsProvider
        .read_bounded_sync(path, PROFILE_MAX_BYTES)
        .map_err(filesystem::io_error)?;
    decode_profile(path, &bytes)
}

fn decode_profile(path: &Path, bytes: &[u8]) -> NativeExecutionResult<LocalSupervisorConfig> {
    let profile: Profile<NativeExecutionConfig> =
        serde_json::from_slice(bytes).map_err(|error| invalid(error.to_string()))?;
    if profile.schema_version != 1 {
        return Err(invalid("unsupported supervisor bootstrap profile"));
    }
    let paths = SupervisorPaths::new(profile.execution.store_root.clone())?;
    if path != paths.root.join("supervisor-profile.json") {
        return Err(invalid(
            "supervisor bootstrap profile differs from its authority root",
        ));
    }
    LocalSupervisorConfig::new(profile.execution, paths)
}

/// The launcher supplies the frozen profile only to its actual detached child.
/// Library-owned supervisors must never replace their caller's standard streams.
pub(super) fn release_startup_diagnostics() -> NativeExecutionResult<()> {
    if std::env::var_os(PROFILE_ENV).is_some() {
        runmat_process_host::release_startup_stderr()?;
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> NativeExecutionError {
    NativeExecutionError::Configuration(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_profile_survives_cold_read_and_rejects_foreign_replacement() {
        let temp = tempfile::tempdir().unwrap();
        let mut execution = NativeExecutionConfig::for_current_executable().unwrap();
        execution.worker_arguments = vec!["--original-worker".into(), "cohort".into()];
        execution.max_workers = 3;
        execution.worker_memory_bytes = 123_456;
        let config = LocalSupervisorConfig::new(
            execution,
            SupervisorPaths::new(temp.path().join("authority")).unwrap(),
        )
        .unwrap();
        let path = publish_profile(&config).unwrap();
        let original = std::fs::read(&path).unwrap();
        let cold = read_profile(&path).unwrap();
        assert_eq!(cold.execution, config.execution);
        assert_eq!(cold.paths, config.paths);
        assert_eq!(cold.identity().unwrap(), config.identity().unwrap());
        let mut foreign = config.clone();
        foreign.execution.max_workers += 1;
        assert!(publish_profile(&foreign).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn profile_rejects_foreign_root_and_oversize_before_loading() {
        let temp = tempfile::tempdir().unwrap();
        let config = LocalSupervisorConfig::new(
            NativeExecutionConfig::for_current_executable().unwrap(),
            SupervisorPaths::new(temp.path().join("authority")).unwrap(),
        )
        .unwrap();
        let path = publish_profile(&config).unwrap();
        let other = temp.path().join("foreign-profile.json");
        std::fs::copy(&path, &other).unwrap();
        assert!(read_profile(&other).is_err());
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(PROFILE_MAX_BYTES + 1)
            .unwrap();
        assert!(read_profile(&path).is_err());
    }
}
