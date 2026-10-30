//! Immutable default supervisor selection; explicit profiles keep their own contract.
use std::path::Path;

use super::{decode_profile, invalid, publish_profile, read_profile, Profile, PROFILE_MAX_BYTES};
use crate::supervisor::{filesystem, LocalSupervisorConfig, SupervisorPaths};
use crate::{NativeExecutionConfig, NativeExecutionError, NativeExecutionResult};

/// Default clients share the own-root immutable authority, not a new memory or
/// device-epoch observation on each CLI invocation. Explicit caller-supplied
/// configurations still use `publish_profile`'s strict equality contract.
pub(in crate::supervisor) fn resolve_default_profile(
    paths: SupervisorPaths,
    executable: &Path,
    discover: impl FnOnce() -> NativeExecutionResult<NativeExecutionConfig>,
) -> NativeExecutionResult<LocalSupervisorConfig> {
    use runmat_filesystem::{AtomicCreateResult, NativeFsProvider, PublicationDurability};

    validate_default_root(&paths.root)?;
    let path = paths.root.join("supervisor-profile.json");
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) => {
            if !metadata.file_type().is_file() {
                return Err(invalid("default supervisor profile must be a regular file"));
            }
            validate_default_permissions(&metadata, 0o600)?;
            return bind_default_executable(read_profile(&path)?, executable);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(filesystem::io_error(error)),
    }
    let config =
        bind_default_executable(LocalSupervisorConfig::new(discover()?, paths)?, executable)?;
    if path.to_str().is_none() {
        return Err(invalid(
            "supervisor bootstrap path must be representable in its host environment",
        ));
    }
    let profile = Profile {
        schema_version: 1,
        execution: &config.execution,
    };
    let bytes = runmat_canonical_codec::encode_json_bounded(&profile, PROFILE_MAX_BYTES)?;
    filesystem::secure_directory(&config.paths.root)?;
    match NativeFsProvider
        .atomic_create_private_new(&path, &bytes, PROFILE_MAX_BYTES)
        .map_err(filesystem::io_error)?
    {
        AtomicCreateResult::Published {
            durability: PublicationDurability::Durable,
        } => Ok(config),
        AtomicCreateResult::Published {
            durability: PublicationDurability::AtomicVisibility,
        } => Err(NativeExecutionError::CheckpointDelivery {
            diagnostic: "supervisor profile is visible but crash durability is unconfirmed".into(),
            evidence: Box::new(crate::NativeCheckpointEvidence::DurabilityIndeterminate {
                diagnostic: "supervisor profile publication directory retention is unconfirmed"
                    .into(),
            }),
        }),
        AtomicCreateResult::Existing(bytes) => {
            // A racing default publisher won. Its exact inventory/budget is
            // authority; a new observation cannot overwrite or enlarge it.
            validate_default_root(&config.paths.root)?;
            let metadata = std::fs::symlink_metadata(&path).map_err(filesystem::io_error)?;
            if !metadata.file_type().is_file() {
                return Err(invalid("default supervisor profile must be a regular file"));
            }
            validate_default_permissions(&metadata, 0o600)?;
            let winner = bind_default_executable(decode_profile(&path, &bytes)?, executable)?;
            publish_profile(&winner)?; // Existing immutable bytes; reaffirm retention.
            Ok(winner)
        }
    }
}

fn validate_default_root(root: &Path) -> NativeExecutionResult<()> {
    match std::fs::symlink_metadata(root) {
        Ok(metadata) if metadata.file_type().is_dir() => {
            validate_default_permissions(&metadata, 0o700)
        }
        Ok(_) => Err(invalid(
            "default supervisor root must be a regular private directory",
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(filesystem::io_error(error)),
    }
}

fn validate_default_permissions(
    metadata: &std::fs::Metadata,
    mode: u32,
) -> NativeExecutionResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        // The effective uid identifies this process; no foreign identity or
        // filesystem contents are dereferenced by this POSIX accessor.
        let owner = unsafe { libc::geteuid() };
        if metadata.uid() != owner || metadata.permissions().mode() & 0o777 != mode {
            return Err(invalid("default supervisor state must be owned by the current user with private directory/file permissions"));
        }
    }
    #[cfg(not(unix))]
    let _ = (metadata, mode); // Match the existing native token/provider boundary.
    Ok(())
}

fn bind_default_executable(
    config: LocalSupervisorConfig,
    executable: &Path,
) -> NativeExecutionResult<LocalSupervisorConfig> {
    if config.execution.executable != executable {
        return Err(invalid("default supervisor profile belongs to a different executable; use its original executable or a separate state root"));
    }
    Ok(config)
}

#[cfg(test)]
mod tests;
