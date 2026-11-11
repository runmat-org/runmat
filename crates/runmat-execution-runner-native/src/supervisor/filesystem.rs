use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::{NativeExecutionError, NativeExecutionResult};

const MAX_PERSISTED_JSON_BYTES: u64 = 64 * 1024 * 1024;

pub(crate) fn atomic_json(path: &Path, value: &impl Serialize) -> NativeExecutionResult<()> {
    let bytes = runmat_canonical_codec::encode_json_bounded(value, MAX_PERSISTED_JSON_BYTES)?;
    atomic_write(path, &bytes)
}

pub(crate) fn immutable_json(path: &Path, value: &impl Serialize) -> NativeExecutionResult<()> {
    use runmat_filesystem::{AtomicCreateResult, NativeFsProvider, PublicationDurability};
    let bytes = runmat_canonical_codec::encode_json_bounded(value, MAX_PERSISTED_JSON_BYTES)?;
    secure_directory(path.parent().ok_or_else(|| {
        NativeExecutionError::Configuration("immutable metadata requires a parent".into())
    })?)?;
    match NativeFsProvider
        .atomic_create_private_new(path, &bytes, MAX_PERSISTED_JSON_BYTES)
        .map_err(io_error)?
    {
        AtomicCreateResult::Published {
            durability: PublicationDurability::Durable,
        } => Ok(()),
        AtomicCreateResult::Published {
            durability: PublicationDurability::AtomicVisibility,
        } => Err(NativeExecutionError::CheckpointDelivery {
            diagnostic: format!(
                "{} is visible but crash durability is unconfirmed",
                path.display()
            ),
            evidence: Box::new(crate::NativeCheckpointEvidence::DurabilityIndeterminate {
                diagnostic: "immutable publication directory retention is unconfirmed".into(),
            }),
        }),
        AtomicCreateResult::Existing(existing) if existing == bytes => {
            // Preserve the winner; reaffirm retention rather than replacing it.
            // Windows FlushFileBuffers requires write access; reopening never
            // truncates or replaces the original immutable publication.
            #[cfg(windows)]
            let retained = std::fs::OpenOptions::new().write(true).open(path);
            #[cfg(not(windows))]
            let retained = std::fs::File::open(path);
            retained
                .and_then(|file| file.sync_all())
                .map_err(|error| retained_publication_error(path, error))?;
            NativeFsProvider
                .synchronize_directory(path.parent().expect("validated parent"))
                .map_err(|error| retained_publication_error(path, error))
        }
        AtomicCreateResult::Existing(_) => Err(NativeExecutionError::Protocol(
            "immutable durable attempt metadata differs from its original publisher".into(),
        )),
    }
}

pub(super) fn read_json<T: DeserializeOwned>(path: &Path) -> NativeExecutionResult<T> {
    read_optional_json(path)?.ok_or_else(|| {
        io_error(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{}: durable metadata is absent", path.display()),
        ))
    })
}

pub(super) fn read_optional_json<T: DeserializeOwned>(
    path: &Path,
) -> NativeExecutionResult<Option<T>> {
    read_optional_json_bounded(path, MAX_PERSISTED_JSON_BYTES)
}

pub(crate) fn read_optional_json_bounded<T: DeserializeOwned>(
    path: &Path,
    maximum_bytes: u64,
) -> NativeExecutionResult<Option<T>> {
    let bytes = match runmat_filesystem::NativeFsProvider
        .read_bounded_sync(path, maximum_bytes.min(MAX_PERSISTED_JSON_BYTES))
    {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io_error(error)),
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| NativeExecutionError::Protocol(format!("{}: {error}", path.display())))
}

pub(super) fn atomic_write(path: &Path, bytes: &[u8]) -> NativeExecutionResult<()> {
    use runmat_filesystem::NativeFsProvider;
    let parent = path.parent().ok_or_else(|| {
        NativeExecutionError::Configuration("durable metadata path has no parent".into())
    })?;
    runmat_filesystem::NativeFsProvider
        .create_private_directory_durable(parent)
        .map_err(|source| NativeExecutionError::MetadataIo {
            operation: crate::MetadataIoOperation::PreparePrivateDirectory,
            path: parent.to_owned(),
            source,
        })?;
    let outcome = NativeFsProvider
        .atomic_replace_sync(path, bytes)
        .map_err(|source| NativeExecutionError::MetadataIo {
            operation: crate::MetadataIoOperation::AtomicReplacement,
            path: path.to_owned(),
            source,
        })?;
    let evidence = crate::NativeCheckpointEvidence::from_publication(outcome);
    if let Some(diagnostic) = evidence.delivery_error() {
        return Err(NativeExecutionError::CheckpointDelivery {
            diagnostic: format!("{}: {diagnostic}", path.display()),
            evidence: Box::new(evidence),
        });
    }
    Ok(())
}

pub(crate) fn secure_directory(path: &Path) -> NativeExecutionResult<()> {
    runmat_filesystem::NativeFsProvider
        .create_private_directory_durable(path)
        .map_err(io_error)
}

pub(super) fn io_error(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

pub(crate) fn unix_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

fn retained_publication_error(path: &Path, error: std::io::Error) -> NativeExecutionError {
    let diagnostic = format!(
        "{} remains visible but retention failed: {error}",
        path.display()
    );
    NativeExecutionError::CheckpointDelivery {
        evidence: Box::new(crate::NativeCheckpointEvidence::DurabilityIndeterminate {
            diagnostic: diagnostic.clone(),
        }),
        diagnostic,
    }
}

#[cfg(test)]
mod tests;
