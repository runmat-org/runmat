//! Captured-provider, bounded-chunk copy with caller-owned cooperation.

use crate::{FsProvider, OpenFlags};
use async_trait::async_trait;
use std::io;
use std::path::Path;

#[async_trait(?Send)]
pub trait FsCopyCheckpoint {
    type Error;

    /// Called before the next provider effect and after completed byte work.
    async fn checkpoint(&mut self, completed_bytes: u64) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub enum FsCopyError<E> {
    Io(io::Error),
    Checkpoint(E),
}

/// Copy without looking up ambient provider/CWD after admission. The provider
/// owns both handles; dropping the future drops them. A failed copy can leave
/// a partial destination, as does the existing synchronous copy contract.
pub async fn copy_file_on_async<C: FsCopyCheckpoint>(
    provider: &dyn FsProvider,
    from: &Path,
    to: &Path,
    checkpoint: &mut C,
) -> Result<u64, FsCopyError<C::Error>> {
    if !provider.path_namespace().is_absolute(from) || !provider.path_namespace().is_absolute(to) {
        return Err(FsCopyError::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            "copy_file_on_async requires captured absolute paths",
        )));
    }
    checkpoint
        .checkpoint(0)
        .await
        .map_err(FsCopyError::Checkpoint)?;
    let mut reader = provider
        .open_async(
            from,
            &OpenFlags {
                read: true,
                ..OpenFlags::default()
            },
        )
        .await
        .map_err(FsCopyError::Io)?;
    checkpoint
        .checkpoint(0)
        .await
        .map_err(FsCopyError::Checkpoint)?;
    let mut writer = provider
        .open_async(
            to,
            &OpenFlags {
                write: true,
                create: true,
                truncate: true,
                ..OpenFlags::default()
            },
        )
        .await
        .map_err(FsCopyError::Io)?;
    let mut buffer = [0u8; 16 * 1024];
    let mut written_total = 0u64;
    loop {
        checkpoint
            .checkpoint(0)
            .await
            .map_err(FsCopyError::Checkpoint)?;
        let read = loop {
            match reader.read_async(&mut buffer).await {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                    checkpoint
                        .checkpoint(0)
                        .await
                        .map_err(FsCopyError::Checkpoint)?;
                }
                result => break result.map_err(FsCopyError::Io)?,
            }
        };
        if read > buffer.len() {
            return Err(FsCopyError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                "filesystem handle reported a read beyond the supplied buffer",
            )));
        }
        if read == 0 {
            break;
        }
        let mut offset = 0usize;
        while offset < read {
            checkpoint
                .checkpoint(0)
                .await
                .map_err(FsCopyError::Checkpoint)?;
            let wrote = match writer.write_async(&buffer[offset..read]).await {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => result.map_err(FsCopyError::Io)?,
            };
            if wrote > read - offset {
                return Err(FsCopyError::Io(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "filesystem handle reported writing beyond the supplied bytes",
                )));
            }
            if wrote == 0 {
                return Err(FsCopyError::Io(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "filesystem handle did not accept a copy chunk",
                )));
            }
            offset += wrote;
            written_total = written_total
                .checked_add(wrote as u64)
                .ok_or_else(|| FsCopyError::Io(io::Error::other("copied byte count overflow")))?;
            checkpoint
                .checkpoint(wrote as u64)
                .await
                .map_err(FsCopyError::Checkpoint)?;
        }
    }
    checkpoint
        .checkpoint(0)
        .await
        .map_err(FsCopyError::Checkpoint)?;
    writer.flush_async().await.map_err(FsCopyError::Io)?;
    checkpoint
        .checkpoint(0)
        .await
        .map_err(FsCopyError::Checkpoint)?;
    Ok(written_total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryFsProvider;

    struct StopAfterFirstChunk;

    #[async_trait(?Send)]
    impl FsCopyCheckpoint for StopAfterFirstChunk {
        type Error = &'static str;

        async fn checkpoint(&mut self, completed_bytes: u64) -> Result<(), Self::Error> {
            if completed_bytes != 0 {
                Err("stopped after first completed chunk")
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn checkpoint_stops_before_next_chunk_and_retains_partial_owner_output() {
        let provider = MemoryFsProvider::with_current_dir("/copy");
        provider
            .write_project_path("/copy/source.bin", &vec![7u8; 32 * 1024])
            .expect("source");
        let error = futures::executor::block_on(copy_file_on_async(
            &provider,
            Path::new("/copy/source.bin"),
            Path::new("/copy/target.bin"),
            &mut StopAfterFirstChunk,
        ))
        .expect_err("checkpoint must stop the transfer");
        assert!(matches!(
            error,
            FsCopyError::Checkpoint("stopped after first completed chunk")
        ));
        assert_eq!(
            provider.read_project_path("/copy/target.bin").unwrap(),
            vec![7u8; 16 * 1024]
        );
        assert_eq!(
            provider
                .read_project_path("/copy/source.bin")
                .unwrap()
                .len(),
            32 * 1024
        );
    }
}
