//! Invocation-owned filesystem source. Relative paths and provider authority
//! are captured together before any suspension or file effect.

use super::*;
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::Arc;

pub(super) enum OpenedAudioInput {
    Bounded(fs::FsBoundedInput),
    Ordinary(fs::File),
}

impl OpenedAudioInput {
    pub(super) async fn metadata_async(&self) -> io::Result<fs::FsMetadata> {
        match self {
            Self::Bounded(source) => source.metadata_async().await,
            Self::Ordinary(source) => source.metadata_async().await,
        }
    }

    pub(super) fn opened_extent(&self) -> Option<u64> {
        match self {
            Self::Bounded(source) => Some(source.opened_extent()),
            Self::Ordinary(_) => None,
        }
    }
}

impl Read for OpenedAudioInput {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Bounded(source) => source.read(bytes),
            Self::Ordinary(source) => source.read(bytes),
        }
    }
}

impl Seek for OpenedAudioInput {
    fn seek(&mut self, offset: SeekFrom) -> io::Result<u64> {
        match self {
            Self::Bounded(source) => source.seek(offset),
            Self::Ordinary(source) => source.seek(offset),
        }
    }
}

pub(super) struct FsInputLease {
    pub(super) provider: Arc<dyn fs::FsProvider>,
    pub(super) path: PathBuf,
    pub(super) resource: Option<fs::CapturedFsResourceLease>,
}

impl FsInputLease {
    pub(super) fn capture(
        path: PathBuf,
        error: &'static BuiltinErrorDescriptor,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        let (provider, path) = fs::capture_provider_path(&path).map_err(|source| {
            build_runtime_error(format!(
                "{builtin}: unable to resolve current folder: {source}"
            ))
            .with_builtin(builtin)
            .with_identifier(error.identifier.unwrap_or("RunMat:audio:Io"))
            .with_source(source)
            .build()
        })?;
        Ok(Self {
            resource: fs::capture_resource_lease(provider.clone()),
            provider,
            path,
        })
    }

    pub(super) async fn open(&self) -> std::io::Result<OpenedAudioInput> {
        if self
            .resource
            .as_ref()
            .is_some_and(|lease| !lease.is_valid())
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "captured audio filesystem resource authority was revoked",
            ));
        }
        if let Some(source) = self.provider.open_bounded_input(&self.path).await? {
            let lease = self.resource.as_ref().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::Unsupported,
                    "bounded audio source has no captured filesystem resource authority",
                )
            })?;
            if !lease.is_valid() || source.resource_revision() != lease.contract().revision {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "opened audio source has a changed filesystem resource authority",
                ));
            }
            return Ok(OpenedAudioInput::Bounded(source));
        }
        let file = fs::File::open_async_on(self.provider.as_ref(), &self.path).await?;
        if self
            .resource
            .as_ref()
            .is_some_and(|lease| !lease.is_valid())
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "captured audio filesystem resource authority was revoked during open",
            ));
        }
        Ok(OpenedAudioInput::Ordinary(file))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn relative_audio_source_keeps_original_provider_and_cwd() {
        let _lock = fs::provider_override_lock();
        let selected = Arc::new(fs::MemoryFsProvider::with_current_dir("/selected"));
        let replacement = Arc::new(fs::MemoryFsProvider::with_current_dir("/replacement"));
        selected
            .write_project_path("/selected/tone.wav", b"selected")
            .expect("selected file");
        replacement
            .write_project_path("/replacement/tone.wav", b"replacement")
            .expect("replacement file");
        let _selected_guard = fs::replace_provider(selected);
        fs::set_current_dir("/selected").expect("selected cwd");
        let input = FsInputLease::capture(
            PathBuf::from("tone.wav"),
            &AUDIOREAD_ERROR_IO,
            AUDIOREAD_BUILTIN_NAME,
        )
        .expect("capture");
        assert_eq!(input.path, PathBuf::from("/selected/tone.wav"));
        let _replacement_guard = fs::replace_provider(replacement);
        fs::set_current_dir("/replacement").expect("replacement cwd");
        let mut opened = block_on(input.open()).expect("captured source open");
        assert_eq!(
            opened.opened_extent(),
            Some(8),
            "MemoryFS now supplies its own bounded opened snapshot, not a native memory-peak certificate"
        );
        let mut opened_bytes = [0; 8];
        opened
            .read_exact(&mut opened_bytes)
            .expect("opened snapshot");
        assert_eq!(&opened_bytes, b"selected");
        drop(opened);
        let bytes = block_on(input.provider.read(&input.path)).expect("captured read");
        assert_eq!(bytes, b"selected");
    }
}
