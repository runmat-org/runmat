//! Invocation-owned filesystem source. Relative paths and provider authority
//! are captured together before any suspension or file effect.

use super::*;
use std::sync::Arc;

pub(super) struct FsInputLease {
    pub(super) provider: Arc<dyn fs::FsProvider>,
    pub(super) path: PathBuf,
}

impl FsInputLease {
    pub(super) fn capture(
        path: PathBuf,
        error: &'static BuiltinErrorDescriptor,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        if path.is_absolute() {
            return Ok(Self {
                provider: fs::current_provider(),
                path,
            });
        }
        let (provider, current_dir) =
            fs::current_provider_with_current_dir().map_err(|source| {
                build_runtime_error(format!(
                    "{builtin}: unable to resolve current folder: {source}"
                ))
                .with_builtin(builtin)
                .with_identifier(error.identifier.unwrap_or("RunMat:audio:Io"))
                .with_source(source)
                .build()
            })?;
        Ok(Self {
            provider,
            path: current_dir.join(path),
        })
    }

    pub(super) async fn open(&self) -> std::io::Result<fs::File> {
        fs::File::open_async_on(self.provider.as_ref(), &self.path).await
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
        let bytes = block_on(input.provider.read(&input.path)).expect("captured read");
        assert_eq!(bytes, b"selected");
    }
}
