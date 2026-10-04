//! One filesystem and working-directory authority for a public data path call.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_filesystem::{self as fs, FsProvider};

use crate::data::{data_error, data_error_with_source};
use crate::BuiltinResult;

pub(super) struct DataPathAuthority {
    provider: Arc<dyn FsProvider>,
    cwd: Option<PathBuf>,
}

impl DataPathAuthority {
    /// The parsed path set is fixed before any filesystem effect. Absolute-only
    /// calls do not need a working directory, including when it is unavailable.
    pub(super) fn capture(paths: &[&str]) -> BuiltinResult<Self> {
        Self::capture_with(
            paths,
            fs::current_provider,
            fs::current_provider_with_current_dir,
        )
    }

    fn capture_with(
        paths: &[&str],
        absolute_provider: impl FnOnce() -> Arc<dyn FsProvider>,
        relative_provider_and_dir: impl FnOnce() -> std::io::Result<(Arc<dyn FsProvider>, PathBuf)>,
    ) -> BuiltinResult<Self> {
        if paths.iter().all(|path| Path::new(path).is_absolute()) {
            return Ok(Self {
                provider: absolute_provider(),
                cwd: None,
            });
        }
        let (provider, cwd) = relative_provider_and_dir().map_err(|source| {
            data_error_with_source(
                format!("data: failed to capture current directory: {source}"),
                source,
            )
        })?;
        Ok(Self {
            provider,
            cwd: Some(cwd),
        })
    }

    pub(super) fn resolve(&self, path: impl AsRef<Path>) -> BuiltinResult<PathBuf> {
        let path = path.as_ref();
        if path.is_absolute() {
            Ok(path.to_path_buf())
        } else {
            self.cwd.as_ref().map(|cwd| cwd.join(path)).ok_or_else(|| {
                data_error("data: relative path was not included in captured authority")
            })
        }
    }

    pub(super) fn provider(&self) -> &dyn FsProvider {
        self.provider.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{CharArray, StringArray, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dataset_path_text_uses_canonical_scalar_decoder() {
        let singleton =
            Value::StringArray(StringArray::new(vec!["one.data".into()], vec![1, 1]).unwrap());
        assert_eq!(
            crate::data::parse_string(&singleton, "data.open path").unwrap(),
            "one.data"
        );

        let multiple = Value::StringArray(
            StringArray::new(vec!["one.data".into(), "two.data".into()], vec![1, 2]).unwrap(),
        );
        assert!(crate::data::parse_string(&multiple, "data.open path").is_err());

        let paged =
            Value::CharArray(CharArray::new_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap());
        let error = crate::data::parse_string(&paged, "data.open path").unwrap_err();
        assert!(error.to_string().contains("expected character row vector"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn absolute_path_does_not_require_working_directory() {
        let provider = Arc::new(fs::MemoryFsProvider::new());
        provider
            .write_project_path("/data/source.data/manifest.json", b"captured")
            .expect("fixture manifest");
        let authority = DataPathAuthority::capture_with(
            &["/data/source.data", "/data/target.data"],
            || provider,
            || Err(std::io::Error::other("cwd unavailable")),
        )
        .expect("absolute-only authority");
        assert_eq!(
            authority.resolve("/data/source.data").unwrap(),
            PathBuf::from("/data/source.data")
        );
        assert_eq!(
            authority.resolve("/data/target.data").unwrap(),
            PathBuf::from("/data/target.data")
        );
        let bytes = futures::executor::block_on(
            authority.provider().read(
                &authority
                    .resolve("/data/source.data/manifest.json")
                    .unwrap(),
            ),
        )
        .expect("read through the captured provider without cwd");
        assert_eq!(bytes, b"captured");
        assert!(authority.resolve("relative.data").is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn relative_path_requires_captured_directory_and_retains_source() {
        let provider: Arc<dyn FsProvider> = Arc::new(fs::MemoryFsProvider::new());
        let error = DataPathAuthority::capture_with(
            &["/data/source.data", "target.data"],
            || provider,
            || Err(std::io::Error::other("cwd unavailable")),
        )
        .err()
        .expect("relative path must fail before effects");
        assert!(error.to_string().contains("cwd unavailable"));
        assert!(std::error::Error::source(&error).is_some());
    }
}
