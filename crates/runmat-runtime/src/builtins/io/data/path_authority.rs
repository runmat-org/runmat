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
    pub(super) fn capture(paths: &[impl AsRef<Path>]) -> BuiltinResult<Self> {
        let paths = paths
            .iter()
            .map(|path| path.as_ref())
            .collect::<Vec<&Path>>();
        let (provider, cwd) = fs::capture_provider_for_paths(&paths).map_err(|source| {
            data_error_with_source(
                format!("data: failed to capture current directory: {source}"),
                source,
            )
        })?;
        Ok(Self { provider, cwd })
    }

    #[cfg(test)]
    fn capture_with(
        paths: &[impl AsRef<Path>],
        absolute_provider: impl FnOnce() -> Arc<dyn FsProvider>,
        relative_provider_and_dir: impl FnOnce() -> std::io::Result<(Arc<dyn FsProvider>, PathBuf)>,
    ) -> BuiltinResult<Self> {
        let provider = absolute_provider();
        if paths
            .iter()
            .all(|path| provider.path_namespace().is_absolute(path.as_ref()))
        {
            return Ok(Self {
                provider,
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
        if self.provider.path_namespace().is_absolute(path) {
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

    pub(super) fn provider_arc(&self) -> &Arc<dyn FsProvider> {
        &self.provider
    }

    /// A recursive copy into its own tree would discover the destination it
    /// just created and recurse without bound. Canonicalize the source and the
    /// destination's existing prefixes through this same provider so
    /// symlink aliases and `..` components cannot evade the pre-effect check.
    pub(super) async fn admit_tree_destination(
        &self,
        source: &Path,
        destination: &Path,
    ) -> BuiltinResult<()> {
        let source_canonical = self.provider.canonicalize(source).await.map_err(|cause| {
            data_error_with_source(
                format!(
                    "data: failed to resolve source '{}': {cause}",
                    source.display()
                ),
                cause,
            )
        })?;
        let destination_canonical = self.canonical_destination(destination).await?;
        if destination_canonical.starts_with(&source_canonical) {
            return Err(data_error(
                "data: destination must not be the source dataset or one of its descendants",
            ));
        }
        Ok(())
    }

    async fn canonical_destination(&self, destination: &Path) -> BuiltinResult<PathBuf> {
        let root = destination
            .ancestors()
            .last()
            .ok_or_else(|| data_error("data: destination has no filesystem root"))?;
        let mut canonical = self.provider.canonicalize(root).await.map_err(|cause| {
            data_error_with_source(
                format!(
                    "data: failed to resolve destination root '{}': {cause}",
                    root.display()
                ),
                cause,
            )
        })?;
        let suffix = destination
            .strip_prefix(root)
            .map_err(|_| data_error("data: destination is outside its filesystem root"))?;
        for component in suffix.components() {
            match component {
                std::path::Component::CurDir => {}
                std::path::Component::Normal(part) => {
                    canonical.push(part);
                    canonical = self.canonicalize_if_existing(&canonical).await?;
                }
                std::path::Component::ParentDir => {
                    if !canonical.pop() {
                        return Err(data_error("data: destination escapes filesystem root"));
                    }
                    canonical = self.canonicalize_if_existing(&canonical).await?;
                }
                std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                    return Err(data_error("data: destination has an invalid path suffix"));
                }
            }
        }
        Ok(canonical)
    }

    async fn canonicalize_if_existing(&self, path: &Path) -> BuiltinResult<PathBuf> {
        match self.provider.canonicalize(path).await {
            Ok(path) => Ok(path),
            Err(cause) if cause.kind() == std::io::ErrorKind::NotFound => Ok(path.to_path_buf()),
            Err(cause) => Err(data_error_with_source(
                format!(
                    "data: failed to resolve destination '{}': {cause}",
                    path.display()
                ),
                cause,
            )),
        }
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

        let paged = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap(),
        );
        let error = crate::data::parse_string(&paged, "data.open path").unwrap_err();
        assert!(error.to_string().contains("expected character row vector"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn absolute_path_does_not_require_working_directory() {
        fn verify(provider: Arc<dyn FsProvider>, source: &Path, target: &Path) {
            assert!(provider.path_namespace().is_absolute(source));
            assert!(provider.path_namespace().is_absolute(target));
            let source_text = source.to_str().expect("fixture path is Unicode");
            let target_text = target.to_str().expect("fixture path is Unicode");
            let authority = DataPathAuthority::capture_with(
                &[source_text, target_text],
                || provider,
                || Err(std::io::Error::other("cwd unavailable")),
            )
            .expect("absolute-only authority");
            assert_eq!(authority.resolve(source_text).unwrap(), source);
            assert_eq!(authority.resolve(target_text).unwrap(), target);
            let bytes = futures::executor::block_on(authority.provider().read(source))
                .expect("read through captured provider without cwd");
            assert_eq!(bytes, b"captured");
            assert!(authority.resolve("relative.data").is_err());
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let fixture = tempfile::tempdir().expect("temporary native root");
            let source = fixture.path().join("source.data");
            let target = fixture.path().join("target.data");
            std::fs::write(&source, b"captured").expect("fixture source");
            verify(Arc::new(fs::NativeFsProvider), &source, &target);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let provider = Arc::new(fs::MemoryFsProvider::new());
            provider
                .write_project_path("/data/source.data", b"captured")
                .expect("browser fixture source");
            verify(
                provider,
                Path::new("/data/source.data"),
                Path::new("/data/target.data"),
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let provider = Arc::new(fs::MemoryFsProvider::new());
            provider
                .write_project_path("/data/source.data", b"captured")
                .expect("virtual fixture source");
            verify(
                provider,
                Path::new("/data/source.data"),
                Path::new("/data/target.data"),
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn relative_path_requires_captured_directory_and_retains_source() {
        const ABSOLUTE: &str = "/data/source.data";
        let provider: Arc<dyn FsProvider> = Arc::new(fs::MemoryFsProvider::new());
        assert!(provider.path_namespace().is_absolute(Path::new(ABSOLUTE)));
        for paths in [[ABSOLUTE, "target.data"], ["source.data", "target.data"]] {
            let error = DataPathAuthority::capture_with(
                &paths,
                || provider.clone(),
                || Err(std::io::Error::other("cwd unavailable")),
            )
            .err()
            .expect("relative path must fail before effects");
            assert!(error.to_string().contains("cwd unavailable"));
            assert!(std::error::Error::source(&error).is_some());
        }
    }
}
