//! Resolution of the persisted user-path spelling against one captured
//! filesystem namespace. The application settings service owns persistence;
//! this owner decides only the live search-path segment.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_builtins::UserPathControl;
use runmat_filesystem::{self as fs, CapturedFsResourceLease, FsProvider};

use crate::context::UserPathSetting;

pub struct CapturedUserPathAuthority {
    provider: Arc<dyn FsProvider>,
    cwd: PathBuf,
    home: Option<PathBuf>,
    installation: Option<PathBuf>,
    resource: Option<CapturedFsResourceLease>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedUserPath {
    pub presentation: String,
    pub search_segment: Option<String>,
}

impl CapturedUserPathAuthority {
    /// Pure shape/control admission before provider capture or application
    /// storage. Resolution still validates the captured namespace separately.
    pub fn validate_custom_text(text: &str) -> io::Result<()> {
        UserPathControl::parse(text).map(|_| ()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "custom user path is not one path-list entry",
            )
        })
    }

    pub fn capture() -> io::Result<Self> {
        let (provider, cwd) = fs::current_provider_with_current_dir()?;
        let resource = fs::capture_resource_lease(Arc::clone(&provider));
        Ok(Self {
            provider,
            installation: crate::builtins::io::repl_fs::installation_path::configured_factory_root(
                &cwd,
            ),
            cwd,
            home: crate::builtins::common::fs::home_directory(),
            resource,
        })
    }

    pub fn check_current(&self) -> io::Result<()> {
        if self
            .resource
            .as_ref()
            .is_some_and(|resource| !resource.is_valid())
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "captured user-path filesystem authority was revoked",
            ));
        }
        Ok(())
    }

    pub fn installation_root(&self) -> io::Result<Option<String>> {
        self.installation.as_deref().map(path_text).transpose()
    }

    pub async fn resolve(&self, setting: &UserPathSetting) -> io::Result<ResolvedUserPath> {
        self.check_current()?;
        match setting {
            UserPathSetting::Cleared => Ok(ResolvedUserPath {
                presentation: String::new(),
                search_segment: None,
            }),
            UserPathSetting::Default => self.resolve_default(false).await,
            UserPathSetting::Custom(text) => self.resolve_custom(text),
        }
    }

    /// An explicit reset may create the platform default through the captured
    /// provider. Ordinary startup hydration and query are read-only.
    pub async fn resolve_after_reset(&self) -> io::Result<ResolvedUserPath> {
        self.check_current()?;
        self.resolve_default(true).await
    }

    pub fn resolve_custom(&self, text: &str) -> io::Result<ResolvedUserPath> {
        Self::validate_custom_text(text)?;
        self.check_current()?;
        let expanded = self.expand_home(text)?;
        let path = Path::new(&expanded);
        let resolved = if self.provider.path_namespace().is_absolute(path) {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        };
        Ok(ResolvedUserPath {
            presentation: text.to_owned(),
            search_segment: Some(path_text(&resolved)?),
        })
    }

    async fn resolve_default(&self, create_when_supported: bool) -> io::Result<ResolvedUserPath> {
        let Some(home) = self.home.as_ref() else {
            return Ok(ResolvedUserPath {
                presentation: String::new(),
                search_segment: None,
            });
        };
        let home = if self.provider.path_namespace().is_absolute(home) {
            home.clone()
        } else {
            self.cwd.join(home)
        };
        let documents = home.join("Documents");
        #[cfg(all(not(target_os = "macos"), not(windows), not(target_arch = "wasm32")))]
        {
            let available = match self.provider.metadata(&documents).await {
                Ok(metadata) => metadata.is_dir(),
                Err(source) if source.kind() == io::ErrorKind::NotFound => false,
                Err(source) => return Err(source),
            };
            self.check_current()?;
            if !available {
                return Ok(ResolvedUserPath {
                    presentation: String::new(),
                    search_segment: None,
                });
            }
        }
        let path = documents.join("MATLAB");
        #[cfg(any(target_os = "macos", windows, target_arch = "wasm32"))]
        {
            if create_when_supported {
                match self.provider.metadata(&path).await {
                    Ok(metadata) if metadata.is_dir() => {}
                    Ok(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            "default user path exists but is not a directory",
                        ));
                    }
                    Err(source) if source.kind() == io::ErrorKind::NotFound => {
                        self.check_current()?;
                        self.provider.create_dir_all(&path).await?;
                    }
                    Err(source) => return Err(source),
                }
                self.check_current()?;
            }
        }
        #[cfg(all(not(target_os = "macos"), not(windows), not(target_arch = "wasm32")))]
        let _ = create_when_supported;
        let text = path_text(&path)?;
        Ok(ResolvedUserPath {
            presentation: text.clone(),
            search_segment: Some(text),
        })
    }

    fn expand_home(&self, text: &str) -> io::Result<String> {
        if text == "~" {
            return self
                .home
                .as_deref()
                .ok_or_else(missing_home)
                .and_then(path_text);
        }
        if let Some(suffix) = text.strip_prefix("~/").or_else(|| text.strip_prefix("~\\")) {
            let home = self.home.as_deref().ok_or_else(missing_home)?;
            return path_text(&home.join(suffix));
        }
        Ok(text.to_owned())
    }
}

fn missing_home() -> io::Error {
    io::Error::new(
        io::ErrorKind::NotFound,
        "user home directory is unavailable for tilde expansion",
    )
}

fn path_text(path: &Path) -> io::Result<String> {
    path.to_str().map(str::to_owned).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "user path cannot be represented as a path-list string",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn relative_setting_keeps_public_spelling_and_captured_namespace() {
        let _serial = fs::provider_override_lock();
        let (first, replacement) = ("/selected", "/replacement");
        let _first = fs::replace_provider(Arc::new(fs::MemoryFsProvider::with_current_dir(first)));
        let selected = CapturedUserPathAuthority::capture().expect("selected namespace");
        let _second = fs::replace_provider(Arc::new(fs::MemoryFsProvider::with_current_dir(
            replacement,
        )));
        let resolved = futures::executor::block_on(
            selected.resolve(&UserPathSetting::Custom("MiXeD/functions".into())),
        )
        .expect("captured namespace remains authoritative");
        assert_eq!(resolved.presentation, "MiXeD/functions");
        assert_eq!(
            resolved.search_segment.as_deref(),
            Path::new(first).join("MiXeD/functions").to_str()
        );
    }

    #[cfg(all(not(target_os = "macos"), not(windows), not(target_arch = "wasm32")))]
    #[test]
    fn linux_default_is_empty_until_documents_directory_exists() {
        let provider = Arc::new(fs::MemoryFsProvider::with_current_dir("/project"));
        let selected = CapturedUserPathAuthority {
            provider: provider.clone(),
            cwd: PathBuf::from("/project"),
            home: Some(PathBuf::from("/isolated-home")),
            installation: None,
            resource: None,
        };
        let missing = futures::executor::block_on(selected.resolve(&UserPathSetting::Default))
            .expect("missing Documents is an empty default on Linux");
        assert_eq!(missing.search_segment, None);
        futures::executor::block_on(provider.create_dir_all(Path::new("/isolated-home/Documents")))
            .expect("create only in isolated memory provider");
        let present = futures::executor::block_on(selected.resolve(&UserPathSetting::Default))
            .expect("existing Documents admits default");
        assert_eq!(
            present.search_segment.as_deref(),
            Some("/isolated-home/Documents/MATLAB")
        );
        assert!(
            futures::executor::block_on(
                provider.metadata(Path::new("/isolated-home/Documents/MATLAB"))
            )
            .is_err(),
            "Linux must not create the MATLAB child folder"
        );
    }

    #[cfg(any(target_os = "macos", windows, target_arch = "wasm32"))]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn supported_default_creation_uses_selected_provider_not_host_files() {
        let (cwd, home) = ("/project", "/isolated-home");
        let provider = Arc::new(fs::MemoryFsProvider::with_current_dir(cwd));
        let selected = CapturedUserPathAuthority {
            provider: provider.clone(),
            cwd: PathBuf::from(cwd),
            home: Some(PathBuf::from(home)),
            installation: None,
            resource: None,
        };
        let queried = futures::executor::block_on(selected.resolve(&UserPathSetting::Default))
            .expect("read-only default query");
        let expected = Path::new(home).join("Documents").join("MATLAB");
        assert_eq!(queried.search_segment.as_deref(), expected.to_str());
        assert!(futures::executor::block_on(provider.metadata(&expected)).is_err());
        let resolved = futures::executor::block_on(selected.resolve_after_reset())
            .expect("provider creates default folder");
        assert_eq!(resolved.search_segment.as_deref(), expected.to_str());
        assert!(futures::executor::block_on(provider.metadata(&expected))
            .expect("default folder metadata")
            .is_dir());
    }
}
