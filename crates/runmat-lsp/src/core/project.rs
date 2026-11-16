#[cfg(target_arch = "wasm32")]
use runmat_thread_local::runmat_thread_local;
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
use std::path::{Path, PathBuf};

#[cfg(target_arch = "wasm32")]
runmat_thread_local! {
    static INSTALLED_HANDOFF: RefCell<Option<runmat_package::FrozenProjectHandoff>> =
        const { RefCell::new(None) };
}

#[derive(Clone, Debug)]
pub struct ProjectContext {
    manifest_path: PathBuf,
    visible_source_files: Vec<PathBuf>,
    source_catalog: runmat_package::DiscoveredSourceSymbols,
    frozen: runmat_package::FrozenProject,
}

/// Requester-visible sources for language tooling, with or without a manifest.
/// Frozen project admission remains the responsibility of `ProjectContext`.
pub struct SourceContext {
    catalog: runmat_package::DiscoveredSourceSymbols,
    files: Vec<PathBuf>,
}

impl SourceContext {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn discover_from_source_name(source_name: Option<&str>) -> Option<Self> {
        futures::executor::block_on(Self::discover_from_source_name_async(source_name))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn discover_from_source_name(_: Option<&str>) -> Option<Self> {
        None
    }

    pub async fn discover_from_source_name_async(source_name: Option<&str>) -> Option<Self> {
        Self::try_discover_from_source_name_async(source_name)
            .await
            .ok()
            .flatten()
    }

    pub(crate) async fn try_discover_from_source_name_async(
        source_name: Option<&str>,
    ) -> Result<Option<Self>, runmat_package::DiscoverSourceSymbolsError> {
        #[cfg(target_arch = "wasm32")]
        if let Some(project) = ProjectContext::installed_from_source_name(source_name) {
            return Ok(Some(Self::from_catalog(project.source_catalog().clone())));
        }
        let Some(source_name) = source_name else {
            let Ok(directory) = runmat_filesystem::current_dir() else {
                return Ok(None);
            };
            return Ok(runmat_package::discover_frozen_project_from_async(
                &directory,
                Default::default(),
            )
            .await?
            .map(|frozen| {
                Self::from_catalog(runmat_package::source_symbols_from_frozen(
                    &frozen, &directory,
                ))
            }));
        };
        let path = Path::new(source_name);
        let directory = if path.is_absolute() {
            path.parent().map(Path::to_path_buf)
        } else {
            runmat_filesystem::current_dir().ok()
        };
        let Some(directory) = directory else {
            return Ok(None);
        };
        Ok(
            runmat_package::discover_source_symbols_from_source_name_async(source_name, &directory)
                .await?
                .map(Self::from_catalog),
        )
    }

    fn from_catalog(catalog: runmat_package::DiscoveredSourceSymbols) -> Self {
        let mut files = catalog
            .definitions
            .iter()
            .filter(|definition| catalog.symbols.contains(&definition.name))
            .map(|definition| definition.source_path.clone())
            .collect::<Vec<_>>();
        files.sort();
        files.dedup();
        Self { catalog, files }
    }

    pub fn visible_source_files(&self) -> &[PathBuf] {
        &self.files
    }
    pub(crate) fn into_catalog(self) -> runmat_package::DiscoveredSourceSymbols {
        self.catalog
    }
    pub fn definition_lookup_names<'a>(
        &'a self,
        source_path: &Path,
        requested_name: &'a str,
    ) -> Vec<&'a str> {
        definition_lookup_names(&self.catalog, source_path, requested_name)
    }
}

fn definition_lookup_names<'a>(
    catalog: &'a runmat_package::DiscoveredSourceSymbols,
    source_path: &Path,
    requested_name: &'a str,
) -> Vec<&'a str> {
    let mut names = vec![requested_name];
    for definition in catalog.definitions.iter().filter(|definition| {
        definition.name == requested_name && definition.source_path == source_path
    }) {
        let leaf = definition
            .qualified_name
            .rsplit_once('.')
            .map(|(_, leaf)| leaf)
            .unwrap_or(&definition.qualified_name);
        if !names.contains(&leaf) {
            names.push(leaf);
        }
    }
    names
}

impl ProjectContext {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn discover_from_source_name(source_name: Option<&str>) -> Option<Self> {
        futures::executor::block_on(Self::discover_from_source_name_async(source_name))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn discover_from_source_name(_source_name: Option<&str>) -> Option<Self> {
        None
    }

    pub async fn discover_from_source_name_async(source_name: Option<&str>) -> Option<Self> {
        let requester = source_name.map(PathBuf::from);
        #[cfg(target_arch = "wasm32")]
        if let Some(project) = Self::installed_from_source_name(source_name) {
            return Some(project);
        }
        let start = match requester.as_ref() {
            Some(path) if path.is_absolute() => {
                if is_file_async(path).await {
                    path.parent().map(Path::to_path_buf)?
                } else {
                    path.clone()
                }
            }
            _ => runmat_filesystem::current_dir().ok()?,
        };
        let frozen = runmat_package::discover_frozen_project_from_async(&start, Default::default())
            .await
            .ok()?;
        frozen.and_then(|frozen| {
            let requester = requester.as_deref().unwrap_or(&start);
            Self::from_handoff(runmat_package::FrozenProjectHandoff::new(frozen), requester).ok()
        })
    }

    pub fn visible_source_files(&self) -> &[PathBuf] {
        &self.visible_source_files
    }

    pub fn contains_visible_source(&self, path: &Path) -> bool {
        self.visible_source_files
            .iter()
            .any(|visible| paths_equivalent(visible, path))
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn graph_digest(&self) -> &runmat_package::ContentDigest {
        self.frozen.graph_digest()
    }

    pub fn source_revision(&self) -> &runmat_package::ContentDigest {
        self.frozen.source_revision()
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn installed_from_source_name(source_name: Option<&str>) -> Option<Self> {
        let handoff = INSTALLED_HANDOFF.with(|slot| slot.borrow().clone())?;
        Self::from_handoff(handoff, Path::new(source_name.unwrap_or("/"))).ok()
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn source_catalog(&self) -> &runmat_package::DiscoveredSourceSymbols {
        &self.source_catalog
    }

    pub fn from_handoff(
        handoff: runmat_package::FrozenProjectHandoff,
        requester: &Path,
    ) -> Result<Self, runmat_package::FrozenProjectHandoffError> {
        handoff.validate()?;
        Ok(Self::from_frozen(handoff.into_project(), requester))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn install_handoff(
        handoff: runmat_package::FrozenProjectHandoff,
    ) -> Result<runmat_package::ProjectRevision, runmat_package::FrozenProjectHandoffError> {
        handoff.validate()?;
        let revision = handoff.revision();
        INSTALLED_HANDOFF.with(|slot| {
            slot.replace(Some(handoff));
        });
        Ok(revision)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn clear_installed_handoff() {
        INSTALLED_HANDOFF.with(|slot| {
            slot.replace(None);
        });
    }

    pub fn definition_lookup_names<'a>(
        &'a self,
        source_path: &Path,
        requested_name: &'a str,
    ) -> Vec<&'a str> {
        definition_lookup_names(&self.source_catalog, source_path, requested_name)
    }

    fn from_frozen(frozen: runmat_package::FrozenProject, requester: &Path) -> Self {
        let manifest_path = frozen.manifest_path.clone();
        let source_catalog = runmat_package::source_symbols_from_frozen(&frozen, requester);
        let mut visible_source_files = frozen
            .visible_sources(requester)
            .into_iter()
            .filter(|visible| visible.directly_visible)
            .map(|visible| visible.access_path.to_path_buf())
            .collect::<Vec<_>>();
        visible_source_files.sort();
        visible_source_files.dedup();
        Self {
            manifest_path,
            visible_source_files,
            source_catalog,
            frozen,
        }
    }
}

async fn is_file_async(path: &Path) -> bool {
    runmat_filesystem::metadata_async(path)
        .await
        .map(|meta| meta.is_file())
        .unwrap_or(false)
}

fn paths_equivalent(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
    #[cfg(not(target_arch = "wasm32"))]
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::{ProjectContext, SourceContext};
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn project_context_revision_tracks_graph_declared_source_content() {
        let temp = TempDir::new().unwrap();
        fs::create_dir_all(temp.path().join("src")).unwrap();
        let source = temp.path().join("src/main.m");
        fs::write(&source, "x = 1;\n").unwrap();
        fs::write(
            temp.path().join("runmat.toml"),
            r#"
[package]
name = "revision-fixture"

[sources]
roots = ["src"]
"#,
        )
        .unwrap();
        let before =
            ProjectContext::discover_from_source_name(source.to_str()).expect("project context");
        fs::write(&source, "x = 2;\n").unwrap();
        let after =
            ProjectContext::discover_from_source_name(source.to_str()).expect("project context");
        assert_ne!(before.graph_digest(), after.graph_digest());
        assert_ne!(before.source_revision(), after.source_revision());
    }

    #[test]
    fn project_context_only_exposes_requester_visible_sources() {
        let temp = TempDir::new().unwrap();
        write(
            &temp.path().join("runmat.toml"),
            r#"
[package]
name = "root"

[sources]
roots = ["src"]

[dependencies]
middle = { path = "deps/middle" }
"#,
        );
        write(&temp.path().join("src/main.m"), "middle.api();");
        write(
            &temp.path().join("src/private/root_secret.m"),
            "function root_secret(); end",
        );
        write(
            &temp.path().join("deps/middle/runmat.toml"),
            r#"
[package]
name = "middle"

[sources]
roots = ["src"]

[dependencies]
leaf = { path = "deps/leaf" }
"#,
        );
        write(
            &temp.path().join("deps/middle/src/api.m"),
            "function api(); end",
        );
        write(
            &temp.path().join("deps/middle/src/private/dep_secret.m"),
            "function dep_secret(); end",
        );
        write(
            &temp.path().join("deps/middle/deps/leaf/runmat.toml"),
            r#"
[package]
name = "leaf"

[sources]
roots = ["src"]
"#,
        );
        write(
            &temp.path().join("deps/middle/deps/leaf/src/transitive.m"),
            "function transitive(); end",
        );

        let source = temp.path().join("src/main.m");
        let context =
            ProjectContext::discover_from_source_name(source.to_str()).expect("project context");
        let visible = context.visible_source_files();
        assert!(visible.iter().any(|path| path.ends_with("src/main.m")));
        assert!(visible
            .iter()
            .any(|path| path.ends_with("src/private/root_secret.m")));
        assert!(visible
            .iter()
            .any(|path| path.ends_with("deps/middle/src/api.m")));
        assert!(!visible.iter().any(|path| path.ends_with("dep_secret.m")));
        assert!(!visible.iter().any(|path| path.ends_with("transitive.m")));
        let middle_api = visible
            .iter()
            .find(|path| path.ends_with("deps/middle/src/api.m"))
            .unwrap();
        assert_eq!(
            context.definition_lookup_names(middle_api, "middle.api"),
            vec!["middle.api", "api"]
        );
        let sources = SourceContext::discover_from_source_name(source.to_str())
            .expect("language source context");
        assert_eq!(sources.visible_source_files(), visible);
        assert_eq!(
            sources.definition_lookup_names(middle_api, "middle.api"),
            vec!["middle.api", "api"]
        );
    }
}
