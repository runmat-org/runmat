use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ProjectContext {
    manifest_path: PathBuf,
    all_source_files: Vec<PathBuf>,
    frozen: runmat_package::FrozenProject,
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
        let start = match source_name.map(PathBuf::from) {
            Some(path) if path.is_absolute() => {
                if is_file_async(&path).await {
                    path.parent().map(Path::to_path_buf)?
                } else {
                    path
                }
            }
            _ => runmat_filesystem::current_dir().ok()?,
        };
        let frozen = runmat_package::discover_frozen_project_from_async(&start, Default::default())
            .await
            .ok()?;
        frozen.map(Self::from_frozen)
    }

    pub fn all_source_files(&self) -> &[PathBuf] {
        &self.all_source_files
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

    fn from_frozen(frozen: runmat_package::FrozenProject) -> Self {
        let manifest_path = frozen.manifest_path.clone();
        let mut all_source_files = frozen
            .all_sources()
            .map(|(_, path)| path.clone())
            .collect::<Vec<_>>();
        all_source_files.sort();
        all_source_files.dedup();
        Self {
            manifest_path,
            all_source_files,
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::ProjectContext;
    use std::fs;
    use tempfile::TempDir;

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
}
