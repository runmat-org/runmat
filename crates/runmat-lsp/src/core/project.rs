use runmat_config::project::{
    build_project_composition_graph_async, discover_project_manifest_from_async,
    ProjectCompositionGraph,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ProjectContext {
    manifest_path: PathBuf,
    all_source_files: Vec<PathBuf>,
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
        let manifest_path = discover_project_manifest_from_async(&start).await?;
        let graph = build_project_composition_graph_async(&manifest_path)
            .await
            .ok()?;
        Some(Self::from_graph(manifest_path, graph))
    }

    pub fn all_source_files(&self) -> &[PathBuf] {
        &self.all_source_files
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    fn from_graph(manifest_path: PathBuf, graph: ProjectCompositionGraph) -> Self {
        let mut all_source_files = Vec::new();

        for package in graph.packages.values() {
            for source in &package.source_index.files {
                let source_file = package
                    .project_root
                    .join(&source.source_root)
                    .join(&source.relative_path);
                all_source_files.push(source_file.clone());
            }
        }
        all_source_files.sort();
        all_source_files.dedup();
        Self {
            manifest_path,
            all_source_files,
        }
    }
}

async fn is_file_async(path: &Path) -> bool {
    runmat_filesystem::metadata_async(path)
        .await
        .map(|meta| meta.is_file())
        .unwrap_or(false)
}
