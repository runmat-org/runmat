use std::path::{Path, PathBuf};

use runmat_analysis_language::{
    bind_program, check_program, CheckedFeaProgram, ExecutionProfileFacts, FeaCompiler,
    FeaModuleGraph, FunctionFacts, FunctionRequest, LogicalSourcePath, ModuleGraphRequest,
};

use super::{FilesystemFeaHost, ProjectFeaHost};

/// A fully loaded `.fea` module graph awaiting immutable project facts.
///
/// Preparing source performs only confined source loading and parsing. Hosts can then compile the
/// exact imported `.m` functions and resolve named execution profiles asynchronously before the
/// graph is bound and checked. The graph is consumed exactly once so no later phase can silently
/// observe different source bytes.
#[derive(Debug)]
pub struct PreparedFeaSource {
    graph: FeaModuleGraph,
    source_root: PathBuf,
}

impl PreparedFeaSource {
    pub fn root_module(&self) -> &LogicalSourcePath {
        &self.graph.root
    }

    pub fn source_root(&self) -> &Path {
        &self.source_root
    }

    pub fn function_requests(&self) -> Result<Vec<FunctionRequest>, String> {
        self.graph
            .function_requests()
            .map_err(|error| error.to_string())
    }

    pub fn source_dependencies(
        &self,
    ) -> Result<std::collections::BTreeSet<LogicalSourcePath>, String> {
        self.graph
            .source_dependencies()
            .map_err(|error| error.to_string())
    }

    pub fn check(
        self,
        functions: impl IntoIterator<Item = FunctionFacts>,
        profiles: impl IntoIterator<Item = ExecutionProfileFacts>,
    ) -> Result<CheckedFeaSource, String> {
        let mut profiles = profiles.into_iter().collect::<Vec<_>>();
        profiles.sort_by(|left, right| left.profile.cmp(&right.profile));
        let host = ProjectFeaHost::new(&self.source_root, functions, profiles.clone())?;
        if self.graph.catalog_fingerprint
            != runmat_analysis_language::AnalysisOperationCatalog::fingerprint(&host)
        {
            return Err(
                "FEA operation catalog changed between source preparation and checking".into(),
            );
        }
        let root_module = self.graph.root.clone();
        let bound = bind_program(&host, self.graph);
        let mut program = check_program(&host, bound);
        super::admission::append_admission_diagnostics(&mut program, &root_module);
        Ok(CheckedFeaSource {
            program,
            root_module,
            source_root: self.source_root,
            execution_profile_facts: profiles,
        })
    }
}

/// Canonical checked source product shared by CLI, editor, browser, and runtime adapters.
#[derive(Debug)]
pub struct CheckedFeaSource {
    pub(super) execution_profile_facts: Vec<ExecutionProfileFacts>,
    pub program: CheckedFeaProgram,
    pub root_module: LogicalSourcePath,
    pub source_root: PathBuf,
}

pub fn check_fea_path(path: &Path) -> Result<CheckedFeaProgram, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    check_fea_path_in_root(parent, path)
}

pub fn check_fea_path_in_root(
    source_root: &Path,
    path: &Path,
) -> Result<CheckedFeaProgram, String> {
    Ok(prepare_fea_path_in_root(source_root, path)?
        .check(Vec::new(), Vec::new())?
        .program)
}

pub fn prepare_fea_path_in_root(
    source_root: &Path,
    path: &Path,
) -> Result<PreparedFeaSource, String> {
    prepare_fea_sources_in_root(source_root, path, &[])
}

/// Check the supplied source as one module in a filesystem-backed project graph.
///
/// The supplied bytes replace only `path`; imports and referenced geometry continue to resolve
/// below `source_root`. This is the canonical entrypoint for editor buffers that have not been
/// saved yet.
pub fn check_fea_source_in_root(
    source_root: &Path,
    path: &Path,
    source: impl Into<Vec<u8>>,
) -> Result<CheckedFeaProgram, String> {
    let physical_path = entry_path(source_root, path);
    check_fea_sources_in_root(
        source_root,
        &physical_path,
        &[(physical_path.clone(), source.into())],
    )
}

/// Check one entry module against an exact set of unsaved project buffers.
///
/// Each overlay must remain below `source_root`. The entry module must be present either in the
/// overlay set or on disk. This lets an editor check a coherent in-memory module graph without
/// mutating the project filesystem.
pub fn check_fea_sources_in_root(
    source_root: &Path,
    path: &Path,
    overlays: &[(PathBuf, Vec<u8>)],
) -> Result<CheckedFeaProgram, String> {
    Ok(prepare_fea_sources_in_root(source_root, path, overlays)?
        .check(Vec::new(), Vec::new())?
        .program)
}

pub fn prepare_fea_sources_in_root(
    source_root: &Path,
    path: &Path,
    overlays: &[(PathBuf, Vec<u8>)],
) -> Result<PreparedFeaSource, String> {
    let physical_path = entry_path(source_root, path);
    let mut host = FilesystemFeaHost::new(source_root)?;
    for (overlay_path, source) in overlays {
        let overlay_path = entry_path(source_root, overlay_path);
        host = host.with_source_overlay(&overlay_path, source.clone())?;
    }
    prepare_with_host(&host, source_root, &physical_path)
}

fn prepare_with_host(
    host: &FilesystemFeaHost,
    source_root: &Path,
    path: &Path,
) -> Result<PreparedFeaSource, String> {
    let physical_path = entry_path(source_root, path);
    let root = host.load_physical_module(&physical_path)?;
    let graph = FeaCompiler::new(host).build_module_graph(ModuleGraphRequest {
        source_root: LogicalSourcePath::new(host.logical_root().to_string())
            .map_err(|error| error.to_string())?,
        root,
    });
    Ok(PreparedFeaSource {
        graph,
        source_root: source_root.to_path_buf(),
    })
}

fn entry_path(source_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() || path.starts_with(source_root) {
        path.to_path_buf()
    } else {
        source_root.join(path)
    }
}
