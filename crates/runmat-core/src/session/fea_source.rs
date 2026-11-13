use std::path::{Path, PathBuf};

use runmat_runtime::analysis::{
    FeaSourceCheckResult, FeaSourceSnapshotBuilder, FunctionFacts, FunctionRequest,
    LogicalSourcePath,
};

use super::RunMatSession;

/// Checked `.fea` source and the immutable project context used to check it.
///
/// Native execution consumes this product so it cannot rediscover a different project graph,
/// execution-profile catalog, or entry path after source checking succeeds.
#[derive(Debug)]
pub struct CheckedFeaSessionSource {
    pub(super) checked: runmat_runtime::analysis::CheckedFeaSource,
    pub(super) physical_path: PathBuf,
    pub(super) frozen_project: Option<runmat_package::FrozenProject>,
    pub(super) project_manifest: Option<runmat_config::project::ProjectManifest>,
    pub(super) program_environment: runmat_execution::ProgramEnvironment,
    pub(super) compatibility_mode: crate::CompatMode,
    pub(super) captured: Option<std::sync::Arc<std::collections::BTreeMap<PathBuf, Vec<u8>>>>,
}

impl CheckedFeaSessionSource {
    /// Canonical source/program identity shared by every execution host.
    pub fn program_revision(&self) -> Result<runmat_execution::ProgramRevision, String> {
        let (graph, source_digest) = match self.frozen_project() {
            Some(project) => (
                runmat_execution::Digest::from_bytes(*project.graph_digest().bytes()),
                runmat_execution::Digest::from_bytes(*project.source_revision().bytes()),
            ),
            None => {
                let mut graph = Vec::new();
                for (path, digest) in self.program.bound.graph.source_digests() {
                    graph.extend_from_slice(path.as_str().as_bytes());
                    graph.push(0);
                    graph.extend_from_slice(&digest.0);
                }
                graph.extend_from_slice(&self.check_result().revision.0);
                let root = self
                    .program
                    .bound
                    .graph
                    .modules
                    .get(&self.root_module)
                    .ok_or_else(|| "checked root module is absent".to_string())?;
                (
                    runmat_execution::Digest::sha256(graph),
                    runmat_execution::Digest::from_bytes(root.source.source_digest.0),
                )
            }
        };
        runmat_execution::ProgramRevision::new(
            graph,
            source_digest,
            self.program_environment().clone(),
        )
        .map_err(|error| error.to_string())
    }

    /// The complete immutable view used for candidate checking, when present.
    /// Materialization must not fall back to live files if a capture lacks a path.
    pub fn captured_sources(&self) -> Option<&std::collections::BTreeMap<PathBuf, Vec<u8>>> {
        self.captured.as_deref()
    }

    pub async fn read_source(&self, path: &Path) -> Result<Vec<u8>, String> {
        match self.captured_sources() {
            Some(files) => files.get(path).cloned().ok_or_else(|| {
                format!(
                    "source is absent from the checked capture: {}",
                    path.display()
                )
            }),
            None => runmat_filesystem::read_async(path)
                .await
                .map_err(|error| error.to_string()),
        }
    }

    pub const fn checked(&self) -> &runmat_runtime::analysis::CheckedFeaSource {
        &self.checked
    }

    pub fn physical_path(&self) -> &Path {
        &self.physical_path
    }

    pub const fn frozen_project(&self) -> Option<&runmat_package::FrozenProject> {
        self.frozen_project.as_ref()
    }

    pub const fn project_manifest(&self) -> Option<&runmat_config::project::ProjectManifest> {
        self.project_manifest.as_ref()
    }

    pub const fn program_environment(&self) -> &runmat_execution::ProgramEnvironment {
        &self.program_environment
    }

    pub const fn compatibility_mode(&self) -> crate::CompatMode {
        self.compatibility_mode
    }
}

impl std::ops::Deref for CheckedFeaSessionSource {
    type Target = runmat_runtime::analysis::CheckedFeaSource;

    fn deref(&self) -> &Self::Target {
        &self.checked
    }
}

impl RunMatSession {
    /// Check a `.fea` source against this session's exact frozen project and executable compiler.
    ///
    /// Hosts call this boundary instead of rebuilding module, callable, execution-profile, and
    /// geometry discovery independently. All source bytes are frozen before the compiler runs.
    pub async fn check_fea_source_path(
        &mut self,
        requested_path: &Path,
    ) -> Result<FeaSourceCheckResult, String> {
        let source =
            std::sync::Arc::new(self.load_and_check_fea_source_path(requested_path).await?);
        let result = source.check_result();
        self.retain_checked_fea_source(&result, source);
        Ok(result)
    }

    /// Load and check a `.fea` source while retaining its exact native execution context.
    pub async fn load_and_check_fea_source_path(
        &mut self,
        requested_path: &Path,
    ) -> Result<CheckedFeaSessionSource, String> {
        let project = self
            .project_handoff
            .as_ref()
            .map(|handoff| handoff.project.clone());
        let (source_root, physical_path, logical_path) =
            source_identity(requested_path, project.as_ref())?;
        let source = read_source(&physical_path, "FEA source").await?;
        let root = FeaSourceSnapshotBuilder::source_module(logical_path, source);
        let mut snapshot = FeaSourceSnapshotBuilder::new(&source_root, root)?;

        let mut attempted_modules = std::collections::BTreeSet::new();
        loop {
            let missing = snapshot
                .missing_module_paths()?
                .into_iter()
                .filter(|path| attempted_modules.insert(path.clone()))
                .collect::<Vec<_>>();
            if missing.is_empty() {
                break;
            }
            for logical_path in missing {
                let physical_path =
                    runmat_runtime::analysis::physical_source_path(&source_root, &logical_path)
                        .map_err(|error| error.to_string())?;
                if let Ok(source) = runmat_filesystem::read_async(&physical_path).await {
                    snapshot.add_module(FeaSourceSnapshotBuilder::source_module(
                        logical_path,
                        source,
                    ))?;
                }
            }
        }

        let project_manifest = project_manifest(project.as_ref()).await?;
        let profiles = execution_profiles(project_manifest.as_ref())?;
        let mut prepared = snapshot.prepare()?;
        let functions = match project.as_ref() {
            Some(project) => {
                self.compile_fea_function_facts(
                    project,
                    &physical_path,
                    prepared.function_requests()?,
                )
                .await?
            }
            None => Vec::new(),
        };
        let requests = prepared.requests(functions.clone(), profiles.clone())?;
        for request in requests.geometry {
            let physical_path =
                runmat_runtime::analysis::physical_source_path(&source_root, &request.logical_path)
                    .map_err(|error| error.to_string())?;
            if let Ok(bytes) = runmat_filesystem::read_async(&physical_path).await {
                prepared.add_geometry(request.logical_path, bytes)?;
            }
        }
        Ok(CheckedFeaSessionSource {
            checked: prepared.check(functions, profiles)?,
            captured: None,
            physical_path,
            frozen_project: project,
            project_manifest,
            program_environment: crate::program_environment(self.compat_mode()),
            compatibility_mode: self.compat_mode(),
        })
    }

    pub(super) async fn compile_fea_function_facts(
        &mut self,
        project: &runmat_package::FrozenProject,
        requester_path: &Path,
        requests: Vec<FunctionRequest>,
    ) -> Result<Vec<FunctionFacts>, String> {
        self.compile_fea_function_facts_from_view(project, requester_path, requests, None)
            .await
    }

    pub(super) async fn compile_fea_function_facts_from_view(
        &mut self,
        project: &runmat_package::FrozenProject,
        requester_path: &Path,
        requests: Vec<FunctionRequest>,
        captured: Option<&std::collections::BTreeMap<PathBuf, Vec<u8>>>,
    ) -> Result<Vec<FunctionFacts>, String> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let project_revision = project.revision();
        let revision = runmat_execution::ProgramRevision::new(
            runmat_execution::Digest::from_bytes(*project_revision.graph_digest.bytes()),
            runmat_execution::Digest::from_bytes(*project_revision.source_revision.bytes()),
            crate::program_environment(self.compat_mode()),
        )
        .map_err(|error| error.to_string())?;
        let mut facts = Vec::with_capacity(requests.len());
        for request in requests {
            let source_path = runmat_runtime::analysis::physical_source_path(
                &project.workspace_root,
                &request.module_path,
            )
            .map_err(|error| error.to_string())?;
            let definition = runmat_package::resolve_visible_function_definition(
                project,
                requester_path,
                &source_path,
                request.symbol.as_str(),
            )?;
            let source_id = definition.source_id.as_ref().ok_or_else(|| {
                format!(
                    "imported function `{}` has no frozen source identity",
                    definition.qualified_name
                )
            })?;
            let source = match captured {
                Some(captured) => captured.get(&source_path).cloned().ok_or_else(|| {
                    format!(
                        "imported function is absent from the captured candidate: {}",
                        source_path.display()
                    )
                })?,
                None => read_source(&source_path, "imported function").await?,
            };
            let source_digest = runmat_execution::Digest::sha256(&source);
            if source_digest.bytes() != source_id.content_digest.bytes() {
                return Err(format!(
                    "imported function `{}` changed after the project source graph was frozen",
                    definition.qualified_name
                ));
            }
            let source_text = String::from_utf8(source).map_err(|_| {
                format!(
                    "imported function {} is not UTF-8 MATLAB text",
                    source_path.display()
                )
            })?;
            let executable_source = crate::ExecutableSource::frozen(
                source_id.package_instance.to_string(),
                source_id.relative_path.as_str(),
                source_path.to_string_lossy(),
                source_text,
            );
            let compiled = match captured {
                Some(captured) => {
                    self.compile_executable_unit_from_snapshot(
                        executable_source,
                        revision.clone(),
                        &runmat_package::FrozenProjectHandoff::new(project.clone()),
                        captured,
                    )
                    .await
                }
                None => {
                    self.compile_executable_unit(executable_source, Some(revision.clone()))
                        .await
                }
            };
            let interface = compiled
                .map_err(|error| {
                    format!(
                        "cannot compile imported function `{}`: {error}",
                        definition.qualified_name
                    )
                })?
                .primary_function_interface(request.symbol.as_str())?;
            facts.push(FunctionFacts {
                module_path: request.module_path,
                symbol: request.symbol,
                source_digest: runmat_runtime::analysis::SourceDigest(*source_digest.bytes()),
                declaration_span: interface.declaration_span,
                inputs: interface
                    .inputs
                    .into_iter()
                    .map(|parameter| runmat_types::OperationParameter {
                        name: parameter.name,
                        value: runmat_runtime::analysis::operation_pattern_from_value_fact(
                            &parameter.value,
                        ),
                        arity: parameter.arity,
                        default: None,
                    })
                    .collect(),
                outputs: interface
                    .outputs
                    .iter()
                    .map(runmat_runtime::analysis::operation_pattern_from_value_fact)
                    .collect(),
                effects: interface.effects,
                cooperative_suspension: interface.cooperative_suspension,
                capabilities: interface.capabilities,
            });
        }
        Ok(facts)
    }
}

fn source_identity(
    requested_path: &Path,
    project: Option<&runmat_package::FrozenProject>,
) -> Result<(PathBuf, PathBuf, LogicalSourcePath), String> {
    if let Some(project) = project {
        let source_root = project.workspace_root.clone();
        let physical_path = if crate::diagnostic_path::host_path_is_absolute(requested_path)
            || requested_path.starts_with(&source_root)
        {
            requested_path.to_path_buf()
        } else {
            source_root.join(requested_path)
        };
        let logical_path =
            runmat_runtime::analysis::logical_source_path(&source_root, &physical_path)?;
        return Ok((source_root, physical_path, logical_path));
    }
    let file_name = requested_path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| "FEA source path must name a UTF-8 file".to_string())?;
    let source_root = requested_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let logical_path =
        LogicalSourcePath::new(file_name.to_owned()).map_err(|error| error.to_string())?;
    Ok((source_root, requested_path.to_path_buf(), logical_path))
}

async fn project_manifest(
    project: Option<&runmat_package::FrozenProject>,
) -> Result<Option<runmat_config::project::ProjectManifest>, String> {
    let Some(project) = project else {
        return Ok(None);
    };
    runmat_config::project::load_project_manifest_async(&project.manifest_path)
        .await
        .map(Some)
        .map_err(|error| format!("cannot load FEA project manifest: {error}"))
}

pub(super) fn execution_profiles(
    manifest: Option<&runmat_config::project::ProjectManifest>,
) -> Result<Vec<runmat_runtime::analysis::ExecutionProfileFacts>, String> {
    let Some(manifest) = manifest else {
        return Ok(Vec::new());
    };
    runmat_runtime::analysis::remote_execution_profile_facts(
        manifest
            .execution_profiles
            .iter()
            .map(|(name, profile)| (name.clone(), profile.accelerator.is_some())),
    )
    .map_err(|error| error.to_string())
}

async fn read_source(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    runmat_filesystem::read_async(path)
        .await
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))
}
