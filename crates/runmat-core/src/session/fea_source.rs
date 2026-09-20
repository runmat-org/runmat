use std::path::{Path, PathBuf};

use runmat_runtime::analysis::{
    FeaSourceCheckResult, FeaSourceSnapshotBuilder, FunctionFacts, FunctionRequest,
    LogicalSourcePath,
};

use super::RunMatSession;

impl RunMatSession {
    /// Check a `.fea` source against this session's exact frozen project and executable compiler.
    ///
    /// Hosts call this boundary instead of rebuilding module, callable, execution-profile, and
    /// geometry discovery independently. All source bytes are frozen before the compiler runs.
    pub async fn check_fea_source_path(
        &mut self,
        requested_path: &Path,
    ) -> Result<FeaSourceCheckResult, String> {
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

        let profiles = execution_profiles(project.as_ref()).await?;
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
        Ok(prepared.check(functions, profiles)?.check_result())
    }

    async fn compile_fea_function_facts(
        &mut self,
        project: &runmat_package::FrozenProject,
        requester_path: &Path,
        requests: Vec<FunctionRequest>,
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
            let source = read_source(&source_path, "imported function").await?;
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
            let interface = self
                .compile_executable_unit(
                    crate::ExecutableSource::frozen(
                        source_id.package_instance.to_string(),
                        source_id.relative_path.as_str(),
                        source_path.to_string_lossy(),
                        source_text,
                    ),
                    Some(revision.clone()),
                )
                .await
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
        let physical_path =
            if requested_path.is_absolute() || requested_path.starts_with(&source_root) {
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
    let logical_path = LogicalSourcePath::new(format!("project/{file_name}"))
        .map_err(|error| error.to_string())?;
    Ok((source_root, requested_path.to_path_buf(), logical_path))
}

async fn execution_profiles(
    project: Option<&runmat_package::FrozenProject>,
) -> Result<Vec<runmat_runtime::analysis::ExecutionProfileFacts>, String> {
    let Some(project) = project else {
        return Ok(Vec::new());
    };
    let manifest = runmat_config::project::load_project_manifest_async(&project.manifest_path)
        .await
        .map_err(|error| format!("cannot load FEA execution profiles: {error}"))?;
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
