//! Candidate checking consumes captured bytes. It never installs a provider,
//! writes a file, realizes geometry, or executes a user function.
use super::RunMatSession;
use runmat_runtime::analysis::{
    FeaCheckStatus, FeaSourceCheckResult, FeaWorkspaceCandidate, FeaWorkspaceEdit,
    FeaWorkspaceFile, FeaWorkspaceSnapshot, SourceDigest,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaCapturedDependency {
    pub source: runmat_package::StableSourceId,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaWorkspaceCapture {
    pub files: Vec<FeaWorkspaceFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaWorkspaceCheckRequest {
    pub snapshot: FeaWorkspaceCapture,
    pub edits: Vec<FeaWorkspaceEdit>,
    pub dependencies: Vec<FeaCapturedDependency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaWorkspaceCheckResult {
    pub candidate_digest: SourceDigest,
    pub environment: runmat_execution::ProgramEnvironment,
    pub project_revision: Option<runmat_package::ProjectRevision>,
    pub roots: Vec<FeaSourceCheckResult>,
    pub publication_valid: bool,
}

impl RunMatSession {
    pub async fn check_fea_workspace(
        &mut self,
        request: FeaWorkspaceCheckRequest,
    ) -> Result<FeaWorkspaceCheckResult, String> {
        self.check_fea_workspace_roots(request, None).await
    }

    pub(super) async fn check_fea_workspace_roots(
        &mut self,
        request: FeaWorkspaceCheckRequest,
        selected_root: Option<&runmat_runtime::analysis::LogicalSourcePath>,
    ) -> Result<FeaWorkspaceCheckResult, String> {
        // The authenticated runtime owns the provider mount. Frontends supply
        // logical project paths, never native paths or another project's mount.
        let source_root = self.project_handoff.as_ref().map_or_else(
            || {
                runmat_filesystem::current_provider()
                    .current_dir_override()
                    .unwrap_or_else(|| PathBuf::from("/"))
            },
            |handoff| handoff.project.workspace_root.clone(),
        );
        let candidate = FeaWorkspaceCandidate::new(
            FeaWorkspaceSnapshot {
                source_root,
                files: request.snapshot.files,
            },
            request.edits,
        )
        .map_err(|error| error.to_string())?;
        // Configuration changes require a newly admitted graph, rather than
        // silently checking under the old package/security configuration.
        if candidate.changed_paths().any(|path| {
            !matches!(
                std::path::Path::new(path.as_str())
                    .extension()
                    .and_then(|extension| extension.to_str()),
                Some("fea" | "m")
            )
        }) {
            return Err("FEA source changes may only mutate admitted .fea/.m sources".into());
        }
        let mut captured = BTreeMap::new();
        for (logical, bytes) in candidate.files() {
            let physical =
                runmat_runtime::analysis::physical_source_path(candidate.source_root(), logical)
                    .map_err(|error| error.to_string())?;
            captured.insert(physical, bytes.to_vec());
        }
        let mut manifest = None;
        let mut project = self
            .project_handoff
            .as_ref()
            .map(|handoff| handoff.project.clone());
        if let Some(frozen) = project.as_ref() {
            if candidate.source_root() != frozen.workspace_root {
                return Err("candidate belongs to a different project".into());
            }
            let bytes = captured
                .get(&frozen.manifest_path)
                .ok_or("captured project manifest is unavailable")?;
            let text =
                std::str::from_utf8(bytes).map_err(|_| "captured project manifest is not UTF-8")?;
            let parsed = if frozen
                .manifest_path
                .extension()
                .and_then(|ext| ext.to_str())
                == Some("json")
            {
                runmat_config::project::parse_project_manifest_json(text)
                    .map_err(|error| error.to_string())?
            } else {
                runmat_config::project::parse_project_manifest_toml(text)
                    .map_err(|error| error.to_string())?
            };
            frozen.validate_root_manifest(&parsed)?;
            let mut root_sources = Vec::new();
            for (path, bytes) in &captured {
                for source_root in &parsed.sources.roots {
                    if let Some(descriptor) = runmat_config::project::project_source_file_from_path(
                        path,
                        &frozen.workspace_root.join(source_root),
                        source_root,
                    ) {
                        root_sources.push((descriptor, bytes.clone()));
                        break;
                    }
                }
            }
            project = Some(frozen.with_root_sources(root_sources)?);
            manifest = Some(parsed);
        }
        for dependency in request.dependencies {
            let Some(project) = project.as_ref() else {
                return Err("package sources require an admitted project".into());
            };
            if dependency.source.package_instance == project.graph.root {
                return Err("root package source cannot be supplied as a dependency".into());
            }
            let path = project
                .access_paths
                .get(&dependency.source)
                .ok_or("dependency source is outside the admitted graph")?;
            if runmat_package::ContentDigest::sha256(&dependency.bytes)
                != dependency.source.content_digest
            {
                return Err("captured dependency digest differs from the admitted source".into());
            }
            if captured.insert(path.clone(), dependency.bytes).is_some() {
                return Err("duplicate captured dependency path".into());
            }
        }
        let profiles = super::fea_source::execution_profiles(manifest.as_ref())?;
        let captured = std::sync::Arc::new(captured);
        let mut roots = Vec::new();
        let changed = candidate
            .changed_paths()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        // Discover consumers from the complete captured inventory, including
        // unopened modules. Unrelated manually broken drafts do not block edits.
        for (root, _) in candidate.files().filter(|(path, _)| {
            std::path::Path::new(path.as_str())
                .extension()
                .and_then(|extension| extension.to_str())
                == Some("fea")
        }) {
            if selected_root.is_some_and(|selected| selected != root) {
                continue;
            }
            let mut prepared = candidate.prepare(root).map_err(|error| error.to_string())?;
            if !changed.is_empty() && !changed.contains(root) {
                match prepared.source_dependencies() {
                    Ok(dependencies) if dependencies.is_disjoint(&changed) => continue,
                    // An unresolvable closure cannot safely be declared unrelated.
                    _ => {}
                }
            }
            let physical: PathBuf =
                runmat_runtime::analysis::physical_source_path(candidate.source_root(), root)
                    .map_err(|error| error.to_string())?;
            let functions = match project.as_ref() {
                Some(project) => {
                    self.compile_fea_function_facts_from_view(
                        project,
                        &physical,
                        prepared.function_requests()?,
                        Some(&captured),
                    )
                    .await?
                }
                None => Vec::new(),
            };
            for geometry in prepared
                .requests(functions.clone(), profiles.clone())?
                .geometry
            {
                if let Some(bytes) = candidate.read(&geometry.logical_path) {
                    prepared.add_geometry(geometry.logical_path, bytes.to_vec())?;
                }
            }
            let source = std::sync::Arc::new(super::CheckedFeaSessionSource {
                checked: prepared.check(functions, profiles.clone())?,
                physical_path: physical,
                frozen_project: project.clone(),
                project_manifest: manifest.clone(),
                program_environment: crate::program_environment(self.compat_mode()),
                compatibility_mode: self.compat_mode(),
                captured: Some(captured.clone()),
            });
            let result = source.check_result();
            self.retain_checked_fea_source(&result, source);
            roots.push(result);
        }
        let publication_valid = !roots.is_empty()
            && roots
                .iter()
                .all(|root| root.status == FeaCheckStatus::Valid);
        Ok(FeaWorkspaceCheckResult {
            candidate_digest: candidate.digest(),
            environment: crate::program_environment(self.compat_mode()),
            project_revision: project.map(|project| project.revision()),
            roots,
            publication_valid,
        })
    }
}
