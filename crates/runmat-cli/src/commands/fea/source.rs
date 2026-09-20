use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use runmat_analysis_language::remote_execution_profile_facts;
use runmat_config::project::{load_project_manifest_async, ProjectManifest};
use runmat_config::runtime::RunMatRuntimeConfig;

use crate::cli::Cli;

pub(crate) struct NativeCheckedFeaSource {
    pub checked: runmat_runtime::analysis::CheckedFeaSource,
    pub frozen_project: Option<runmat_package::FrozenProject>,
    pub project_manifest: Option<ProjectManifest>,
}

impl std::ops::Deref for NativeCheckedFeaSource {
    type Target = runmat_runtime::analysis::CheckedFeaSource;

    fn deref(&self) -> &Self::Target {
        &self.checked
    }
}

pub(crate) async fn check_source(
    path: &Path,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<NativeCheckedFeaSource> {
    let project = crate::commands::package::resolve_for_source(path, cli).await?;
    let source_root = project
        .as_ref()
        .map(|project| project.resolved.frozen.workspace_root.clone())
        .unwrap_or_else(|| source_parent(path));
    let physical_path = absolute_entry_path(&source_root, path);
    let prepared = runmat_runtime::analysis::prepare_fea_path_in_root(&source_root, &physical_path)
        .map_err(anyhow::Error::msg)?;
    let requests = prepared.function_requests().map_err(anyhow::Error::msg)?;
    let functions = if requests.is_empty() {
        Vec::new()
    } else {
        let frozen = &project
            .as_ref()
            .context("imported `.m` functions require an enclosing frozen RunMat project")?
            .resolved
            .frozen;
        super::callables::compile_function_facts(
            &physical_path,
            &source_root,
            &requests,
            frozen,
            config,
        )
        .await?
    };
    let manifest = match project.as_ref() {
        Some(project) => Some(
            load_project_manifest_async(&project.resolved.frozen.manifest_path)
                .await
                .context("failed to load frozen-project execution profiles")?,
        ),
        None => None,
    };
    let profiles = manifest
        .as_ref()
        .map(|manifest| {
            remote_execution_profile_facts(
                manifest
                    .execution_profiles
                    .iter()
                    .map(|(name, profile)| (name.clone(), profile.accelerator.is_some())),
            )
            .map_err(anyhow::Error::new)
        })
        .transpose()?
        .unwrap_or_default();
    let checked = prepared
        .check(functions, profiles)
        .map_err(anyhow::Error::msg)?;
    Ok(NativeCheckedFeaSource {
        checked,
        frozen_project: project.map(|project| project.resolved.frozen),
        project_manifest: manifest,
    })
}

fn source_parent(path: &Path) -> PathBuf {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn absolute_entry_path(source_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() || path.starts_with(source_root) {
        path.to_path_buf()
    } else {
        source_root.join(path)
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_language::ExecutionPlacement;
    use runmat_config::project::parse_project_manifest_toml;
    use runmat_types::CapabilityRequirement;

    use super::*;

    #[test]
    fn project_profiles_become_exact_compiler_facts() {
        let manifest = parse_project_manifest_toml(
            r#"
[package]
name = "analysis"
[sources]
roots = ["src"]
[execution-profiles.cpu]
cluster = "cpu-cluster"
[execution-profiles.gpu]
cluster = "gpu-cluster"
[execution-profiles.gpu.accelerator]
class = "metal"
memory-bytes = 17179869184
"#,
        )
        .unwrap();
        let facts = remote_execution_profile_facts(
            manifest
                .execution_profiles
                .iter()
                .map(|(name, profile)| (name.clone(), profile.accelerator.is_some())),
        )
        .unwrap();
        assert_eq!(facts.len(), 2);
        let cpu = facts
            .iter()
            .find(|facts| facts.profile.as_str() == "cpu")
            .unwrap();
        let gpu = facts
            .iter()
            .find(|facts| facts.profile.as_str() == "gpu")
            .unwrap();
        assert_eq!(cpu.placement, ExecutionPlacement::Remote);
        assert!(cpu
            .capabilities
            .0
            .contains(&CapabilityRequirement::DistributedRuntime));
        assert!(!cpu
            .capabilities
            .0
            .contains(&CapabilityRequirement::Accelerator));
        assert!(gpu
            .capabilities
            .0
            .contains(&CapabilityRequirement::Accelerator));
    }
}
