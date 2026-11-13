use std::path::{Path, PathBuf};

use anyhow::Result;
use runmat_config::project::ProjectManifest;
use runmat_config::runtime::RunMatRuntimeConfig;

use crate::cli::Cli;

pub(crate) struct NativeCheckedFeaSource {
    pub session_source: runmat_core::CheckedFeaSessionSource,
    pub frozen_project: Option<runmat_package::FrozenProject>,
    pub project_manifest: Option<ProjectManifest>,
}

pub(crate) fn is_fea_source_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("fea"))
}

impl std::ops::Deref for NativeCheckedFeaSource {
    type Target = runmat_runtime::analysis::CheckedFeaSource;

    fn deref(&self) -> &Self::Target {
        self.session_source.checked()
    }
}

pub(crate) async fn check_source(
    path: &Path,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<NativeCheckedFeaSource> {
    check_source_with_settings(path, cli, config, false).await
}

pub(super) async fn check_source_for_cluster(
    path: &Path,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<NativeCheckedFeaSource> {
    check_source_with_settings(path, cli, config, true).await
}

async fn check_source_with_settings(
    path: &Path,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
    cluster: bool,
) -> Result<NativeCheckedFeaSource> {
    let project = crate::commands::package::resolve_for_source(path, cli).await?;
    let source_root = project
        .as_ref()
        .map(|project| project.resolved.frozen.workspace_root.clone())
        .unwrap_or_else(|| source_parent(path));
    let physical_path = absolute_entry_path(&source_root, path);
    let mut session = crate::commands::session::create_session(
        false,
        false,
        config,
        "create FEA source-check session",
    )?;
    if cluster {
        session.apply_cluster_runtime_config(config)?;
    }
    if let Some(project) = project.as_ref() {
        session
            .install_project_handoff(runmat_package::FrozenProjectHandoff::new(
                project.resolved.frozen.clone(),
            ))
            .await?;
    }
    let session_source = session
        .load_and_check_fea_source_path(&physical_path)
        .await
        .map_err(anyhow::Error::msg)?;
    Ok(NativeCheckedFeaSource {
        frozen_project: session_source.frozen_project().cloned(),
        project_manifest: session_source.project_manifest().cloned(),
        session_source,
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
        let facts = runmat_analysis_language::remote_execution_profile_facts(
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
