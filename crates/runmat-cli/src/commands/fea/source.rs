use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use runmat_analysis_language::{
    bind_program, check_program, resolve_logical_path, CheckedFeaProgram, ExecutionPlacement,
    ExecutionProfileFacts, ExecutionProfileIdentity, FeaCompiler, LogicalSourcePath,
    ModuleGraphRequest, SourceSymbol,
};
use runmat_config::project::{load_project_manifest_async, ProjectManifest};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_parser::fea::{FeaImportKind, FeaModuleItem};
use runmat_runtime::analysis::{FilesystemFeaHost, ProjectFeaHost};
use runmat_types::{CapabilityRequirement, CapabilitySet};

use crate::cli::Cli;

pub(crate) struct CheckedFeaSource {
    pub program: CheckedFeaProgram,
    pub root_module: LogicalSourcePath,
    pub source_root: PathBuf,
    pub frozen_project: Option<runmat_package::FrozenProject>,
    pub project_manifest: Option<ProjectManifest>,
}

pub(crate) async fn check_source(
    path: &Path,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<CheckedFeaSource> {
    let project = crate::commands::package::resolve_for_source(path, cli).await?;
    let source_root = project
        .as_ref()
        .map(|project| project.resolved.frozen.workspace_root.clone())
        .unwrap_or_else(|| source_parent(path));
    let physical_path = absolute_entry_path(&source_root, path);
    let files = FilesystemFeaHost::new(&source_root).map_err(anyhow::Error::msg)?;
    let root = files
        .load_physical_module(&physical_path)
        .map_err(anyhow::Error::msg)?;
    let root_module = root.logical_path.clone();
    let graph = FeaCompiler::new(&files).build_module_graph(ModuleGraphRequest {
        source_root: LogicalSourcePath::new(files.logical_root().to_string())?,
        root,
    });
    let requests = graph
        .modules
        .iter()
        .flat_map(|(module_path, module)| {
            module
                .syntax
                .iter()
                .flat_map(|syntax| syntax.items.iter())
                .filter_map(move |item| match item {
                    FeaModuleItem::Import(import) if import.kind == FeaImportKind::Function => {
                        Some((module_path, import))
                    }
                    _ => None,
                })
        })
        .map(|(module_path, import)| {
            Ok((
                resolve_logical_path(&graph.source_root, module_path, &import.source)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?,
                SourceSymbol::new(import.name.text.clone())?,
            ))
        })
        .collect::<Result<BTreeSet<_>>>()?
        .into_iter()
        .collect::<Vec<_>>();
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
        .map(execution_profile_facts)
        .transpose()?
        .unwrap_or_default();
    let host =
        ProjectFeaHost::new(&source_root, functions, profiles).map_err(anyhow::Error::msg)?;
    let bound = bind_program(&host, graph);
    let program = check_program(&host, bound);
    Ok(CheckedFeaSource {
        program,
        root_module,
        source_root,
        frozen_project: project.map(|project| project.resolved.frozen),
        project_manifest: manifest,
    })
}

fn execution_profile_facts(manifest: &ProjectManifest) -> Result<Vec<ExecutionProfileFacts>> {
    manifest
        .execution_profiles
        .iter()
        .map(|(name, profile)| {
            let mut capabilities = std::collections::BTreeSet::from([
                CapabilityRequirement::DistributedRuntime,
                CapabilityRequirement::ParallelRuntime,
            ]);
            if profile.accelerator.is_some() {
                capabilities.insert(CapabilityRequirement::Accelerator);
            }
            Ok(ExecutionProfileFacts {
                profile: ExecutionProfileIdentity::new(name.clone())?,
                placement: ExecutionPlacement::Remote,
                capabilities: CapabilitySet(capabilities),
            })
        })
        .collect()
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
        let facts = execution_profile_facts(&manifest).unwrap();
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
