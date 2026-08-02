use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use runmat_config::project::{
    discover_project_manifest_from, load_project_manifest, ProjectTestConfig,
};
use runmat_package::FrozenProjectHandoff;
use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::FrozenTestRunSnapshot;
use runmat_test_runner_native::snapshot::{
    freeze_native_snapshot, source_prefix, NativeSnapshotInput,
};
use sha2::{Digest, Sha256};

use crate::cli::{Cli, TestArgs};
use crate::commands::package::{resolve_for_test_manifest, NativeResolvedProject};

pub(super) struct PreparedDiscovery {
    pub snapshot: FrozenTestRunSnapshot,
    pub selector: TestSelector,
    pub test_config: ProjectTestConfig,
    pub project_root: PathBuf,
    pub project_handoff: Option<FrozenProjectHandoff>,
    _resolved_project: Option<NativeResolvedProject>,
}

pub(super) async fn prepare(args: &TestArgs, cli: &Cli) -> Result<PreparedDiscovery> {
    let current = std::env::current_dir().context("failed to resolve current directory")?;
    let manifest_path = cli
        .config
        .as_deref()
        .filter(|path| declares_project(path))
        .map(Path::to_path_buf)
        .or_else(|| discover_project_manifest_from(&current));
    let (project_root, manifest, manifest_bytes, canonical_manifest) =
        if let Some(path) = manifest_path {
            let canonical = std::fs::canonicalize(&path)
                .with_context(|| format!("failed to locate project manifest {}", path.display()))?;
            let root = canonical
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            let bytes = std::fs::read(&canonical)
                .with_context(|| format!("failed to read {}", canonical.display()))?;
            let manifest = load_project_manifest(&canonical)
                .with_context(|| format!("invalid project manifest {}", canonical.display()))?;
            (root, Some(manifest), bytes, Some(canonical))
        } else {
            (current, None, Vec::new(), None)
        };
    let test_config = manifest
        .as_ref()
        .map(|manifest| manifest.test.clone())
        .unwrap_or_default();
    let targets = execution_targets(args, &test_config, &project_root)?;
    let mut catalog_roots = targets.clone();
    if let Some(manifest) = &manifest {
        catalog_roots.extend(
            manifest
                .sources
                .roots
                .iter()
                .map(|path| project_root.join(path)),
        );
    }
    catalog_roots.extend(
        test_config
            .paths
            .iter()
            .chain(args.paths.iter())
            .map(|path| resolve_from(&project_root, path)),
    );
    deduplicate_paths(&mut catalog_roots);
    let resolved_project = if let Some(manifest) = &canonical_manifest {
        Some(
            resolve_for_test_manifest(manifest.clone(), cli)
                .await
                .context("failed to resolve runtime and test package dependencies")?,
        )
    } else {
        None
    };
    let project_handoff = resolved_project
        .as_ref()
        .map(|project| FrozenProjectHandoff::new(project.resolved.frozen.clone()));
    let (graph_digest, base_source_digest) = project_handoff
        .as_ref()
        .map(|handoff| {
            let revision = handoff.revision();
            (
                revision.graph_digest.to_string(),
                revision.source_revision.to_string(),
            )
        })
        .map(|(graph_digest, base_source_digest)| (graph_digest, Some(base_source_digest)))
        .unwrap_or_else(|| {
            (
                if manifest_bytes.is_empty() {
                    "sha256:loose-project".into()
                } else {
                    format!("sha256:{:x}", Sha256::digest(&manifest_bytes))
                },
                None,
            )
        });
    let test_config_digest = format!(
        "sha256:{:x}",
        Sha256::digest(serde_json::to_vec(&test_config).context("failed to encode test config")?)
    );
    let snapshot = freeze_native_snapshot(NativeSnapshotInput {
        project_root: project_root.clone(),
        catalog_roots,
        graph_digest,
        base_source_digest,
        test_config_digest,
        unsaved_buffers: Vec::new(),
    })
    .context("failed to freeze test source snapshot")?;
    let source_prefixes = targets
        .iter()
        .filter_map(|target| source_prefix(&project_root, target))
        .collect();
    Ok(PreparedDiscovery {
        snapshot,
        selector: TestSelector {
            names: args.names.clone(),
            tags: args.tags.clone(),
            source_prefixes,
            excluded_tags: args.excluded_tags.clone(),
        },
        test_config,
        project_root,
        project_handoff,
        _resolved_project: resolved_project,
    })
}

fn execution_targets(
    args: &TestArgs,
    config: &ProjectTestConfig,
    project_root: &Path,
) -> Result<Vec<PathBuf>> {
    let mut targets = if !args.targets.is_empty() {
        args.targets
            .iter()
            .map(|path| resolve_from(project_root, path))
            .collect()
    } else if !config.roots.is_empty() {
        config
            .roots
            .iter()
            .map(|path| project_root.join(path))
            .collect()
    } else if project_root.join("tests").is_dir() {
        vec![project_root.join("tests")]
    } else {
        vec![project_root.to_path_buf()]
    };
    deduplicate_paths(&mut targets);
    for target in &targets {
        if !target.exists() {
            bail!("test input does not exist: {}", target.display());
        }
    }
    Ok(targets)
}

fn resolve_from(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn deduplicate_paths(paths: &mut Vec<PathBuf>) {
    for path in paths.iter_mut() {
        *path = std::fs::canonicalize(&*path).unwrap_or_else(|_| path.clone());
    }
    paths.sort();
    paths.dedup();
}

fn declares_project(path: &Path) -> bool {
    std::fs::read_to_string(path).ok().is_some_and(|content| {
        content.contains("[package]")
            || content.contains("\"package\"")
            || content.contains("[sources]")
    })
}
