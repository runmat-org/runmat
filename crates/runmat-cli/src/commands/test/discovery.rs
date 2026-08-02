use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use runmat_config::project::{
    discover_project_manifest_from, load_project_manifest, ProjectTestConfig,
};
use runmat_package::FrozenProjectHandoff;
use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
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
    let sources = collect_sources(&project_root, &catalog_roots)?;
    if sources.is_empty() {
        bail!("no MATLAB source files were found in the selected test inputs");
    }
    let base_source_digest = source_catalog_digest(&sources);
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
        .unwrap_or_else(|| {
            (
                if manifest_bytes.is_empty() {
                    "sha256:loose-project".into()
                } else {
                    format!("sha256:{:x}", Sha256::digest(&manifest_bytes))
                },
                base_source_digest,
            )
        });
    let test_config_digest = format!(
        "sha256:{:x}",
        Sha256::digest(serde_json::to_vec(&test_config).context("failed to encode test config")?)
    );
    let snapshot = FrozenTestRunSnapshot::freeze(
        graph_digest,
        base_source_digest,
        1,
        1,
        test_config_digest,
        sources,
        Vec::new(),
    )
    .context("failed to freeze test source snapshot")?;
    let source_prefixes = targets
        .iter()
        .filter_map(|target| snapshot_prefix(&project_root, target))
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

fn collect_sources(project_root: &Path, roots: &[PathBuf]) -> Result<Vec<SavedRunSource>> {
    let mut files = BTreeSet::new();
    for root in roots {
        collect_matlab_files(root, &mut files)?;
    }
    files
        .into_iter()
        .map(|path| {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read test source {}", path.display()))?;
            let (owner_identity, relative_path) = source_identity(project_root, &path);
            Ok(SavedRunSource {
                owner_identity,
                relative_path,
                content,
            })
        })
        .collect()
}

fn collect_matlab_files(path: &Path, files: &mut BTreeSet<PathBuf>) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Ok(());
    }
    if metadata.is_file() {
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("m"))
        {
            files.insert(std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()));
        }
        return Ok(());
    }
    let mut entries = std::fs::read_dir(path)
        .with_context(|| format!("failed to read test directory {}", path.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        collect_matlab_files(&entry.path(), files)?;
    }
    Ok(())
}

fn source_identity(project_root: &Path, path: &Path) -> (String, String) {
    if let Ok(relative) = path.strip_prefix(project_root) {
        return (
            "path:workspace".into(),
            relative.to_string_lossy().replace('\\', "/"),
        );
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    (
        format!(
            "path:external:{:x}",
            Sha256::digest(parent.to_string_lossy().as_bytes())
        ),
        path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
    )
}

fn snapshot_prefix(project_root: &Path, target: &Path) -> Option<String> {
    let relative = target.strip_prefix(project_root).ok()?;
    let mut prefix = relative.to_string_lossy().replace('\\', "/");
    if target.is_dir() && !prefix.is_empty() {
        prefix.push('/');
    }
    Some(prefix)
}

fn source_catalog_digest(sources: &[SavedRunSource]) -> String {
    let mut hasher = Sha256::new();
    for source in sources {
        for field in [
            source.owner_identity.as_bytes(),
            source.relative_path.as_bytes(),
            source.content.as_bytes(),
        ] {
            hasher.update((field.len() as u64).to_be_bytes());
            hasher.update(field);
        }
    }
    format!("sha256:{:x}", hasher.finalize())
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
