use super::*;
pub(super) fn prune_filesystem_runs(root: &Path) -> Result<(), String> {
    let retention = current_retention_config();
    let max_runs = retention.max_runs.unwrap_or_else(|| {
        std::env::var("RUNMAT_FEA_ARTIFACT_MAX_RUNS")
            .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS"))
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
    });
    let max_runs_per_kind = retention.max_runs_per_kind.unwrap_or_else(|| {
        std::env::var("RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND")
            .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS_PER_KIND"))
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
    });
    if max_runs == 0 && max_runs_per_kind == 0 {
        return Ok(());
    }

    let runs_dir = root.join("runs");
    ensure_owned_directory(root, "analysis artifact root")?;
    ensure_owned_directory(&runs_dir, "analysis artifact runs directory")?;
    let mut artifacts = Vec::new();
    for entry in fs_read_dir(&runs_dir).map_err(|err| format!("failed to scan artifacts: {err}"))? {
        let path = entry.path().to_path_buf();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        if entry.file_type() != FsFileType::File {
            return Err(format!(
                "analysis run artifact must be a regular file: {}",
                path.display()
            ));
        }
        let run_id = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| {
                format!(
                    "analysis run artifact filename is invalid: {}",
                    path.display()
                )
            })?;
        let bytes = fs_read(&path).map_err(|err| format!("failed to read artifact file: {err}"))?;
        let Ok(persisted) = decode_persisted_artifact(&bytes, run_id) else {
            continue;
        };
        let (identity, embedded_run_id, created_at) = match persisted {
            PersistedAnalysisArtifact::Standard(persisted) => (
                persisted.metadata.run.identity,
                persisted.metadata.run_id,
                parse_recorded_at(&persisted.created_at),
            ),
            PersistedAnalysisArtifact::StructuralFrequency(persisted) => (
                persisted.step.identity,
                persisted.step.run_id,
                parse_recorded_at(&persisted.created_at),
            ),
        };
        let modified_at = fs_modified(&path).ok().flatten().map(DateTime::<Utc>::from);
        artifacts.push(FilesystemRunArtifact {
            path,
            identity,
            run_id: embedded_run_id,
            created_at,
            modified_at,
        });
    }
    sort_filesystem_artifacts_newest_first(&mut artifacts);

    let mut to_remove = Vec::new();
    if max_runs_per_kind > 0 {
        let mut per_kind_counts: HashMap<AnalysisRunIdentity, usize> = HashMap::new();
        for artifact in &artifacts {
            let count = per_kind_counts
                .entry(artifact.identity.clone())
                .or_default();
            *count += 1;
            if *count > max_runs_per_kind {
                to_remove.push(artifact.path.clone());
            }
        }
    }
    if max_runs > 0 {
        for (index, artifact) in artifacts.iter().enumerate() {
            if index >= max_runs {
                to_remove.push(artifact.path.clone());
            }
        }
    }
    to_remove.sort();
    to_remove.dedup();
    for path in to_remove {
        if fs_remove_file(&path).is_ok() {
            if let Some(run_id) = path.file_stem().and_then(|name| name.to_str()) {
                observations::prune_attachment(root, run_id)?;
            }
        }
    }
    Ok(())
}
