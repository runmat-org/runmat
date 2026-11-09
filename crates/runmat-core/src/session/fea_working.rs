//! Read-only working-source capture. Captured bytes are checked through the
//! candidate compiler; editor overlays do not write or rebind a provider.
use super::{FeaCapturedDependency, FeaWorkspaceCapture, FeaWorkspaceCheckRequest, RunMatSession};
use runmat_runtime::analysis::{FeaSourceCheckResult, FeaWorkspaceFile, LogicalSourcePath};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

use runmat_fea_source::{
    MAX_FEA_GEOMETRY_BYTES as MAX_GEOMETRY_BYTES, MAX_FEA_SOURCE_BYTES as MAX_SOURCE_BYTES,
    MAX_FEA_WORKSPACE_FILES as MAX_FILES,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaWorkingSource {
    pub path: LogicalSourcePath,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaWorkingCheckRequest {
    pub root: LogicalSourcePath,
    pub overlays: Vec<FeaWorkingSource>,
}

impl RunMatSession {
    pub async fn check_fea_working_source(
        &mut self,
        request: FeaWorkingCheckRequest,
    ) -> Result<FeaSourceCheckResult, String> {
        let provider = runmat_filesystem::current_provider();
        let root = self.project_handoff.as_ref().map_or_else(
            || {
                provider
                    .current_dir_override()
                    .unwrap_or_else(|| PathBuf::from("/"))
            },
            |handoff| handoff.project.workspace_root.clone(),
        );
        let mut directories = vec![root.clone()];
        let mut files = BTreeMap::new();
        let mut budget = 0usize;
        let mut visited = 0usize;
        while let Some(directory) = directories.pop() {
            for entry in provider
                .read_dir(&directory)
                .await
                .map_err(|error| error.to_string())?
            {
                visited += 1;
                if visited > MAX_FILES {
                    return Err("working-source inventory exceeds 10000 entries".into());
                }
                let path = entry.path();
                let relative = path
                    .strip_prefix(&root)
                    .map_err(|_| "provider returned an entry outside the project")?;
                if matches!(
                    entry.file_name().to_str(),
                    Some(".artifacts" | ".git" | "node_modules" | "target")
                ) {
                    continue;
                }
                if entry.is_dir() {
                    directories.push(path.to_path_buf());
                    continue;
                }
                if !matches!(
                    path.extension().and_then(|value| value.to_str()),
                    Some("fea" | "m")
                ) && self
                    .project_handoff
                    .as_ref()
                    .is_none_or(|handoff| path != handoff.project.manifest_path)
                {
                    continue;
                }
                let metadata = provider
                    .metadata(path)
                    .await
                    .map_err(|error| error.to_string())?;
                if metadata.len() > (MAX_SOURCE_BYTES - budget) as u64 {
                    return Err("working-source capture exceeds 32 MiB".into());
                }
                let bytes = provider
                    .read(path)
                    .await
                    .map_err(|error| error.to_string())?;
                budget = budget
                    .checked_add(bytes.len())
                    .ok_or("working-source size overflow")?;
                if budget > MAX_SOURCE_BYTES {
                    return Err("working-source capture exceeds 32 MiB".into());
                }
                let logical = LogicalSourcePath::new(relative.to_string_lossy().into_owned())
                    .map_err(|error| error.to_string())?;
                files.insert(logical.clone(), captured(logical, bytes));
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for overlay in request.overlays {
            runmat_runtime::analysis::physical_source_path(&root, &overlay.path)
                .map_err(|error| error.to_string())?;
            if !matches!(
                std::path::Path::new(overlay.path.as_str())
                    .extension()
                    .and_then(|value| value.to_str()),
                Some("fea" | "m")
            ) {
                return Err("working editor overlays must be FEA or MATLAB sources".into());
            }
            if !seen.insert(overlay.path.clone()) {
                return Err("duplicate working editor overlay".into());
            }
            budget = budget
                .checked_add(overlay.bytes.len())
                .ok_or("working-source size overflow")?;
            if budget > MAX_SOURCE_BYTES {
                return Err("working-source capture exceeds 32 MiB".into());
            }
            files.insert(overlay.path.clone(), captured(overlay.path, overlay.bytes));
        }
        // Only referenced exchange geometry is captured; a large CAD directory
        // does not get copied just because a question or source view is opened.
        let candidate = runmat_runtime::analysis::FeaWorkspaceCandidate::new(
            runmat_runtime::analysis::FeaWorkspaceSnapshot {
                source_root: root.clone(),
                files: files.values().cloned().collect(),
            },
            vec![],
        )
        .map_err(|error| error.to_string())?;
        let mut geometry_paths = std::collections::BTreeSet::new();
        {
            let prepared = candidate
                .prepare(&request.root)
                .map_err(|error| error.to_string())?;
            for geometry in prepared.requests(Vec::new(), Vec::new())?.geometry {
                geometry_paths.insert(geometry.logical_path);
            }
        }
        let mut geometry_bytes = 0usize;
        for path in geometry_paths {
            let physical = runmat_runtime::analysis::physical_source_path(&root, &path)
                .map_err(|error| error.to_string())?;
            match provider.metadata(&physical).await {
                Ok(metadata) if metadata.len() > (MAX_GEOMETRY_BYTES - geometry_bytes) as u64 => {
                    return Err("referenced geometry exceeds the 64 MiB source-check budget".into())
                }
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.to_string()),
            }
            match provider.read(&physical).await {
                Ok(bytes) => {
                    geometry_bytes = geometry_bytes
                        .checked_add(bytes.len())
                        .ok_or("geometry capture size overflow")?;
                    if geometry_bytes > MAX_GEOMETRY_BYTES {
                        return Err(
                            "referenced geometry exceeds the 64 MiB source-check budget".into()
                        );
                    }
                    files.insert(path.clone(), captured(path, bytes));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.to_string()),
            }
        }
        let mut dependencies = Vec::new();
        if let Some(handoff) = &self.project_handoff {
            for (source, path) in &handoff.project.access_paths {
                if source.package_instance == handoff.project.graph.root {
                    continue;
                }
                let metadata = provider
                    .metadata(path)
                    .await
                    .map_err(|error| error.to_string())?;
                if metadata.len() > (MAX_SOURCE_BYTES - budget) as u64 {
                    return Err("source dependencies exceed the 32 MiB capture budget".into());
                }
                let bytes = provider
                    .read(path)
                    .await
                    .map_err(|error| error.to_string())?;
                budget = budget
                    .checked_add(bytes.len())
                    .ok_or("dependency capture size overflow")?;
                if budget > MAX_SOURCE_BYTES {
                    return Err("source dependencies exceed the 32 MiB capture budget".into());
                }
                dependencies.push(FeaCapturedDependency {
                    source: source.clone(),
                    bytes,
                });
            }
        }
        let result = self
            .check_fea_workspace_roots(
                FeaWorkspaceCheckRequest {
                    snapshot: FeaWorkspaceCapture {
                        files: files.into_values().collect(),
                    },
                    edits: vec![],
                    dependencies,
                },
                Some(&request.root),
            )
            .await?;
        result
            .roots
            .into_iter()
            .find(|result| result.root_module == request.root)
            .ok_or_else(|| "working FEA root is outside the captured source inventory".into())
    }
}

fn captured(path: LogicalSourcePath, bytes: Vec<u8>) -> FeaWorkspaceFile {
    let digest = runmat_package::ContentDigest::sha256(&bytes);
    FeaWorkspaceFile {
        path,
        version: digest.to_string(),
        bytes,
    }
}
