use super::{catalog::compute_source_revision, FrozenProject, ProjectRevision};
use crate::GraphError;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const FROZEN_PROJECT_HANDOFF_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenProjectHandoff {
    pub schema_version: u32,
    pub project: FrozenProject,
}

impl FrozenProjectHandoff {
    pub fn new(project: FrozenProject) -> Self {
        Self {
            schema_version: FROZEN_PROJECT_HANDOFF_SCHEMA_VERSION,
            project,
        }
    }

    pub fn validate(&self) -> Result<(), FrozenProjectHandoffError> {
        if self.schema_version != FROZEN_PROJECT_HANDOFF_SCHEMA_VERSION {
            return Err(FrozenProjectHandoffError::UnsupportedSchema {
                found: self.schema_version,
                supported: FROZEN_PROJECT_HANDOFF_SCHEMA_VERSION,
            });
        }
        self.project
            .graph
            .validate_digest()
            .map_err(FrozenProjectHandoffError::Graph)?;
        validate_sources(&self.project)?;
        Ok(())
    }

    pub fn revision(&self) -> ProjectRevision {
        self.project.revision()
    }

    pub fn into_project(self) -> FrozenProject {
        self.project
    }
}

#[derive(Debug, Error)]
pub enum FrozenProjectHandoffError {
    #[error("unsupported frozen-project handoff schema {found}; supported schema is {supported}")]
    UnsupportedSchema { found: u32, supported: u32 },
    #[error("invalid frozen-project graph: {0}")]
    Graph(GraphError),
    #[error("invalid frozen-project source catalog: {0}")]
    SourceCatalog(String),
    #[error("failed to encode frozen-project source revision input: {0}")]
    Revision(String),
}

fn validate_sources(project: &FrozenProject) -> Result<(), FrozenProjectHandoffError> {
    if project
        .sources
        .packages
        .keys()
        .ne(project.graph.packages.keys())
    {
        return Err(FrozenProjectHandoffError::SourceCatalog(
            "source catalog packages do not match the resolved graph".to_string(),
        ));
    }
    for (instance, package) in &project.sources.packages {
        if instance != &package.package_instance || instance != &package.mount.package_instance {
            return Err(FrozenProjectHandoffError::SourceCatalog(format!(
                "source catalog key {instance} does not match package or mount identity"
            )));
        }
        let graph_package = project.graph.packages.get(instance).ok_or_else(|| {
            FrozenProjectHandoffError::SourceCatalog(format!(
                "source catalog package {instance} is absent from the graph"
            ))
        })?;
        if package.local_name != graph_package.local_name
            || package.mount.source != graph_package.instance.source
        {
            return Err(FrozenProjectHandoffError::SourceCatalog(format!(
                "source catalog metadata for package {instance} does not match the graph"
            )));
        }
        for source in &package.sources {
            if &source.id.package_instance != instance {
                return Err(FrozenProjectHandoffError::SourceCatalog(format!(
                    "source {} belongs to a different package instance",
                    source.id.relative_path
                )));
            }
            if !project.access_paths.contains_key(&source.id) {
                return Err(FrozenProjectHandoffError::SourceCatalog(format!(
                    "source {} has no access path",
                    source.id.relative_path
                )));
            }
        }
    }
    let source_count = project
        .sources
        .packages
        .values()
        .map(|package| package.sources.len())
        .sum::<usize>();
    if project.access_paths.len() != source_count {
        return Err(FrozenProjectHandoffError::SourceCatalog(
            "access paths do not correspond one-to-one with source descriptors".to_string(),
        ));
    }
    let expected = compute_source_revision(&project.graph.graph_digest, &project.sources.packages)
        .map_err(|error| FrozenProjectHandoffError::Revision(error.to_string()))?;
    if expected != project.sources.revision {
        return Err(FrozenProjectHandoffError::SourceCatalog(
            "source revision does not match the canonical source catalog".to_string(),
        ));
    }
    Ok(())
}
