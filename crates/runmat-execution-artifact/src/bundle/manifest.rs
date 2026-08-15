use std::collections::BTreeSet;

use runmat_execution::resource::Capability;
use runmat_execution::{Digest, ProgramRevision};
use serde::{Deserialize, Serialize};

use crate::{
    ArtifactError, ArtifactResult, BundleCodeClosure, LogicalObject, ObjectDescriptor,
    ProgramArtifact, ProgramBuildRecipe,
};

pub const EXECUTION_BUNDLE_SCHEMA_VERSION: u16 = 3;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectRevisionRecord {
    pub graph_digest: Digest,
    pub source_digest: Digest,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleCallable {
    pub owner_identity: String,
    pub qualified_name: String,
    pub source_digest: Digest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildResourceDeclaration {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub scratch_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleManifest {
    pub schema_version: u16,
    pub program_revision: ProgramRevision,
    pub project_revision: ProjectRevisionRecord,
    pub code_closure: BundleCodeClosure,
    pub sources: Vec<ObjectDescriptor>,
    pub callables: Vec<BundleCallable>,
    pub recipes: Vec<ProgramBuildRecipe>,
    pub artifacts: Vec<ProgramArtifact>,
    pub requested_capabilities: BTreeSet<Capability>,
    pub resources: BuildResourceDeclaration,
    pub portable_environment: Vec<(String, String)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionBundle {
    pub manifest: BundleManifest,
    pub objects: Vec<LogicalObject>,
}

impl ExecutionBundle {
    pub fn validate(&self) -> ArtifactResult<()> {
        super::validator::validate(self)
    }

    pub fn identity(&self) -> ArtifactResult<Digest> {
        self.validate()?;
        super::validator::identity(&self.manifest)
    }

    /// Rebind the bundle's logical source paths to one host-owned storage root.
    ///
    /// The returned handoff is the same frozen package graph and source catalog
    /// that the submitter used. Only the physical access paths change.
    pub fn project_handoff_at(
        &self,
        root: &std::path::Path,
    ) -> ArtifactResult<runmat_package::FrozenProjectHandoff> {
        self.validate()?;
        let BundleCodeClosure::SourceProject { handoff } = &self.manifest.code_closure else {
            return Err(ArtifactError::Invalid(
                "compiled execution bundle has no source project to materialize".into(),
            ));
        };
        let mut handoff = handoff.clone();
        handoff.project.workspace_root = root.to_path_buf();
        handoff.project.manifest_path = root.join("runmat.toml");
        for path in handoff.project.access_paths.values_mut() {
            let relative = runmat_package::NormalizedRelativePath::new(path.as_path())
                .map_err(|error| crate::ArtifactError::Invalid(error.to_string()))?;
            *path = root.join(relative.as_str());
        }
        handoff
            .validate()
            .map_err(|error| crate::ArtifactError::Invalid(error.to_string()))?;
        Ok(handoff)
    }

    pub fn requires_source_project(&self) -> bool {
        matches!(
            self.manifest.code_closure,
            BundleCodeClosure::SourceProject { .. }
        )
    }
}
