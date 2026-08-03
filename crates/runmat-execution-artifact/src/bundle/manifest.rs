use std::collections::BTreeSet;

use runmat_execution::resource::Capability;
use runmat_execution::{Digest, ProgramRevision};
use serde::{Deserialize, Serialize};

use crate::{ArtifactResult, LogicalObject, ObjectDescriptor, ProgramArtifact, ProgramBuildRecipe};

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
}
