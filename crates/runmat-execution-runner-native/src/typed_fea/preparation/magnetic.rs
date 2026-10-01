//! Source-bound handoff before exact geometry/mesh and numerical admission exist.
//! This is not a completed magnetic result root or a new execution coordinator.
use std::collections::BTreeMap;

use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_analysis_execution::InputMaterializationPlan;
use runmat_analysis_fea::{
    planning::resource::LocalExecutionPlan, study::MagneticStudyPlanningPolicy,
};
use runmat_execution::{resource::AcceleratorRequest, Digest, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;

pub struct PreparedAuthoredMagneticStudy {
    pub geometry_meshing: crate::PreparedGeometryMeshingRoot,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub study: RealizedStudy,
    pub materials: BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub input_materializations: Vec<InputMaterializationPlan>,
    pub planning: MagneticStudyPlanningPolicy,
    pub local_execution: LocalExecutionPlan,
    pub analysis_accelerator_requests: Vec<AcceleratorRequest>,
}
