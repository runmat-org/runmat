//! Source-bound handoff before exact geometry/mesh and numerical admission exist.
//! This is not a completed thermal result root or a new execution coordinator.
use std::collections::BTreeMap;

use runmat_analysis_core::{AuthoredMaterialRef, MaterialModel, RealizedStudy};
use runmat_analysis_execution::PreparedInputMaterialization;
use runmat_analysis_fea::{
    planning::resource::LocalExecutionPlan, study::ThermalStudyPlanningPolicy,
};
use runmat_execution::{resource::AcceleratorRequest, Digest, ProgramRevision};
use runmat_execution_artifact::ArtifactAccess;

pub struct PreparedAuthoredThermalStudy {
    pub geometry_meshing: crate::PreparedGeometryMeshingRoot,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub study: RealizedStudy,
    pub materials: BTreeMap<AuthoredMaterialRef, MaterialModel>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub input_materializations: Vec<PreparedInputMaterialization>,
    pub planning: ThermalStudyPlanningPolicy,
    pub local_execution: LocalExecutionPlan,
    pub analysis_accelerator_requests: Vec<AcceleratorRequest>,
}
