//! Original authored replay inputs; decoding does not certify physics.
use super::*;
pub(in crate::nonlinear_study) struct ReplayContext {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub mesh: SolverMeshArtifact,
    pub accepted_block_limit: usize,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub output_policy: AnalysisOutputPolicy,
}
impl CompletedNonlinearSolidStudy {
    pub fn artifact_access(&self) -> &ArtifactAccess {
        &self.replay.artifact_access
    }
    pub fn output_policy(&self) -> AnalysisOutputPolicy {
        self.replay.output_policy
    }
    pub fn retained_component_bytes(&self) -> AnalysisBridgeResult<u64> {
        super::retained_storage(self)
    }
    pub fn mesh(&self) -> &SolverMeshArtifact {
        &self.replay.mesh
    }
    pub fn frozen_project_digest(&self) -> Option<Digest> {
        self.replay.frozen_project_digest
    }
    pub fn result_closures(&self) -> &[NonlinearSolidStudyResultClosure] {
        &self.closures
    }
    pub(in crate::nonlinear_study) fn preparation(
        &self,
    ) -> AnalysisBridgeResult<NonlinearSolidStudyPreparation> {
        Ok(NonlinearSolidStudyPreparation {
            local_execution: self.replay.local_execution,
            product: NonlinearSolidStudyProduct {
                compiled: self.compiled.clone(),
                workload: self.workload.clone(),
            },
            mesh: self.replay.mesh.clone(),
            accepted_block_limit: self.replay.accepted_block_limit,
            frozen_project_digest: self.replay.frozen_project_digest,
            program_revision: self.replay.program_revision.clone(),
            artifact_access: self.replay.artifact_access.clone(),
            output_policy: self.replay.output_policy,
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PreparationWire {
    #[serde(default = "runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial")]
    pub(super) local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub(super) accepted_block_limit: usize,
    pub(super) compiled: CompiledAnalysisStudy,
    pub(super) workload: NonlinearSolidWorkload,
    pub(super) mesh: SolverMeshArtifact,
    pub(super) frozen_project_digest: Option<Digest>,
    pub(super) program_revision: ProgramRevision,
    pub(super) chunk_bytes: usize,
    pub(super) max_objects: usize,
    pub(super) max_object_bytes: u64,
    pub(super) max_total_bytes: u64,
}

impl PreparationWire {
    pub(super) fn new(preparation: NonlinearSolidStudyPreparation) -> Self {
        let limits = preparation.output_policy.inventory_limits();
        let workload = preparation.product.workload;
        Self {
            local_execution: preparation.local_execution,
            accepted_block_limit: preparation.accepted_block_limit,
            compiled: preparation.product.compiled,
            workload,
            mesh: preparation.mesh,
            frozen_project_digest: preparation.frozen_project_digest,
            program_revision: preparation.program_revision,
            chunk_bytes: preparation.output_policy.chunk_bytes(),
            max_objects: limits.max_objects,
            max_object_bytes: limits.max_object_bytes,
            max_total_bytes: limits.max_total_bytes,
        }
    }

    pub(super) fn into_preparation(
        self,
        access: ArtifactAccess,
    ) -> AnalysisBridgeResult<NonlinearSolidStudyPreparation> {
        let output_policy = AnalysisOutputPolicy::new(
            self.chunk_bytes,
            ObjectInventoryLimits {
                max_objects: self.max_objects,
                max_object_bytes: self.max_object_bytes,
                max_total_bytes: self.max_total_bytes,
            },
        )?;
        if output_policy.chunk_bytes() != self.chunk_bytes {
            return Err(invalid("noncanonical nonlinear output chunk size"));
        }
        let workload = plan_nonlinear_solid_study_product(
            self.compiled.clone(),
            &self.mesh,
            self.workload.controls.clone(),
        )
        .map_err(|error| invalid(&error.to_string()))?
        .workload;
        if workload != self.workload {
            return Err(invalid(
                "nonlinear root workload differs from authored plan",
            ));
        }
        Ok(NonlinearSolidStudyPreparation {
            local_execution: self.local_execution,
            accepted_block_limit: self.accepted_block_limit,
            product: NonlinearSolidStudyProduct {
                compiled: self.compiled,
                workload,
            },
            mesh: self.mesh,
            frozen_project_digest: self.frozen_project_digest,
            artifact_access: access,
            program_revision: self.program_revision,
            output_policy,
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CompletionWire {
    pub(super) schema_version: u16,
    pub(super) preparation: PreparationWire,
    pub(super) closures: Vec<NonlinearSolidStudyResultClosure>,
    pub(super) descendants: PublishedObjectCatalog,
}
