use std::collections::BTreeSet;

use runmat_analysis_core::{StateKind, StateUse};
use runmat_analysis_fea::execution::{
    AnalysisTaskIdentity, SemanticStatePublicationSource, StructuralBucklingTemplate,
    StructuralStepWorkload, StructuralStudyWorkloads,
};
use runmat_execution::{resource::AcceleratorRequest, Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    bind_upstream_state_transition, import_semantic_state_completion,
    import_structural_linear_completion, prepare_semantic_state_publication,
    prepare_structural_buckling_task, prepare_structural_sequence_step, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    ImportedSemanticState, ImportedStructuralBucklingResult, ImportedStructuralLinearResult,
    PreparedSemanticStatePublication, PreparedStructuralBucklingTask,
    PreparedStructuralLinearProgram, SemanticStatePublicationPreparation,
    StructuralBucklingTaskPreparation, StructuralLinearProgramPreparation,
};

/// Placement-neutral inputs for an authored preload → eigenvalue-buckling study.
pub struct StructuralBucklingStudyPreparation {
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub workloads: StructuralStudyWorkloads,
    pub mesh: SolverMeshArtifact,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
    pub accelerator_requests: Vec<AcceleratorRequest>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StudyPhase {
    ReadyForPreload,
    AwaitingPreload(Digest),
    ReadyForState,
    AwaitingState(Digest),
    ReadyForBuckling,
    AwaitingBuckling(Digest),
    Complete,
}

/// Resolves a buckling template only after its exact geometric-stiffness state is published.
///
/// Scheduling, attempts, placement, and transport remain outside this state machine. Each
/// completion is admitted against the exact immutable task prepared by the preceding transition.
pub struct StructuralBucklingStudyCoordinator {
    preparation: StructuralBucklingStudyPreparation,
    preload_workload: runmat_analysis_fea::execution::StructuralLinearWorkload,
    buckling_template: StructuralBucklingTemplate,
    preload: Option<ImportedStructuralLinearResult>,
    preload_host: Option<AnalysisHostWorkload>,
    preload_task: Option<AnalysisTaskIdentity>,
    state: Option<ImportedSemanticState>,
    state_task: Option<AnalysisTaskIdentity>,
    buckling: Option<ImportedStructuralBucklingResult>,
    buckling_task: Option<AnalysisTaskIdentity>,
    phase: StudyPhase,
}

impl StructuralBucklingStudyCoordinator {
    pub fn new(preparation: StructuralBucklingStudyPreparation) -> AnalysisBridgeResult<Self> {
        preparation.workloads.validate()?;
        preparation
            .mesh
            .validate_canonical()
            .map_err(|error| invalid(error.to_string()))?;
        if preparation.chunk_bytes == 0 {
            return Err(invalid("buckling study requires bounded object chunks"));
        }
        let StructuralStudyWorkloads::HeterogeneousSequence { workloads } = &preparation.workloads
        else {
            return Err(invalid(
                "buckling coordination requires a heterogeneous authored sequence",
            ));
        };
        let ordered = preparation.workloads.step_ids()?;
        let [preload_step, buckling_step] = ordered.as_slice() else {
            return Err(invalid(
                "buckling coordination requires exactly one preload and one buckling step",
            ));
        };
        let StructuralStepWorkload::Linear { workload } = &workloads[preload_step] else {
            return Err(invalid(
                "the first authored step must be a structural preload",
            ));
        };
        let StructuralStepWorkload::Buckling { template } = &workloads[buckling_step] else {
            return Err(invalid(
                "the second authored step must be an unresolved buckling solve",
            ));
        };
        let preload = workload.procedure_step()?;
        if preload
            .state_outputs
            .iter()
            .filter(|declaration| declaration.kind == StateKind::GeometricStiffness)
            .count()
            != 1
        {
            return Err(invalid(
                "buckling preload must publish exactly one geometric-stiffness state",
            ));
        }
        Ok(Self {
            preload_workload: (**workload).clone(),
            buckling_template: (**template).clone(),
            preparation,
            preload: None,
            preload_host: None,
            preload_task: None,
            state: None,
            state_task: None,
            buckling: None,
            buckling_task: None,
            phase: StudyPhase::ReadyForPreload,
        })
    }

    pub fn prepare_preload(&mut self) -> AnalysisBridgeResult<PreparedStructuralLinearProgram> {
        if self.phase != StudyPhase::ReadyForPreload {
            return Err(invalid("buckling study is not ready for its preload"));
        }
        let prepared = prepare_structural_sequence_step(StructuralLinearProgramPreparation {
            authored_study_digest: self.preparation.authored_study_digest,
            frozen_project_digest: self.preparation.frozen_project_digest,
            program_revision: self.preparation.program_revision.clone(),
            artifact_access: self.preparation.artifact_access.clone(),
            workload: self.preload_workload.clone(),
            mesh: Some(self.preparation.mesh.clone()),
            initial_guess: None,
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
            required_capabilities: self.preparation.required_capabilities.clone(),
            accelerator_requests: self.preparation.accelerator_requests.clone(),
        })?;
        self.phase = StudyPhase::AwaitingPreload(prepared.host.workload.identity.digest());
        Ok(prepared)
    }

    pub fn complete_preload(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralLinearProgram,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingPreload(prepared.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "preload response differs from the awaited authored step",
            ));
        }
        let preload = import_structural_linear_completion(
            source,
            response,
            prepared,
            self.preparation.inventory_limits,
        )?;
        self.preload_task = Some(AnalysisTaskIdentity::new(
            prepared.host.workload.identity,
            Vec::new(),
        )?);
        self.preload_host = Some(prepared.host.clone());
        self.preload = Some(preload);
        self.phase = StudyPhase::ReadyForState;
        Ok(())
    }

    pub fn prepare_state(&mut self) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
        if self.phase != StudyPhase::ReadyForState {
            return Err(invalid(
                "buckling study is not ready to publish geometric stiffness",
            ));
        }
        let preload = self
            .preload
            .as_ref()
            .ok_or_else(|| invalid("completed buckling preload is absent"))?;
        let host = self
            .preload_host
            .as_ref()
            .ok_or_else(|| invalid("completed buckling preload host is absent"))?;
        let declaration = self
            .preload_workload
            .procedure_step()?
            .state_outputs
            .iter()
            .find(|declaration| declaration.kind == StateKind::GeometricStiffness)
            .cloned()
            .ok_or_else(|| invalid("buckling preload state declaration is absent"))?;
        let prepared = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: host.clone(),
            source_result: preload.objects.clone(),
            source: SemanticStatePublicationSource::StructuralLinearization,
            declaration,
            topology_digest: preload
                .result
                .constraint_result
                .topology
                .discretization_digest(),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
        })?;
        self.phase = StudyPhase::AwaitingState(prepared.host.workload.identity.digest());
        Ok(prepared)
    }

    pub fn complete_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedSemanticStatePublication,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingState(prepared.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "prestress publication differs from the awaited authored transition",
            ));
        }
        let state = import_semantic_state_completion(
            source,
            response,
            prepared,
            self.preparation.inventory_limits,
        )?;
        let preload_task = self
            .preload_task
            .ok_or_else(|| invalid("preload producer task identity is absent"))?;
        self.state_task = Some(AnalysisTaskIdentity::new(
            prepared.host.workload.identity,
            vec![preload_task],
        )?);
        self.state = Some(state);
        self.phase = StudyPhase::ReadyForBuckling;
        Ok(())
    }

    pub fn prepare_buckling(&mut self) -> AnalysisBridgeResult<PreparedStructuralBucklingTask> {
        if self.phase != StudyPhase::ReadyForBuckling {
            return Err(invalid(
                "buckling study has not published its prestress state",
            ));
        }
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| invalid("published prestress state is absent"))?;
        let transition = bind_upstream_state_transition(
            &self.buckling_template.model,
            &self.buckling_template.step_id,
            &state.state,
            StateUse::Prestress,
            Digest::from_bytes(*self.preparation.mesh.canonical_digest.bytes()),
        )?;
        let workload = self.buckling_template.clone().bind(transition)?;
        let prepared = prepare_structural_buckling_task(StructuralBucklingTaskPreparation {
            authored_study_digest: self.preparation.authored_study_digest,
            frozen_project_digest: self.preparation.frozen_project_digest,
            program_revision: self.preparation.program_revision.clone(),
            artifact_access: self.preparation.artifact_access.clone(),
            workload,
            mesh: self.preparation.mesh.clone(),
            prestress_state: state.state.clone(),
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
            required_capabilities: self.preparation.required_capabilities.clone(),
            accelerator_requests: self.preparation.accelerator_requests.clone(),
        })?;
        self.phase = StudyPhase::AwaitingBuckling(prepared.host.workload.identity.digest());
        Ok(prepared)
    }

    pub fn complete_buckling(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralBucklingTask,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingBuckling(prepared.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "buckling response differs from the awaited authored step",
            ));
        }
        let buckling = crate::import_structural_buckling_result(
            source,
            response,
            prepared,
            self.preparation.inventory_limits,
        )?;
        let state_task = self
            .state_task
            .ok_or_else(|| invalid("prestress publication task identity is absent"))?;
        self.buckling_task = Some(AnalysisTaskIdentity::new(
            prepared.host.workload.identity,
            vec![state_task],
        )?);
        self.buckling = Some(buckling);
        self.phase = StudyPhase::Complete;
        Ok(())
    }

    pub fn preload_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.preload_task
    }

    pub fn state_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.state_task
    }

    pub fn buckling_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.buckling_task
    }

    pub fn preload_result(&self) -> Option<&ImportedStructuralLinearResult> {
        self.preload.as_ref()
    }

    pub fn semantic_state(&self) -> Option<&ImportedSemanticState> {
        self.state.as_ref()
    }

    pub fn buckling_result(&self) -> Option<&ImportedStructuralBucklingResult> {
        self.buckling.as_ref()
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
