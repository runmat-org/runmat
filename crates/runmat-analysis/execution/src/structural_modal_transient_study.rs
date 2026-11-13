use std::collections::BTreeSet;

use runmat_analysis_core::StateUse;
use runmat_analysis_fea::execution::{
    AnalysisTaskIdentity, StructuralModalTransientTemplate, StructuralStepWorkload,
    StructuralStudyWorkloads, StructuralTransientBlockPlan,
};
use runmat_execution::{resource::AcceleratorRequest, Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::structural_modal_producer::{
    StructuralModalProducerCoordinator, StructuralModalProducerPreparation,
};
use crate::{
    bind_upstream_state_transition, prepare_structural_transient_sequence, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, ImportedSemanticState,
    ImportedStructuralModalResult, PreparedSemanticStatePublication, PreparedStructuralModalTask,
    StructuralTransientSequencePlan, StructuralTransientSequencePreparationInput,
};

/// Placement-neutral inputs for an authored modal-extraction -> modal-transient study.
pub struct StructuralModalTransientStudyPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
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

/// Binds a published modal basis into the authored transient consumer.
pub struct StructuralModalTransientStudyCoordinator {
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    producer: StructuralModalProducerCoordinator,
    mesh: SolverMeshArtifact,
    transient_template: StructuralModalTransientTemplate,
    block_plan: StructuralTransientBlockPlan,
    artifact_access: ArtifactAccess,
    program_revision: ProgramRevision,
    chunk_bytes: usize,
    inventory_limits: ObjectInventoryLimits,
}

impl StructuralModalTransientStudyCoordinator {
    pub fn new(
        preparation: StructuralModalTransientStudyPreparation,
    ) -> AnalysisBridgeResult<Self> {
        preparation.workloads.validate()?;
        preparation
            .mesh
            .validate_canonical()
            .map_err(|error| invalid(error.to_string()))?;
        let StructuralStudyWorkloads::HeterogeneousSequence { workloads } = &preparation.workloads
        else {
            return Err(invalid(
                "modal-transient coordination requires a heterogeneous authored sequence",
            ));
        };
        let ordered = preparation.workloads.step_ids()?;
        let [modal_step, transient_step] = ordered.as_slice() else {
            return Err(invalid(
                "modal-transient coordination requires exactly two authored steps",
            ));
        };
        let StructuralStepWorkload::ModalExtraction { workload } = &workloads[modal_step] else {
            return Err(invalid(
                "the first authored step must extract structural modes",
            ));
        };
        let StructuralStepWorkload::ModalTransient {
            template,
            block_plan,
        } = &workloads[transient_step]
        else {
            return Err(invalid(
                "the second authored step must be an unresolved modal transient",
            ));
        };
        let producer =
            StructuralModalProducerCoordinator::new(StructuralModalProducerPreparation {
                local_execution: preparation.local_execution,
                authored_study_digest: preparation.authored_study_digest,
                frozen_project_digest: preparation.frozen_project_digest,
                workload: (**workload).clone(),
                mesh: preparation.mesh.clone(),
                artifact_access: preparation.artifact_access.clone(),
                program_revision: preparation.program_revision.clone(),
                chunk_bytes: preparation.chunk_bytes,
                inventory_limits: preparation.inventory_limits,
                required_capabilities: preparation.required_capabilities,
                accelerator_requests: preparation.accelerator_requests,
            })?;
        Ok(Self {
            local_execution: preparation.local_execution,
            producer,
            mesh: preparation.mesh,
            transient_template: (**template).clone(),
            block_plan: *block_plan,
            artifact_access: preparation.artifact_access,
            program_revision: preparation.program_revision,
            chunk_bytes: preparation.chunk_bytes,
            inventory_limits: preparation.inventory_limits,
        })
    }

    pub fn prepare_modal(&mut self) -> AnalysisBridgeResult<PreparedStructuralModalTask> {
        self.producer.prepare_modal()
    }

    pub fn complete_modal(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralModalTask,
    ) -> AnalysisBridgeResult<()> {
        self.producer.complete_modal(source, response, prepared)
    }

    pub fn prepare_state(&mut self) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
        self.producer.prepare_state()
    }

    pub fn complete_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedSemanticStatePublication,
    ) -> AnalysisBridgeResult<()> {
        self.producer.complete_state(source, response, prepared)
    }

    pub fn prepare_transient(&self) -> AnalysisBridgeResult<StructuralTransientSequencePlan> {
        let state = self
            .producer
            .semantic_state()
            .ok_or_else(|| invalid("modal-transient study has not published its modal basis"))?;
        let transition = bind_upstream_state_transition(
            &self.transient_template.model,
            &self.transient_template.step_id,
            &state.state,
            StateUse::ModalBasis,
            Digest::from_bytes(*self.mesh.canonical_digest.bytes()),
        )?;
        let workload = self.transient_template.clone().bind(transition)?;
        prepare_structural_transient_sequence(StructuralTransientSequencePreparationInput {
            local_execution: self.local_execution,
            workload: &workload,
            mesh: &self.mesh,
            modal_state: Some(&state.state),
            block_plan: self.block_plan,
            initial_dependencies: vec![self
                .producer
                .state_task_identity()
                .ok_or_else(|| invalid("basis publication task identity is absent"))?],
            artifact_access: self.artifact_access.clone(),
            program_revision: self.program_revision.clone(),
            chunk_bytes: self.chunk_bytes,
            inventory_limits: self.inventory_limits,
        })
    }

    pub fn modal_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.producer.modal_task_identity()
    }

    pub fn state_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.producer.state_task_identity()
    }

    pub fn modal_result(&self) -> Option<&ImportedStructuralModalResult> {
        self.producer.modal_result()
    }

    pub fn semantic_state(&self) -> Option<&ImportedSemanticState> {
        self.producer.semantic_state()
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
