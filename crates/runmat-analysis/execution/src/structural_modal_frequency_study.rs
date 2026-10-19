use std::collections::BTreeSet;

use runmat_analysis_core::StateUse;
use runmat_analysis_fea::execution::{
    AnalysisTaskIdentity, CanonicalSemanticState, StructuralFrequencyWorkload,
    StructuralModalFrequencyTemplate, StructuralStepWorkload, StructuralStudyWorkloads,
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
    bind_upstream_state_transition, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, ImportedSemanticState, ImportedStructuralModalResult,
    PreparedSemanticStatePublication, PreparedStructuralModalTask,
};

pub struct StructuralModalFrequencyStudyPreparation {
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

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedStructuralModalFrequency {
    pub workload: StructuralFrequencyWorkload,
    pub modal_state: CanonicalSemanticState,
    pub state_task: AnalysisTaskIdentity,
}

/// Publishes a modal basis and binds it into a modal-reduced harmonic workload.
pub struct StructuralModalFrequencyStudyCoordinator {
    producer: StructuralModalProducerCoordinator,
    mesh: SolverMeshArtifact,
    template: StructuralModalFrequencyTemplate,
    bound: bool,
}

impl StructuralModalFrequencyStudyCoordinator {
    pub fn new(
        preparation: StructuralModalFrequencyStudyPreparation,
    ) -> AnalysisBridgeResult<Self> {
        preparation.workloads.validate()?;
        preparation
            .mesh
            .validate_canonical()
            .map_err(|error| invalid(error.to_string()))?;
        let StructuralStudyWorkloads::HeterogeneousSequence { workloads } = &preparation.workloads
        else {
            return Err(invalid(
                "modal-frequency coordination requires a heterogeneous authored sequence",
            ));
        };
        let ordered = preparation.workloads.step_ids()?;
        let [modal_step, frequency_step] = ordered.as_slice() else {
            return Err(invalid(
                "modal-frequency coordination requires exactly two authored steps",
            ));
        };
        let StructuralStepWorkload::ModalExtraction { workload } = &workloads[modal_step] else {
            return Err(invalid(
                "the first authored step must extract structural modes",
            ));
        };
        let StructuralStepWorkload::ModalFrequency { template } = &workloads[frequency_step] else {
            return Err(invalid(
                "the second authored step must be unresolved modal harmonic response",
            ));
        };
        let producer =
            StructuralModalProducerCoordinator::new(StructuralModalProducerPreparation {
                local_execution: preparation.local_execution,
                authored_study_digest: preparation.authored_study_digest,
                frozen_project_digest: preparation.frozen_project_digest,
                workload: (**workload).clone(),
                mesh: preparation.mesh.clone(),
                artifact_access: preparation.artifact_access,
                program_revision: preparation.program_revision,
                chunk_bytes: preparation.chunk_bytes,
                inventory_limits: preparation.inventory_limits,
                required_capabilities: preparation.required_capabilities,
                accelerator_requests: preparation.accelerator_requests,
            })?;
        Ok(Self {
            producer,
            mesh: preparation.mesh,
            template: (**template).clone(),
            bound: false,
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

    pub fn bind_frequency(&mut self) -> AnalysisBridgeResult<PreparedStructuralModalFrequency> {
        if self.bound {
            return Err(invalid("modal-frequency consumer is already bound"));
        }
        let state = self
            .producer
            .semantic_state()
            .ok_or_else(|| invalid("modal-frequency study has not published its modal basis"))?;
        let transition = bind_upstream_state_transition(
            &self.template.model,
            &self.template.step_id,
            &state.state,
            StateUse::ModalBasis,
            Digest::from_bytes(*self.mesh.canonical_digest.bytes()),
        )?;
        let workload = self.template.clone().bind(transition.source_state)?;
        let state_task = self
            .producer
            .state_task_identity()
            .ok_or_else(|| invalid("basis publication task identity is absent"))?;
        self.bound = true;
        Ok(PreparedStructuralModalFrequency {
            workload,
            modal_state: state.state.clone(),
            state_task,
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
