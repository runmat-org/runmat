use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AnalysisTaskIdentity, SemanticStatePublicationSource, StructuralModalWorkload,
};
use runmat_execution::{resource::AcceleratorRequest, Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    import_semantic_state_completion, import_structural_modal_result,
    prepare_semantic_state_publication, prepare_structural_modal_task, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, ImportedSemanticState,
    ImportedStructuralModalResult, PreparedSemanticStatePublication, PreparedStructuralModalTask,
    SemanticStatePublicationPreparation, StructuralModalTaskPreparation,
};

pub(crate) struct StructuralModalProducerPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub workload: StructuralModalWorkload,
    pub mesh: SolverMeshArtifact,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
    pub accelerator_requests: Vec<AcceleratorRequest>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProducerPhase {
    ReadyForModal,
    AwaitingModal(Digest),
    ReadyForState,
    AwaitingState(Digest),
    Complete,
}

/// Shared modal-extraction and typed basis-publication lifecycle for modal consumers.
pub(crate) struct StructuralModalProducerCoordinator {
    preparation: StructuralModalProducerPreparation,
    modal: Option<ImportedStructuralModalResult>,
    modal_task: Option<AnalysisTaskIdentity>,
    state: Option<ImportedSemanticState>,
    state_task: Option<AnalysisTaskIdentity>,
    phase: ProducerPhase,
}

impl StructuralModalProducerCoordinator {
    pub fn new(preparation: StructuralModalProducerPreparation) -> AnalysisBridgeResult<Self> {
        preparation.workload.validate()?;
        preparation
            .mesh
            .validate_canonical()
            .map_err(|error| invalid(error.to_string()))?;
        if preparation.chunk_bytes == 0 {
            return Err(invalid("modal producer requires bounded object chunks"));
        }
        preparation.workload.modal_basis_output()?;
        Ok(Self {
            preparation,
            modal: None,
            modal_task: None,
            state: None,
            state_task: None,
            phase: ProducerPhase::ReadyForModal,
        })
    }

    pub fn prepare_modal(&mut self) -> AnalysisBridgeResult<PreparedStructuralModalTask> {
        if self.phase != ProducerPhase::ReadyForModal {
            return Err(invalid("modal producer is not ready for extraction"));
        }
        let prepared = prepare_structural_modal_task(StructuralModalTaskPreparation {
            local_execution: self.preparation.local_execution,
            authored_study_digest: self.preparation.authored_study_digest,
            frozen_project_digest: self.preparation.frozen_project_digest,
            program_revision: self.preparation.program_revision.clone(),
            artifact_access: self.preparation.artifact_access.clone(),
            workload: self.preparation.workload.clone(),
            mesh: self.preparation.mesh.clone(),
            prestress_state: None,
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
            required_capabilities: self.preparation.required_capabilities.clone(),
            accelerator_requests: self.preparation.accelerator_requests.clone(),
            solve_selection: runmat_analysis_fea::planning::resource::ModalSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
        })?;
        self.phase = ProducerPhase::AwaitingModal(prepared.host.workload.identity.digest());
        Ok(prepared)
    }

    pub fn complete_modal(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralModalTask,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != ProducerPhase::AwaitingModal(prepared.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "modal response differs from the awaited authored step",
            ));
        }
        let modal = import_structural_modal_result(
            source,
            response,
            prepared,
            self.preparation.inventory_limits,
        )?;
        self.modal_task = Some(AnalysisTaskIdentity::new(
            modal.host.workload.identity,
            Vec::new(),
        )?);
        self.modal = Some(modal);
        self.phase = ProducerPhase::ReadyForState;
        Ok(())
    }

    pub fn prepare_state(&mut self) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
        if self.phase != ProducerPhase::ReadyForState {
            return Err(invalid("modal producer is not ready to publish its basis"));
        }
        let modal = self
            .modal
            .as_ref()
            .ok_or_else(|| invalid("completed modal result is absent"))?;
        let prepared = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: modal.host.clone(),
            source_result: modal.objects.clone(),
            source_profiles: modal.state_publication_profile_objects.clone(),
            source: SemanticStatePublicationSource::ModalBasis,
            declaration: self.preparation.workload.modal_basis_output()?.clone(),
            topology_digest: Digest::from_bytes(*self.preparation.mesh.canonical_digest.bytes()),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
        })?;
        self.phase = ProducerPhase::AwaitingState(prepared.host.workload.identity.digest());
        Ok(prepared)
    }

    pub fn complete_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedSemanticStatePublication,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != ProducerPhase::AwaitingState(prepared.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "basis publication differs from the awaited authored transition",
            ));
        }
        let state = import_semantic_state_completion(
            source,
            response,
            prepared,
            self.preparation.inventory_limits,
        )?;
        let modal_task = self
            .modal_task
            .ok_or_else(|| invalid("modal producer task identity is absent"))?;
        self.state_task = Some(AnalysisTaskIdentity::new(
            prepared.host.workload.identity,
            vec![modal_task],
        )?);
        self.state = Some(state);
        self.phase = ProducerPhase::Complete;
        Ok(())
    }

    pub fn modal_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.modal_task
    }

    pub fn state_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.state_task
    }

    pub fn modal_result(&self) -> Option<&ImportedStructuralModalResult> {
        self.modal.as_ref()
    }

    pub fn semantic_state(&self) -> Option<&ImportedSemanticState> {
        self.state.as_ref()
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
