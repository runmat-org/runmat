use std::collections::BTreeSet;

use runmat_analysis_core::StateUse;
use runmat_analysis_fea::execution::{
    AnalysisTaskIdentity, SemanticStatePublicationSource, StructuralModalTransientTemplate,
    StructuralModalWorkload, StructuralStepWorkload, StructuralStudyWorkloads,
    StructuralTransientBlockPlan,
};
use runmat_execution::{resource::AcceleratorRequest, Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    bind_upstream_state_transition, import_semantic_state_completion,
    import_structural_modal_result, prepare_semantic_state_publication,
    prepare_structural_modal_task, prepare_structural_transient_sequence, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, ImportedSemanticState,
    ImportedStructuralModalResult, PreparedSemanticStatePublication, PreparedStructuralModalTask,
    SemanticStatePublicationPreparation, StructuralModalTaskPreparation,
    StructuralTransientSequencePreparation, StructuralTransientSequencePreparationInput,
};

/// Placement-neutral inputs for an authored modal-extraction → modal-transient study.
pub struct StructuralModalTransientStudyPreparation {
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
    ReadyForModal,
    AwaitingModal(Digest),
    ReadyForState,
    AwaitingState(Digest),
    ReadyForTransient,
}

/// Binds produced identities into an authored heterogeneous structural sequence.
///
/// Numerical work, scheduler placement, and live task identifiers remain outside this state
/// machine. Each transition accepts only the exact portable response it previously prepared.
pub struct StructuralModalTransientStudyCoordinator {
    preparation: StructuralModalTransientStudyPreparation,
    modal_workload: StructuralModalWorkload,
    transient_template: StructuralModalTransientTemplate,
    block_plan: StructuralTransientBlockPlan,
    modal: Option<ImportedStructuralModalResult>,
    modal_task: Option<AnalysisTaskIdentity>,
    state: Option<ImportedSemanticState>,
    state_task: Option<AnalysisTaskIdentity>,
    phase: StudyPhase,
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
        if preparation.chunk_bytes == 0 {
            return Err(invalid(
                "modal-transient study requires bounded object chunks",
            ));
        }
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
        workload.modal_basis_output()?;
        Ok(Self {
            modal_workload: (**workload).clone(),
            transient_template: (**template).clone(),
            block_plan: *block_plan,
            preparation,
            modal: None,
            modal_task: None,
            state: None,
            state_task: None,
            phase: StudyPhase::ReadyForModal,
        })
    }

    pub fn prepare_modal(&mut self) -> AnalysisBridgeResult<PreparedStructuralModalTask> {
        if self.phase != StudyPhase::ReadyForModal {
            return Err(invalid(
                "modal-transient study is not ready for modal extraction",
            ));
        }
        let prepared = prepare_structural_modal_task(StructuralModalTaskPreparation {
            authored_study_digest: self.preparation.authored_study_digest,
            frozen_project_digest: self.preparation.frozen_project_digest,
            program_revision: self.preparation.program_revision.clone(),
            artifact_access: self.preparation.artifact_access.clone(),
            workload: self.modal_workload.clone(),
            mesh: self.preparation.mesh.clone(),
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
            required_capabilities: self.preparation.required_capabilities.clone(),
            accelerator_requests: self.preparation.accelerator_requests.clone(),
        })?;
        self.phase = StudyPhase::AwaitingModal(prepared.host.workload.identity.digest());
        Ok(prepared)
    }

    pub fn complete_modal(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralModalTask,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingModal(prepared.host.workload.identity.digest()) {
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
        self.phase = StudyPhase::ReadyForState;
        Ok(())
    }

    pub fn prepare_state(&mut self) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
        if self.phase != StudyPhase::ReadyForState {
            return Err(invalid(
                "modal-transient study is not ready to publish a basis",
            ));
        }
        let modal = self
            .modal
            .as_ref()
            .ok_or_else(|| invalid("completed modal result is absent"))?;
        let prepared = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: modal.host.clone(),
            source_result: modal.objects.clone(),
            source: SemanticStatePublicationSource::ModalBasis,
            declaration: self.modal_workload.modal_basis_output()?.clone(),
            topology_digest: Digest::from_bytes(*self.preparation.mesh.canonical_digest.bytes()),
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
        self.phase = StudyPhase::ReadyForTransient;
        Ok(())
    }

    pub fn prepare_transient(
        &self,
    ) -> AnalysisBridgeResult<StructuralTransientSequencePreparation> {
        if self.phase != StudyPhase::ReadyForTransient {
            return Err(invalid(
                "modal-transient study has not published its modal basis",
            ));
        }
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| invalid("published modal state is absent"))?;
        let transition = bind_upstream_state_transition(
            &self.transient_template.model,
            &self.transient_template.step_id,
            &state.state,
            StateUse::ModalBasis,
            Digest::from_bytes(*self.preparation.mesh.canonical_digest.bytes()),
        )?;
        let workload = self.transient_template.clone().bind(transition)?;
        prepare_structural_transient_sequence(StructuralTransientSequencePreparationInput {
            workload: &workload,
            mesh: &self.preparation.mesh,
            modal_state: Some(&state.state),
            block_plan: self.block_plan,
            initial_dependencies: vec![self
                .state_task
                .ok_or_else(|| invalid("basis publication task identity is absent"))?],
            artifact_access: self.preparation.artifact_access.clone(),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
        })
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
