use std::collections::BTreeSet;

use runmat_analysis_core::StateUse;
use runmat_analysis_fea::execution::{
    AnalysisTaskIdentity, SemanticStatePublicationSource, StructuralModalWorkload,
    StructuralPrestressedModalTemplate, StructuralStepWorkload, StructuralStudyWorkloads,
};
use runmat_analysis_fea::planning::resource::LocalExecutionPlan;
use runmat_execution::{resource::AcceleratorRequest, Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::structural_prestress_producer::{
    StructuralPrestressProducerCoordinator, StructuralPrestressProducerPreparation,
};
use crate::{
    bind_upstream_state_transition, import_semantic_state_completion,
    import_structural_modal_result, prepare_semantic_state_publication,
    prepare_structural_modal_task, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, ImportedSemanticState, ImportedStructuralLinearResult,
    ImportedStructuralModalResult, PreparedSemanticStatePublication,
    PreparedStructuralLinearProgram, PreparedStructuralModalTask,
    SemanticStatePublicationPreparation, StructuralModalTaskPreparation,
};

pub struct StructuralPrestressedModalStudyPreparation {
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
    pub local_execution: LocalExecutionPlan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConsumerPhase {
    AwaitingPrestress,
    Ready,
    Awaiting(Digest),
    ReadyForBasis,
    AwaitingBasis(Digest),
    Complete,
}

/// Binds prestressed-modal extraction only to the state published by its exact preload task.
pub struct StructuralPrestressedModalStudyCoordinator {
    local_execution: LocalExecutionPlan,
    authored_study_digest: Digest,
    frozen_project_digest: Option<Digest>,
    mesh: SolverMeshArtifact,
    artifact_access: ArtifactAccess,
    program_revision: ProgramRevision,
    chunk_bytes: usize,
    required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
    accelerator_requests: Vec<AcceleratorRequest>,
    producer: StructuralPrestressProducerCoordinator,
    template: StructuralPrestressedModalTemplate,
    bound_modal_workload: Option<StructuralModalWorkload>,
    modal: Option<ImportedStructuralModalResult>,
    modal_task: Option<AnalysisTaskIdentity>,
    basis_state: Option<ImportedSemanticState>,
    basis_state_task: Option<AnalysisTaskIdentity>,
    phase: ConsumerPhase,
}

impl StructuralPrestressedModalStudyCoordinator {
    pub fn new(
        preparation: StructuralPrestressedModalStudyPreparation,
    ) -> AnalysisBridgeResult<Self> {
        Self::with_source(preparation, None)
    }

    pub fn new_with_thermal_source(
        preparation: StructuralPrestressedModalStudyPreparation,
        source: crate::ThermalPrestressSource,
    ) -> AnalysisBridgeResult<Self> {
        Self::with_source(preparation, Some(source))
    }

    pub(super) fn with_source(
        preparation: StructuralPrestressedModalStudyPreparation,
        thermal_source: Option<crate::ThermalPrestressSource>,
    ) -> AnalysisBridgeResult<Self> {
        preparation.workloads.validate()?;
        preparation
            .mesh
            .validate_canonical()
            .map_err(|error| invalid(error.to_string()))?;
        let StructuralStudyWorkloads::HeterogeneousSequence { workloads } = &preparation.workloads
        else {
            return Err(invalid(
                "prestressed-modal coordination requires a heterogeneous authored sequence",
            ));
        };
        let prefix = preparation.workloads.prestress_prefix()?;
        let modal_step = match prefix.consumer_step_ids.as_slice() {
            [modal_step] | [modal_step, _] => modal_step,
            _ => {
                return Err(invalid(
                    "prestressed-modal coordination requires a preload and modal prefix",
                ))
            }
        };
        let StructuralStepWorkload::PrestressedModal { template } = &workloads[modal_step] else {
            return Err(invalid(
                "the preload consumer must be unresolved prestressed-modal extraction",
            ));
        };
        let producer = StructuralPrestressProducerCoordinator::new(
            StructuralPrestressProducerPreparation {
                thermal_workload: prefix.thermal.cloned(),
                authored_study_digest: preparation.authored_study_digest,
                frozen_project_digest: preparation.frozen_project_digest,
                preload_workload: prefix.preload.clone(),
                mesh: preparation.mesh.clone(),
                artifact_access: preparation.artifact_access.clone(),
                program_revision: preparation.program_revision.clone(),
                chunk_bytes: preparation.chunk_bytes,
                inventory_limits: preparation.inventory_limits,
                required_capabilities: preparation.required_capabilities.clone(),
                accelerator_requests: preparation.accelerator_requests.clone(),
                local_execution: preparation.local_execution,
            },
            thermal_source,
        )?;
        Ok(Self {
            local_execution: preparation.local_execution,
            authored_study_digest: preparation.authored_study_digest,
            frozen_project_digest: preparation.frozen_project_digest,
            mesh: preparation.mesh,
            artifact_access: preparation.artifact_access,
            program_revision: preparation.program_revision,
            chunk_bytes: preparation.chunk_bytes,
            required_capabilities: preparation.required_capabilities,
            accelerator_requests: preparation.accelerator_requests,
            producer,
            template: (**template).clone(),
            bound_modal_workload: None,
            modal: None,
            modal_task: None,
            basis_state: None,
            basis_state_task: None,
            phase: ConsumerPhase::AwaitingPrestress,
        })
    }

    pub fn prepare_preload(&mut self) -> AnalysisBridgeResult<PreparedStructuralLinearProgram> {
        self.producer.prepare_preload()
    }

    pub fn preload_dependencies(&self) -> Vec<AnalysisTaskIdentity> {
        self.producer.preload_dependencies()
    }

    pub fn thermal_source(&self) -> Option<&crate::ThermalPrestressSource> {
        self.producer.thermal_source()
    }

    pub fn complete_preload(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralLinearProgram,
    ) -> AnalysisBridgeResult<()> {
        self.producer.complete_preload(source, response, prepared)
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
        if self.phase != ConsumerPhase::AwaitingPrestress {
            return Err(invalid(
                "prestressed-modal consumer has already bound a prestress state",
            ));
        }
        self.producer.complete_state(source, response, prepared)?;
        self.phase = ConsumerPhase::Ready;
        Ok(())
    }

    pub fn prepare_modal(&mut self) -> AnalysisBridgeResult<PreparedStructuralModalTask> {
        if self.phase != ConsumerPhase::Ready {
            return Err(invalid(
                "prestressed-modal study has not published its preload state",
            ));
        }
        let state = self
            .producer
            .semantic_state()
            .ok_or_else(|| invalid("published prestress state is absent"))?;
        let transition = bind_upstream_state_transition(
            &self.template.model,
            &self.template.step_id,
            &state.state,
            StateUse::Prestress,
            Digest::from_bytes(*self.mesh.canonical_digest.bytes()),
        )?;
        let workload = self.template.clone().bind(transition)?;
        self.bound_modal_workload = Some(workload.clone());
        let prepared = prepare_structural_modal_task(StructuralModalTaskPreparation {
            local_execution: self.local_execution,
            authored_study_digest: self.authored_study_digest,
            frozen_project_digest: self.frozen_project_digest,
            program_revision: self.program_revision.clone(),
            artifact_access: self.artifact_access.clone(),
            workload,
            mesh: self.mesh.clone(),
            prestress_state: Some(state.state.clone()),
            chunk_bytes: self.chunk_bytes,
            inventory_limits: self.producer.inventory_limits(),
            required_capabilities: self.required_capabilities.clone(),
            accelerator_requests: self.accelerator_requests.clone(),
            solve_selection:
                runmat_analysis_fea::planning::resource::ModalSolveSelection::WithinCeiling {
                    machine_memory_bytes: None,
                },
        })?;
        self.phase = ConsumerPhase::Awaiting(prepared.host.workload.identity.digest());
        Ok(prepared)
    }

    pub fn complete_modal(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralModalTask,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != ConsumerPhase::Awaiting(prepared.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "prestressed-modal response differs from the awaited authored step",
            ));
        }
        self.modal = Some(import_structural_modal_result(
            source,
            response,
            prepared,
            self.producer.inventory_limits(),
        )?);
        let state_task = self
            .producer
            .state_task_identity()
            .ok_or_else(|| invalid("prestress publication task identity is absent"))?;
        self.modal_task = Some(AnalysisTaskIdentity::new(
            prepared.host.workload.identity,
            vec![state_task],
        )?);
        self.phase = ConsumerPhase::ReadyForBasis;
        Ok(())
    }

    pub fn prepare_basis_state(
        &mut self,
    ) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
        if self.phase != ConsumerPhase::ReadyForBasis {
            return Err(invalid(
                "prestressed-modal study is not ready to publish its modal basis",
            ));
        }
        let modal = self
            .modal
            .as_ref()
            .ok_or_else(|| invalid("completed prestressed-modal result is absent"))?;
        let prepared = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: modal.host.clone(),
            source_result: modal.objects.clone(),
            source_profiles: modal.state_publication_profile_objects.clone(),
            source: SemanticStatePublicationSource::ModalBasis,
            declaration: self
                .bound_modal_workload
                .as_ref()
                .ok_or_else(|| invalid("bound prestressed-modal workload is absent"))?
                .modal_basis_output()?
                .clone(),
            topology_digest: Digest::from_bytes(*self.mesh.canonical_digest.bytes()),
            program_revision: self.program_revision.clone(),
            chunk_bytes: self.chunk_bytes,
            inventory_limits: self.producer.inventory_limits(),
        })?;
        self.phase = ConsumerPhase::AwaitingBasis(prepared.host.workload.identity.digest());
        Ok(prepared)
    }

    pub fn complete_basis_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedSemanticStatePublication,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != ConsumerPhase::AwaitingBasis(prepared.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "modal-basis publication differs from the awaited prestressed-modal result",
            ));
        }
        self.basis_state = Some(import_semantic_state_completion(
            source,
            response,
            prepared,
            self.producer.inventory_limits(),
        )?);
        let modal_task = self
            .modal_task
            .ok_or_else(|| invalid("prestressed-modal task identity is absent"))?;
        self.basis_state_task = Some(AnalysisTaskIdentity::new(
            prepared.host.workload.identity,
            vec![modal_task],
        )?);
        self.phase = ConsumerPhase::Complete;
        Ok(())
    }

    pub fn preload_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.producer.preload_task_identity()
    }

    pub fn state_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.producer.state_task_identity()
    }

    pub const fn modal_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.modal_task
    }

    pub const fn basis_state_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.basis_state_task
    }

    pub fn preload_result(&self) -> Option<&ImportedStructuralLinearResult> {
        self.producer.preload_result()
    }

    pub fn semantic_state(&self) -> Option<&ImportedSemanticState> {
        self.producer.semantic_state()
    }

    pub fn modal_result(&self) -> Option<&ImportedStructuralModalResult> {
        self.modal.as_ref()
    }

    pub fn basis_state(&self) -> Option<&ImportedSemanticState> {
        self.basis_state.as_ref()
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
