use runmat_analysis_core::StateUse;
use runmat_analysis_fea::execution::{
    StructuralPrestressedFrequencySolverTemplate, StructuralPrestressedFrequencyTemplate,
    StructuralStepWorkload, StructuralStudyWorkloads,
};
use runmat_execution::Digest;
use runmat_execution_artifact::{cache::CacheImport, ProgramExecutionResponse};
use runmat_meshing_core::CanonicalMeshingContract;

use crate::{
    bind_upstream_state_transition, AnalysisBridgeError, AnalysisBridgeResult,
    ImportedSemanticState, ImportedStructuralLinearResult, ImportedStructuralModalResult,
    PreparedSemanticStatePublication, PreparedStructuralLinearProgram, PreparedStructuralModalTask,
    PreparedStructuralPrestressedFrequency, StructuralPrestressedFrequencyStudyPreparation,
    StructuralPrestressedModalStudyCoordinator, StructuralPrestressedModalStudyPreparation,
};

/// Resolves preload → prestressed modal → modal-reduced prestressed harmonic execution while
/// retaining the exact geometric-stiffness and modal-basis publications as separate authorities.
pub struct StructuralPrestressedModalFrequencyStudyCoordinator {
    mesh_digest: Digest,
    modal: StructuralPrestressedModalStudyCoordinator,
    frequency_template: StructuralPrestressedFrequencyTemplate,
}

impl StructuralPrestressedModalFrequencyStudyCoordinator {
    pub fn new(
        preparation: StructuralPrestressedFrequencyStudyPreparation,
    ) -> AnalysisBridgeResult<Self> {
        Self::with_source(preparation, None)
    }

    pub fn new_with_thermal_source(
        preparation: StructuralPrestressedFrequencyStudyPreparation,
        source: crate::ThermalPrestressSource,
    ) -> AnalysisBridgeResult<Self> {
        Self::with_source(preparation, Some(source))
    }

    fn with_source(
        preparation: StructuralPrestressedFrequencyStudyPreparation,
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
                "prestressed modal-frequency coordination requires a heterogeneous sequence",
            ));
        };
        let prefix = preparation.workloads.prestress_prefix()?;
        let [modal_step, frequency_step] = prefix.consumer_step_ids.as_slice() else {
            return Err(invalid(
                "prestressed modal-frequency coordination requires modal and frequency preload consumers",
            ));
        };
        if !matches!(
            workloads.get(modal_step),
            Some(StructuralStepWorkload::PrestressedModal { .. })
        ) {
            return Err(invalid(
                "prestressed modal-frequency coordination requires preload and prestressed-modal producers",
            ));
        }
        let Some(StructuralStepWorkload::PrestressedFrequency { template }) =
            workloads.get(frequency_step)
        else {
            return Err(invalid(
                "prestressed modal-frequency coordination requires a harmonic consumer",
            ));
        };
        if template.solver != StructuralPrestressedFrequencySolverTemplate::ModalReduced {
            return Err(invalid(
                "three-step prestressed harmonic coordination requires modal reduction",
            ));
        }
        let frequency_template = (**template).clone();
        let mesh_digest = Digest::from_bytes(*preparation.mesh.canonical_digest.bytes());
        let modal = StructuralPrestressedModalStudyCoordinator::with_source(
            StructuralPrestressedModalStudyPreparation {
                authored_study_digest: preparation.authored_study_digest,
                frozen_project_digest: preparation.frozen_project_digest,
                workloads: preparation.workloads,
                mesh: preparation.mesh,
                artifact_access: preparation.artifact_access,
                program_revision: preparation.program_revision,
                chunk_bytes: preparation.chunk_bytes,
                inventory_limits: preparation.inventory_limits,
                required_capabilities: preparation.required_capabilities,
                accelerator_requests: preparation.accelerator_requests,
                local_execution: preparation.local_execution,
            },
            thermal_source,
        )?;
        Ok(Self {
            mesh_digest,
            modal,
            frequency_template,
        })
    }

    pub fn prepare_preload(&mut self) -> AnalysisBridgeResult<PreparedStructuralLinearProgram> {
        self.modal.prepare_preload()
    }

    pub fn preload_dependencies(
        &self,
    ) -> Vec<runmat_analysis_fea::execution::AnalysisTaskIdentity> {
        self.modal.preload_dependencies()
    }

    pub fn thermal_source(&self) -> Option<&crate::ThermalPrestressSource> {
        self.modal.thermal_source()
    }

    pub fn complete_preload(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralLinearProgram,
    ) -> AnalysisBridgeResult<()> {
        self.modal.complete_preload(source, response, prepared)
    }

    pub fn prepare_prestress_state(
        &mut self,
    ) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
        self.modal.prepare_state()
    }

    pub fn complete_prestress_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedSemanticStatePublication,
    ) -> AnalysisBridgeResult<()> {
        self.modal.complete_state(source, response, prepared)
    }

    pub fn prepare_modal(&mut self) -> AnalysisBridgeResult<PreparedStructuralModalTask> {
        self.modal.prepare_modal()
    }

    pub fn complete_modal(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralModalTask,
    ) -> AnalysisBridgeResult<()> {
        self.modal.complete_modal(source, response, prepared)
    }

    pub fn prepare_basis_state(
        &mut self,
    ) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
        self.modal.prepare_basis_state()
    }

    pub fn complete_basis_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedSemanticStatePublication,
    ) -> AnalysisBridgeResult<()> {
        self.modal.complete_basis_state(source, response, prepared)
    }

    pub fn bind_frequency(&self) -> AnalysisBridgeResult<PreparedStructuralPrestressedFrequency> {
        let prestress = self
            .modal
            .semantic_state()
            .ok_or_else(|| invalid("published prestress state is absent"))?;
        let basis = self
            .modal
            .basis_state()
            .ok_or_else(|| invalid("published modal-basis state is absent"))?;
        let prestress_transition = bind_upstream_state_transition(
            &self.frequency_template.model,
            &self.frequency_template.step_id,
            &prestress.state,
            StateUse::Prestress,
            self.mesh_digest,
        )?;
        bind_upstream_state_transition(
            &self.frequency_template.model,
            &self.frequency_template.step_id,
            &basis.state,
            StateUse::ModalBasis,
            self.mesh_digest,
        )?;
        let workload = self
            .frequency_template
            .clone()
            .bind(prestress_transition, Some(basis.state.identity))?;
        Ok(PreparedStructuralPrestressedFrequency {
            workload,
            prestress_state: prestress.state.clone(),
            modal_state: Some(basis.state.clone()),
            state_task: self
                .modal
                .basis_state_task_identity()
                .ok_or_else(|| invalid("modal-basis publication task identity is absent"))?,
        })
    }

    pub fn preload_result(&self) -> Option<&ImportedStructuralLinearResult> {
        self.modal.preload_result()
    }

    pub fn prestress_state(&self) -> Option<&ImportedSemanticState> {
        self.modal.semantic_state()
    }

    pub fn modal_result(&self) -> Option<&ImportedStructuralModalResult> {
        self.modal.modal_result()
    }

    pub fn basis_state(&self) -> Option<&ImportedSemanticState> {
        self.modal.basis_state()
    }

    pub fn preload_task_identity(
        &self,
    ) -> Option<runmat_analysis_fea::execution::AnalysisTaskIdentity> {
        self.modal.preload_task_identity()
    }

    pub fn prestress_state_task_identity(
        &self,
    ) -> Option<runmat_analysis_fea::execution::AnalysisTaskIdentity> {
        self.modal.state_task_identity()
    }

    pub fn modal_task_identity(
        &self,
    ) -> Option<runmat_analysis_fea::execution::AnalysisTaskIdentity> {
        self.modal.modal_task_identity()
    }

    pub fn basis_state_task_identity(
        &self,
    ) -> Option<runmat_analysis_fea::execution::AnalysisTaskIdentity> {
        self.modal.basis_state_task_identity()
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
