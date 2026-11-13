use std::collections::BTreeSet;

use runmat_analysis_core::StateKind;
use runmat_analysis_fea::execution::{
    AnalysisTaskIdentity, SemanticStatePublicationSource, StructuralLinearWorkload,
};
use runmat_analysis_fea::planning::resource::LocalExecutionPlan;
use runmat_execution::{resource::AcceleratorRequest, Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    import_semantic_state_completion, import_structural_linear_completion,
    prepare_semantic_state_publication, prepare_structural_sequence_step, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    ImportedSemanticState, ImportedStructuralLinearResult, PreparedSemanticStatePublication,
    PreparedStructuralLinearProgram, SemanticStatePublicationPreparation,
    StructuralLinearProgramPreparation,
};

mod thermal_source;
pub use thermal_source::ThermalPrestressSource;

pub(crate) struct StructuralPrestressProducerPreparation {
    pub thermal_workload: Option<runmat_analysis_fea::execution::ThermalWorkload>,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub preload_workload: StructuralLinearWorkload,
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
enum ProducerPhase {
    ReadyForPreload,
    AwaitingPreload(Digest),
    ReadyForState,
    AwaitingState(Digest),
    Complete,
}

/// Shared exact preload and geometric-stiffness publication state machine.
///
/// Consumer-specific binding stays in the buckling or prestressed-modal coordinator. This type is
/// the sole owner of producer ordering, imported artifacts, and task dependency identities.
pub(crate) struct StructuralPrestressProducerCoordinator {
    preparation: StructuralPrestressProducerPreparation,
    preload: Option<ImportedStructuralLinearResult>,
    preload_host: Option<AnalysisHostWorkload>,
    preload_task: Option<AnalysisTaskIdentity>,
    state: Option<ImportedSemanticState>,
    state_task: Option<AnalysisTaskIdentity>,
    phase: ProducerPhase,
    thermal_source: Option<ThermalPrestressSource>,
    thermal_input: Option<thermal_source::BoundThermalPrestressSource>,
}

impl StructuralPrestressProducerCoordinator {
    pub(crate) fn new(
        preparation: StructuralPrestressProducerPreparation,
        thermal_source: Option<ThermalPrestressSource>,
    ) -> AnalysisBridgeResult<Self> {
        preparation
            .mesh
            .validate_canonical()
            .map_err(|error| invalid(error.to_string()))?;
        if preparation.chunk_bytes == 0
            || preparation
                .preload_workload
                .procedure_step()?
                .state_outputs
                .iter()
                .filter(|declaration| declaration.kind == StateKind::GeometricStiffness)
                .count()
                != 1
        {
            return Err(invalid(
                "structural prestress production requires bounded chunks and exactly one geometric-stiffness output",
            ));
        }
        let thermal_input = match (&preparation.thermal_workload, &thermal_source) {
            (Some(workload), Some(source)) => Some(source.bind(workload, &preparation)?),
            (None, None) => None,
            _ => {
                return Err(invalid(
                    "thermal prestress inventory and authenticated source must agree",
                ))
            }
        };
        Ok(Self {
            preparation,
            preload: None,
            preload_host: None,
            preload_task: None,
            state: None,
            state_task: None,
            phase: ProducerPhase::ReadyForPreload,
            thermal_source,
            thermal_input,
        })
    }

    pub(crate) fn prepare_preload(
        &mut self,
    ) -> AnalysisBridgeResult<PreparedStructuralLinearProgram> {
        if self.phase != ProducerPhase::ReadyForPreload {
            return Err(invalid("prestress producer is not ready for its preload"));
        }
        let prepared = prepare_structural_sequence_step(StructuralLinearProgramPreparation {
            authored_study_digest: self.preparation.authored_study_digest,
            frozen_project_digest: self.preparation.frozen_project_digest,
            program_revision: self.preparation.program_revision.clone(),
            artifact_access: self.preparation.artifact_access.clone(),
            workload: self.preparation.preload_workload.clone(),
            mesh: Some(self.preparation.mesh.clone()),
            state_input: self
                .thermal_input
                .as_ref()
                .map(|source| source.objects.clone()),
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
            required_capabilities: self.preparation.required_capabilities.clone(),
            accelerator_requests: self.preparation.accelerator_requests.clone(),
            local_execution: self.preparation.local_execution,
            solve_selection:
                runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                    machine_memory_bytes: None,
                },
        })?;
        self.phase = ProducerPhase::AwaitingPreload(prepared.host.workload.identity.digest());
        Ok(prepared)
    }

    pub(crate) fn complete_preload(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralLinearProgram,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != ProducerPhase::AwaitingPreload(prepared.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "preload response differs from the awaited prestress producer step",
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
            self.preload_dependencies(),
        )?);
        self.preload_host = Some(prepared.host.clone());
        self.preload = Some(preload);
        self.phase = ProducerPhase::ReadyForState;
        Ok(())
    }

    pub(crate) fn prepare_state(
        &mut self,
    ) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
        if self.phase != ProducerPhase::ReadyForState {
            return Err(invalid(
                "prestress producer is not ready to publish geometric stiffness",
            ));
        }
        let preload = self
            .preload
            .as_ref()
            .ok_or_else(|| invalid("completed prestress preload is absent"))?;
        let host = self
            .preload_host
            .as_ref()
            .ok_or_else(|| invalid("completed prestress preload host is absent"))?;
        let declaration = self
            .preparation
            .preload_workload
            .procedure_step()?
            .state_outputs
            .iter()
            .find(|declaration| declaration.kind == StateKind::GeometricStiffness)
            .cloned()
            .ok_or_else(|| invalid("prestress state declaration is absent"))?;
        let prepared = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: host.clone(),
            source_result: preload.objects.clone(),
            source_profiles: preload.state_publication_profile_objects.clone(),
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
        self.phase = ProducerPhase::AwaitingState(prepared.host.workload.identity.digest());
        Ok(prepared)
    }

    pub(crate) fn complete_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedSemanticStatePublication,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != ProducerPhase::AwaitingState(prepared.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "prestress publication differs from the awaited producer transition",
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
            .ok_or_else(|| invalid("prestress producer task identity is absent"))?;
        self.state_task = Some(AnalysisTaskIdentity::new(
            prepared.host.workload.identity,
            vec![preload_task],
        )?);
        self.state = Some(state);
        self.phase = ProducerPhase::Complete;
        Ok(())
    }

    pub(crate) const fn preload_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.preload_task
    }

    pub(crate) const fn state_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.state_task
    }

    pub(crate) fn preload_result(&self) -> Option<&ImportedStructuralLinearResult> {
        self.preload.as_ref()
    }

    pub(crate) fn semantic_state(&self) -> Option<&ImportedSemanticState> {
        self.state.as_ref()
    }

    pub(crate) const fn inventory_limits(&self) -> ObjectInventoryLimits {
        self.preparation.inventory_limits
    }

    pub(crate) fn preload_dependencies(&self) -> Vec<AnalysisTaskIdentity> {
        self.thermal_input
            .as_ref()
            .map(|source| source.dependency)
            .into_iter()
            .collect()
    }

    pub(crate) fn thermal_source(&self) -> Option<&ThermalPrestressSource> {
        self.thermal_source.as_ref()
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
