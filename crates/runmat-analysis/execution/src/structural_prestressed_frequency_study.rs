use std::collections::BTreeSet;

use runmat_analysis_core::StateUse;
use runmat_analysis_fea::execution::{
    AnalysisTaskIdentity, CanonicalSemanticState, StructuralFrequencyWorkload,
    StructuralPrestressedFrequencySolverTemplate, StructuralPrestressedFrequencyTemplate,
    StructuralStepWorkload, StructuralStudyWorkloads,
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
    bind_upstream_state_transition, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, ImportedSemanticState, ImportedStructuralLinearResult,
    PreparedSemanticStatePublication, PreparedStructuralLinearProgram,
};

pub struct StructuralPrestressedFrequencyStudyPreparation {
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

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedStructuralPrestressedFrequency {
    pub workload: StructuralFrequencyWorkload,
    pub prestress_state: CanonicalSemanticState,
    pub modal_state: Option<CanonicalSemanticState>,
    pub state_task: AnalysisTaskIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConsumerPhase {
    AwaitingPrestress,
    Ready,
    Bound,
}

/// Resolves a two-step preload → direct prestressed-harmonic sequence only after the exact
/// geometric-stiffness state has been published.
pub struct StructuralPrestressedFrequencyStudyCoordinator {
    mesh: SolverMeshArtifact,
    producer: StructuralPrestressProducerCoordinator,
    template: StructuralPrestressedFrequencyTemplate,
    phase: ConsumerPhase,
}

impl StructuralPrestressedFrequencyStudyCoordinator {
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
                "prestressed-frequency coordination requires a heterogeneous authored sequence",
            ));
        };
        let prefix = preparation.workloads.prestress_prefix()?;
        let [frequency_step] = prefix.consumer_step_ids.as_slice() else {
            return Err(invalid(
                "direct prestressed-frequency coordination requires one preload consumer",
            ));
        };
        let StructuralStepWorkload::PrestressedFrequency { template } = &workloads[frequency_step]
        else {
            return Err(invalid(
                "the preload consumer must be unresolved prestressed harmonic response",
            ));
        };
        if !matches!(
            &template.solver,
            StructuralPrestressedFrequencySolverTemplate::Direct { .. }
        ) {
            return Err(invalid(
                "direct prestressed harmonic coordination requires a direct solver",
            ));
        }
        let producer = StructuralPrestressProducerCoordinator::new(
            StructuralPrestressProducerPreparation {
                thermal_workload: prefix.thermal.cloned(),
                authored_study_digest: preparation.authored_study_digest,
                frozen_project_digest: preparation.frozen_project_digest,
                preload_workload: prefix.preload.clone(),
                mesh: preparation.mesh.clone(),
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
            mesh: preparation.mesh,
            producer,
            template: (**template).clone(),
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
                "prestressed-frequency consumer has already bound a prestress state",
            ));
        }
        self.producer.complete_state(source, response, prepared)?;
        self.phase = ConsumerPhase::Ready;
        Ok(())
    }

    pub fn bind_frequency(
        &mut self,
    ) -> AnalysisBridgeResult<PreparedStructuralPrestressedFrequency> {
        if self.phase != ConsumerPhase::Ready {
            return Err(invalid(
                "prestressed-frequency study has not published its preload state",
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
        let workload = self.template.clone().bind(transition, None)?;
        let state_task = self
            .producer
            .state_task_identity()
            .ok_or_else(|| invalid("prestress publication task identity is absent"))?;
        self.phase = ConsumerPhase::Bound;
        Ok(PreparedStructuralPrestressedFrequency {
            workload,
            prestress_state: state.state.clone(),
            modal_state: None,
            state_task,
        })
    }

    pub fn preload_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.producer.preload_task_identity()
    }

    pub fn state_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.producer.state_task_identity()
    }

    pub fn preload_result(&self) -> Option<&ImportedStructuralLinearResult> {
        self.producer.preload_result()
    }

    pub fn semantic_state(&self) -> Option<&ImportedSemanticState> {
        self.producer.semantic_state()
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
