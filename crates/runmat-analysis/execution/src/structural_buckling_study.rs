use std::collections::BTreeSet;

use runmat_analysis_core::StateUse;
use runmat_analysis_fea::execution::{
    AnalysisTaskIdentity, StructuralBucklingTemplate, StructuralStepWorkload,
    StructuralStudyWorkloads,
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
    bind_upstream_state_transition, prepare_structural_buckling_task, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, ImportedSemanticState,
    ImportedStructuralBucklingResult, ImportedStructuralLinearResult,
    PreparedSemanticStatePublication, PreparedStructuralBucklingTask,
    PreparedStructuralLinearProgram, StructuralBucklingTaskPreparation,
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
    pub local_execution: LocalExecutionPlan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConsumerPhase {
    AwaitingPrestress,
    Ready,
    Awaiting(Digest),
    Complete,
}

/// Resolves a buckling template only after the shared producer publishes its exact preload state.
pub struct StructuralBucklingStudyCoordinator {
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
    buckling_template: StructuralBucklingTemplate,
    buckling: Option<ImportedStructuralBucklingResult>,
    buckling_task: Option<AnalysisTaskIdentity>,
    phase: ConsumerPhase,
}

impl StructuralBucklingStudyCoordinator {
    pub fn new(preparation: StructuralBucklingStudyPreparation) -> AnalysisBridgeResult<Self> {
        Self::with_source(preparation, None)
    }

    pub fn new_with_thermal_source(
        preparation: StructuralBucklingStudyPreparation,
        source: crate::ThermalPrestressSource,
    ) -> AnalysisBridgeResult<Self> {
        Self::with_source(preparation, Some(source))
    }

    fn with_source(
        preparation: StructuralBucklingStudyPreparation,
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
                "buckling coordination requires a heterogeneous authored sequence",
            ));
        };
        let prefix = preparation.workloads.prestress_prefix()?;
        let [buckling_step] = prefix.consumer_step_ids.as_slice() else {
            return Err(invalid(
                "buckling coordination requires exactly one preload and one buckling step",
            ));
        };
        let StructuralStepWorkload::Buckling { template } = &workloads[buckling_step] else {
            return Err(invalid(
                "the preload consumer must be an unresolved buckling solve",
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
            buckling_template: (**template).clone(),
            buckling: None,
            buckling_task: None,
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
                "buckling consumer has already bound a prestress state",
            ));
        }
        self.producer.complete_state(source, response, prepared)?;
        self.phase = ConsumerPhase::Ready;
        Ok(())
    }

    pub fn prepare_buckling(&mut self) -> AnalysisBridgeResult<PreparedStructuralBucklingTask> {
        if self.phase != ConsumerPhase::Ready {
            return Err(invalid(
                "buckling study has not published its prestress state",
            ));
        }
        let state = self
            .producer
            .semantic_state()
            .ok_or_else(|| invalid("published prestress state is absent"))?;
        let transition = bind_upstream_state_transition(
            &self.buckling_template.model,
            &self.buckling_template.step_id,
            &state.state,
            StateUse::Prestress,
            Digest::from_bytes(*self.mesh.canonical_digest.bytes()),
        )?;
        let workload = self.buckling_template.clone().bind(transition)?;
        let prepared = prepare_structural_buckling_task(StructuralBucklingTaskPreparation {
            local_execution: self.local_execution,
            authored_study_digest: self.authored_study_digest,
            frozen_project_digest: self.frozen_project_digest,
            program_revision: self.program_revision.clone(),
            artifact_access: self.artifact_access.clone(),
            workload,
            mesh: self.mesh.clone(),
            prestress_state: state.state.clone(),
            chunk_bytes: self.chunk_bytes,
            inventory_limits: self.producer.inventory_limits(),
            required_capabilities: self.required_capabilities.clone(),
            accelerator_requests: self.accelerator_requests.clone(),
        })?;
        self.phase = ConsumerPhase::Awaiting(prepared.host.workload.identity.digest());
        Ok(prepared)
    }

    pub fn complete_buckling(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        prepared: &PreparedStructuralBucklingTask,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != ConsumerPhase::Awaiting(prepared.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "buckling response differs from the awaited authored step",
            ));
        }
        self.buckling = Some(crate::import_structural_buckling_result(
            source,
            response,
            prepared,
            self.producer.inventory_limits(),
        )?);
        let state_task = self
            .producer
            .state_task_identity()
            .ok_or_else(|| invalid("prestress publication task identity is absent"))?;
        self.buckling_task = Some(AnalysisTaskIdentity::new(
            prepared.host.workload.identity,
            vec![state_task],
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

    pub const fn buckling_task_identity(&self) -> Option<AnalysisTaskIdentity> {
        self.buckling_task
    }

    pub fn preload_result(&self) -> Option<&ImportedStructuralLinearResult> {
        self.producer.preload_result()
    }

    pub fn semantic_state(&self) -> Option<&ImportedSemanticState> {
        self.producer.semantic_state()
    }

    pub fn buckling_result(&self) -> Option<&ImportedStructuralBucklingResult> {
        self.buckling.as_ref()
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
