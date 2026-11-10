use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{StateKind, StateUse};
use runmat_analysis_fea::execution::{
    AccuracySourceKind, AnalysisNumericalObjectKind, SemanticStatePublicationSource,
    SequenceIdentity, StructuralModalWorkload, StructuralRandomResponseTemplate,
    StructuralRandomResponseWorkload, StructuralStepWorkload, StructuralStudyWorkloads,
};
use runmat_execution::{resource::AcceleratorRequest, Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    bind_upstream_state_transition, import_accuracy, import_published_result,
    import_semantic_state_completion, import_structural_modal_result,
    import_structural_random_response_interval, import_structural_random_response_result,
    prepare_accuracy, prepare_result_publication, prepare_semantic_state_publication,
    prepare_structural_modal_task, prepare_structural_random_response_assembly,
    prepare_structural_random_response_fanout, AccuracyPreparation, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, ImportedAccuracy, ImportedPublishedResult,
    ImportedSemanticState, ImportedStructuralModalResult, ImportedStructuralRandomResponseInterval,
    ImportedStructuralRandomResponseResult, PreparedAccuracy, PreparedResourceUsage,
    PreparedResultPublication, PreparedSemanticStatePublication, PreparedStructuralModalTask,
    PreparedStructuralRandomResponseAssemblyTask, PreparedStructuralRandomResponseIntervalTask,
    ResultPublicationPreparation, SemanticStatePublicationPreparation,
    StructuralModalTaskPreparation, StructuralRandomResponseAssemblyPreparation,
    StructuralRandomResponseFanoutPreparation,
};

pub struct StructuralRandomResponseStudyPreparation {
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

#[derive(Clone, Debug, Eq, PartialEq)]
enum StudyPhase {
    ReadyForModal,
    AwaitingModal(Digest),
    ReadyForBasisState,
    AwaitingBasisState(Digest),
    ReadyForIntervals,
    AwaitingIntervals(BTreeSet<Digest>),
    ReadyForAssembly,
    AwaitingAssembly(Digest),
    Observing,
    ReadyForAccuracy,
    AwaitingAccuracy(Digest),
    ReadyForPublication,
    AwaitingPublication(Digest),
    ReadyForStatisticsState,
    AwaitingStatisticsState(Digest),
    Complete,
}

/// Placement-neutral lifecycle for an authored modal random-response study.
pub struct StructuralRandomResponseStudyCoordinator {
    preparation: StructuralRandomResponseStudyPreparation,
    modal_workload: StructuralModalWorkload,
    response_template: StructuralRandomResponseTemplate,
    response_workload: Option<StructuralRandomResponseWorkload>,
    modal: Option<ImportedStructuralModalResult>,
    basis_state: Option<ImportedSemanticState>,
    intervals: Vec<ImportedStructuralRandomResponseInterval>,
    assembly_host: Option<crate::AnalysisHostWorkload>,
    assembly_source_context_inputs: Vec<crate::PreparedAnalysisObjects>,
    result: Option<ImportedStructuralRandomResponseResult>,
    accuracy_host: Option<crate::AnalysisHostWorkload>,
    accuracy: Option<ImportedAccuracy>,
    resource_usage: Option<PreparedResourceUsage>,
    publication: Option<ImportedPublishedResult>,
    statistics_state: Option<ImportedSemanticState>,
    observations: Option<crate::StatisticalStudyObservations>,
    phase: StudyPhase,
}

impl StructuralRandomResponseStudyCoordinator {
    pub fn new(
        preparation: StructuralRandomResponseStudyPreparation,
    ) -> AnalysisBridgeResult<Self> {
        preparation.workloads.validate()?;
        preparation
            .mesh
            .validate_canonical()
            .map_err(|error| invalid(error.to_string()))?;
        if preparation.chunk_bytes == 0
            || !preparation.accelerator_requests.is_empty()
            || !preparation.required_capabilities.iter().all(|capability| {
                !matches!(
                    capability,
                    AnalysisCapabilityRequirement::Accelerator { .. }
                )
            })
        {
            return Err(invalid(
                "random-response studies require bounded chunks and CPU placement",
            ));
        }
        let StructuralStudyWorkloads::HeterogeneousSequence { workloads } = &preparation.workloads
        else {
            return Err(invalid(
                "random-response coordination requires a heterogeneous authored sequence",
            ));
        };
        let ordered = preparation.workloads.step_ids()?;
        let [modal_step, response_step] = ordered.as_slice() else {
            return Err(invalid(
                "random-response coordination requires exactly two authored steps",
            ));
        };
        let StructuralStepWorkload::ModalExtraction { workload } = &workloads[modal_step] else {
            return Err(invalid(
                "the first random-response study step must extract structural modes",
            ));
        };
        let StructuralStepWorkload::RandomResponse { template } = &workloads[response_step] else {
            return Err(invalid(
                "the second random-response study step must consume the modal basis",
            ));
        };
        workload.modal_basis_output()?;
        Ok(Self {
            modal_workload: (**workload).clone(),
            response_template: (**template).clone(),
            preparation,
            response_workload: None,
            modal: None,
            basis_state: None,
            intervals: Vec::new(),
            assembly_host: None,
            assembly_source_context_inputs: Vec::new(),
            result: None,
            accuracy_host: None,
            accuracy: None,
            resource_usage: None,
            publication: None,
            statistics_state: None,
            observations: None,
            phase: StudyPhase::ReadyForModal,
        })
    }

    pub fn prepare_modal(&mut self) -> AnalysisBridgeResult<PreparedStructuralModalTask> {
        if self.phase != StudyPhase::ReadyForModal {
            return Err(invalid(
                "random-response study is not ready for modal extraction",
            ));
        }
        let task = prepare_structural_modal_task(StructuralModalTaskPreparation {
            local_execution: self.preparation.local_execution,
            authored_study_digest: self.preparation.authored_study_digest,
            frozen_project_digest: self.preparation.frozen_project_digest,
            program_revision: self.preparation.program_revision.clone(),
            artifact_access: self.preparation.artifact_access.clone(),
            workload: self.modal_workload.clone(),
            mesh: self.preparation.mesh.clone(),
            prestress_state: None,
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
            required_capabilities: self.preparation.required_capabilities.clone(),
            accelerator_requests: Vec::new(),
            solve_selection:
                runmat_analysis_fea::planning::resource::ModalSolveSelection::WithinCeiling {
                    machine_memory_bytes: None,
                },
        })?;
        self.phase = StudyPhase::AwaitingModal(task.host.workload.identity.digest());
        Ok(task)
    }

    pub fn complete_modal(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        task: &PreparedStructuralModalTask,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingModal(task.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "modal completion differs from the awaited random-response producer",
            ));
        }
        self.modal = Some(import_structural_modal_result(
            source,
            response,
            task,
            self.preparation.inventory_limits,
        )?);
        self.phase = StudyPhase::ReadyForBasisState;
        Ok(())
    }

    pub fn prepare_basis_state(
        &mut self,
    ) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
        if self.phase != StudyPhase::ReadyForBasisState {
            return Err(invalid(
                "random-response study is not ready to publish its modal basis",
            ));
        }
        let modal = self
            .modal
            .as_ref()
            .ok_or_else(|| invalid("completed modal result is absent"))?;
        let task = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: modal.host.clone(),
            source_result: modal.objects.clone(),
            source_profiles: modal.state_publication_profile_objects.clone(),
            source: SemanticStatePublicationSource::ModalBasis,
            declaration: self.modal_workload.modal_basis_output()?.clone(),
            topology_digest: self.topology_digest(),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
        })?;
        self.phase = StudyPhase::AwaitingBasisState(task.host.workload.identity.digest());
        Ok(task)
    }

    pub fn complete_basis_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        task: &PreparedSemanticStatePublication,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingBasisState(task.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "modal-basis publication differs from the awaited random-response transition",
            ));
        }
        let state = import_semantic_state_completion(
            source,
            response,
            task,
            self.preparation.inventory_limits,
        )?;
        let transition = bind_upstream_state_transition(
            &self.response_template.model,
            &self.response_template.step_id,
            &state.state,
            StateUse::ModalBasis,
            self.topology_digest(),
        )?;
        self.response_workload = Some(self.response_template.clone().bind(transition)?);
        self.basis_state = Some(state);
        self.phase = StudyPhase::ReadyForIntervals;
        Ok(())
    }

    pub fn prepare_intervals(
        &mut self,
    ) -> AnalysisBridgeResult<Vec<PreparedStructuralRandomResponseIntervalTask>> {
        if self.phase != StudyPhase::ReadyForIntervals {
            return Err(invalid(
                "random-response study is not ready for interval fan-out",
            ));
        }
        let workload = self.bound_workload()?;
        let state = self
            .basis_state
            .as_ref()
            .ok_or_else(|| invalid("published modal basis is absent"))?;
        let tasks =
            prepare_structural_random_response_fanout(StructuralRandomResponseFanoutPreparation {
                local_execution: self.preparation.local_execution,
                authored_study_digest: self.preparation.authored_study_digest,
                frozen_project_digest: self.preparation.frozen_project_digest,
                program_revision: self.preparation.program_revision.clone(),
                artifact_access: self.preparation.artifact_access.clone(),
                workload: workload.clone(),
                mesh: self.preparation.mesh.clone(),
                modal_state: state.state.clone(),
                chunk_bytes: self.preparation.chunk_bytes,
                inventory_limits: self.preparation.inventory_limits,
            })?;
        self.phase = StudyPhase::AwaitingIntervals(
            tasks
                .iter()
                .map(|task| task.host.workload.identity.digest())
                .collect(),
        );
        Ok(tasks)
    }

    pub fn complete_intervals(
        &mut self,
        source: &impl CacheImport,
        tasks: Vec<PreparedStructuralRandomResponseIntervalTask>,
        mut responses: BTreeMap<Digest, ProgramExecutionResponse>,
    ) -> AnalysisBridgeResult<()> {
        let StudyPhase::AwaitingIntervals(expected) = &self.phase else {
            return Err(invalid(
                "random-response study is not awaiting interval results",
            ));
        };
        let supplied = tasks
            .iter()
            .map(|task| task.host.workload.identity.digest())
            .collect::<BTreeSet<_>>();
        if &supplied != expected || responses.keys().copied().collect::<BTreeSet<_>>() != *expected
        {
            return Err(AnalysisBridgeError::Identity(
                "random-response interval completions do not exactly cover the prepared fan-out",
            ));
        }
        let mut intervals = Vec::with_capacity(tasks.len());
        for task in tasks {
            let identity = task.host.workload.identity.digest();
            let response = responses
                .remove(&identity)
                .ok_or_else(|| invalid("prepared random-response interval response is absent"))?;
            intervals.push(import_structural_random_response_interval(
                source,
                response,
                &task,
                self.preparation.inventory_limits,
            )?);
        }
        intervals.sort_by_key(|entry| entry.result.request().task_ordinal());
        self.intervals = intervals;
        self.phase = StudyPhase::ReadyForAssembly;
        Ok(())
    }

    pub fn prepare_assembly(
        &mut self,
    ) -> AnalysisBridgeResult<PreparedStructuralRandomResponseAssemblyTask> {
        if self.phase != StudyPhase::ReadyForAssembly {
            return Err(invalid(
                "random-response study is not ready for deterministic fan-in",
            ));
        }
        let workload = self.bound_workload()?;
        let state = self
            .basis_state
            .as_ref()
            .ok_or_else(|| invalid("published modal basis is absent"))?;
        let task = prepare_structural_random_response_assembly(
            StructuralRandomResponseAssemblyPreparation {
                local_execution: self.preparation.local_execution,
                authored_study_digest: self.preparation.authored_study_digest,
                frozen_project_digest: self.preparation.frozen_project_digest,
                program_revision: self.preparation.program_revision.clone(),
                artifact_access: self.preparation.artifact_access.clone(),
                workload: workload.clone(),
                mesh: self.preparation.mesh.clone(),
                modal_state: state.state.clone(),
                tasks: self
                    .intervals
                    .iter()
                    .map(|entry| entry.objects.clone())
                    .collect(),
                chunk_bytes: self.preparation.chunk_bytes,
                inventory_limits: self.preparation.inventory_limits,
            },
        )?;
        self.phase = StudyPhase::AwaitingAssembly(task.host.workload.identity.digest());
        Ok(task)
    }

    pub fn complete_assembly(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        task: &PreparedStructuralRandomResponseAssemblyTask,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingAssembly(task.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "random-response assembly differs from the awaited fan-in",
            ));
        }
        self.result = Some(import_structural_random_response_result(
            source,
            response,
            task,
            self.preparation.inventory_limits,
        )?);
        self.assembly_host = Some(task.host.clone());
        self.assembly_source_context_inputs = task
            .inputs
            .iter()
            .filter(|input| {
                matches!(
                    input.manifest.kind,
                    AnalysisNumericalObjectKind::StructuralRandomResponseWorkload
                        | AnalysisNumericalObjectKind::Mesh
                )
            })
            .cloned()
            .collect();
        // Authored outputs observe the step's recovered statistic fields before accuracy.
        let step = self.bound_workload()?.procedure_step()?.clone();
        if step.observations.is_empty() {
            self.phase = StudyPhase::ReadyForAccuracy;
        } else {
            let result = self
                .result
                .as_ref()
                .ok_or_else(|| invalid("completed random-response result is absent"))?;
            self.observations = Some(crate::StatisticalStudyObservations::new(
                crate::StatisticalObservationSource {
                    assembly_host: task.host.clone(),
                    assembly_inputs: self.assembly_source_context_inputs.clone(),
                    source_result: result.result_objects.clone(),
                    source_fields: None,
                    sequence: SequenceIdentity::from_solver_contract(
                        &self.bound_workload()?.model().sequence,
                    )?,
                    step,
                    mesh: self.preparation.mesh.clone(),
                    program_revision: self.preparation.program_revision.clone(),
                    chunk_bytes: self.preparation.chunk_bytes,
                    inventory_limits: self.preparation.inventory_limits,
                    local_execution: self.preparation.local_execution,
                },
            )?);
            self.phase = StudyPhase::Observing;
        }
        Ok(())
    }

    /// The authored observations awaiting execution after assembly, if the step has any.
    pub fn observations_mut(&mut self) -> Option<&mut crate::StatisticalStudyObservations> {
        if self.phase == StudyPhase::Observing {
            self.observations.as_mut()
        } else {
            None
        }
    }

    /// Accept completed observations; accuracy and publication then bind their catalog.
    pub fn finish_observations(&mut self) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::Observing
            || !self.observations.as_ref().is_some_and(|o| o.is_complete())
        {
            return Err(invalid("random-response observations are not complete"));
        }
        self.phase = StudyPhase::ReadyForAccuracy;
        Ok(())
    }

    pub fn prepare_accuracy(&mut self) -> AnalysisBridgeResult<PreparedAccuracy> {
        if self.phase != StudyPhase::ReadyForAccuracy {
            return Err(invalid(
                "random-response study is not ready for accuracy evaluation",
            ));
        }
        let workload = self.bound_workload()?;
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| invalid("completed random-response result is absent"))?;
        let task = prepare_accuracy(AccuracyPreparation {
            source_context_inputs: self.assembly_source_context_inputs.clone(),
            solve_host: self
                .assembly_host
                .as_ref()
                .ok_or_else(|| invalid("random-response assembly host is absent"))?
                .clone(),
            source_kind: AccuracySourceKind::StructuralRandomResponse,
            source_numerical_receipt: result.receipt_objects.clone(),
            source_observations: self.observations.as_ref().and_then(|o| o.catalog()),
            sequence: SequenceIdentity::from_solver_contract(&workload.model().sequence)?,
            step: workload.procedure_step()?.clone(),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
        })?;
        self.phase = StudyPhase::AwaitingAccuracy(task.host.workload.identity.digest());
        Ok(task)
    }

    pub fn complete_accuracy(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        task: &PreparedAccuracy,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingAccuracy(task.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "random-response accuracy differs from the awaited evaluation",
            ));
        }
        let accuracy = import_accuracy(source, response, task, self.preparation.inventory_limits)?;
        self.accuracy_host = Some(task.host.clone());
        self.accuracy = Some(accuracy);
        self.phase = StudyPhase::ReadyForStatisticsState;
        Ok(())
    }

    pub fn prepare_publication(
        &mut self,
        resource_usage: PreparedResourceUsage,
    ) -> AnalysisBridgeResult<PreparedResultPublication> {
        if self.phase != StudyPhase::ReadyForPublication {
            return Err(invalid(
                "random-response study is not ready for result publication",
            ));
        }
        let task = prepare_result_publication(ResultPublicationPreparation {
            accuracy_host: self
                .accuracy_host
                .as_ref()
                .ok_or_else(|| invalid("random-response accuracy host is absent"))?
                .clone(),
            source_accuracy: self
                .accuracy
                .as_ref()
                .ok_or_else(|| invalid("random-response accuracy is absent"))?
                .objects
                .clone(),
            source_resource_usage: resource_usage.objects.clone(),
            source_observations: self.observations.as_ref().and_then(|o| o.catalog()),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
        })?;
        self.resource_usage = Some(resource_usage);
        self.phase = StudyPhase::AwaitingPublication(task.host.workload.identity.digest());
        Ok(task)
    }

    pub fn complete_publication(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        task: &PreparedResultPublication,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingPublication(task.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "random-response publication differs from the awaited terminal receipt",
            ));
        }
        self.publication = Some(import_published_result(
            source,
            response,
            task,
            self.preparation.inventory_limits,
        )?);
        self.phase = StudyPhase::Complete;
        Ok(())
    }

    pub fn prepare_statistics_state(
        &mut self,
    ) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
        if self.phase != StudyPhase::ReadyForStatisticsState {
            return Err(invalid(
                "random-response study is not ready to publish its statistics",
            ));
        }
        let declaration = self
            .bound_workload()?
            .procedure_step()?
            .state_outputs
            .iter()
            .find(|output| output.kind == StateKind::StructuralRandomStatistics)
            .ok_or_else(|| invalid("random-response statistics declaration is absent"))?
            .clone();
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| invalid("completed random-response result is absent"))?;
        let task = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: self
                .assembly_host
                .as_ref()
                .ok_or_else(|| invalid("random-response assembly host is absent"))?
                .clone(),
            source_result: result.result_objects.clone(),
            source_profiles: result.state_publication_profile_objects.clone(),
            source: SemanticStatePublicationSource::StructuralRandomStatistics,
            declaration,
            topology_digest: self.topology_digest(),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.chunk_bytes,
            inventory_limits: self.preparation.inventory_limits,
        })?;
        self.phase = StudyPhase::AwaitingStatisticsState(task.host.workload.identity.digest());
        Ok(task)
    }

    pub fn complete_statistics_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        task: &PreparedSemanticStatePublication,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != StudyPhase::AwaitingStatisticsState(task.host.workload.identity.digest()) {
            return Err(AnalysisBridgeError::Identity(
                "random-statistics publication differs from the awaited terminal state",
            ));
        }
        self.statistics_state = Some(import_semantic_state_completion(
            source,
            response,
            task,
            self.preparation.inventory_limits,
        )?);
        self.phase = StudyPhase::ReadyForPublication;
        Ok(())
    }

    pub fn is_complete(&self) -> bool {
        self.phase == StudyPhase::Complete
    }

    pub fn result(&self) -> Option<&ImportedStructuralRandomResponseResult> {
        self.result.as_ref()
    }

    pub fn modal_result(&self) -> Option<&ImportedStructuralModalResult> {
        self.modal.as_ref()
    }

    pub fn basis_state(&self) -> Option<&ImportedSemanticState> {
        self.basis_state.as_ref()
    }

    pub fn resolved_workload(&self) -> Option<&StructuralRandomResponseWorkload> {
        self.response_workload.as_ref()
    }

    pub fn statistics_state(&self) -> Option<&ImportedSemanticState> {
        self.statistics_state.as_ref()
    }

    /// The authored-observation catalog accuracy and publication bound, if the step has outputs.
    pub fn observation_catalog(
        &self,
    ) -> Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog> {
        self.observations.as_ref().and_then(|o| o.catalog())
    }

    pub fn accuracy(&self) -> Option<&ImportedAccuracy> {
        self.accuracy.as_ref()
    }

    pub fn resource_usage(&self) -> Option<&PreparedResourceUsage> {
        self.resource_usage.as_ref()
    }

    pub fn publication(&self) -> Option<&ImportedPublishedResult> {
        self.publication.as_ref()
    }

    fn bound_workload(&self) -> AnalysisBridgeResult<&StructuralRandomResponseWorkload> {
        self.response_workload
            .as_ref()
            .ok_or_else(|| invalid("bound random-response workload is absent"))
    }

    fn topology_digest(&self) -> Digest {
        Digest::from_bytes(*self.preparation.mesh.canonical_digest.bytes())
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
