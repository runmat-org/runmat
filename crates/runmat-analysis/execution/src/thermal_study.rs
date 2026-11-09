//! Placement-neutral orchestration of a fresh authored thermal step.
//!
//! Prepared tasks remain owned here: callers execute their immutable requests and return the
//! associated scheduler response/accounting pair. Failed imports never advance the lifecycle.
use crate::phase_accounting::validate_analysis_phase_accounting as validate_accounting;
use std::collections::BTreeSet;

use crate::AnalysisPhaseExecutionEvidence;
use runmat_analysis_core::AuthoredOutput;
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, SemanticStatePublicationSource, SequenceIdentity},
    planning::resource::FeaResourcePlan,
    study::{CompiledAnalysisStudy, ThermalStudyProduct},
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, ArtifactAccess, ProgramExecutionResponse};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    complete_result_publication_resource_usage, extend_resource_usage, import_accuracy,
    import_published_result, import_semantic_state_completion, import_thermal_completion,
    prepare_accuracy, prepare_result_publication, prepare_semantic_state_publication,
    prepare_thermal_task, AccuracyPreparation, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisOutputPolicy, ImportedAccuracy,
    ImportedPublishedResult, ImportedSemanticState, ImportedThermalResult, PreparedAccuracy,
    PreparedResourceUsage, PreparedResultPublication, PreparedSemanticStatePublication,
    PreparedThermalTask, ResultPublicationPreparation, SemanticStatePublicationPreparation,
    ThermalTaskPreparation,
};

mod completion;
mod observations;
mod root;
mod scope;
pub use completion::{
    import_thermal_study_completion, ThermalStudyResultClosure, ThermalStudyResultPhase,
};
pub use root::{import_thermal_study_root, prepare_thermal_study_root, PreparedThermalStudyRoot};
pub use scope::ThermalStudyScope;
#[cfg(test)]
pub(crate) mod tests;
use crate::sampled_history_observations::{CompletedSampledObservations, SampledObservationTask};

/// Owned terminal artifacts for the declared scope, not necessarily the whole authored study.
/// Consuming the coordinator does not clone numerical storage.
#[derive(Clone, Debug, PartialEq)]
pub struct CompletedThermalStudy {
    replay: completion::ThermalStudyReplayContext,
    closures: Vec<ThermalStudyResultClosure>,
    pub observations: CompletedSampledObservations,
    pub compiled: CompiledAnalysisStudy,
    pub workload: runmat_analysis_fea::execution::ThermalWorkload,
    pub result: ImportedThermalResult,
    pub accuracy: ImportedAccuracy,
    pub states: Vec<ImportedSemanticState>,
    pub baseline_usage: PreparedResourceUsage,
    pub terminal_usage: PreparedResourceUsage,
    pub publication: ImportedPublishedResult,
}

pub struct ThermalStudyPreparation {
    /// Effective execution capacity, carried separately from the compiled study definition.
    pub maximum_resident_bytes: u64,
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub scope: ThermalStudyScope,
    pub field_source: Option<runmat_analysis_fea::execution::RetainedJouleHeatSource>,
    pub product: ThermalStudyProduct,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThermalStudyPhase {
    ReadyForSolve,
    AwaitingSolve,
    ReadyForObservations,
    AwaitingObservation,
    ReadyForAccuracy,
    AwaitingAccuracy,
    ReadyForState { ordinal: usize },
    AwaitingState { ordinal: usize },
    ReadyForBaseline,
    ReadyForPublication,
    AwaitingPublication,
    Complete,
}

/// No numerical kernels or placement decisions live in this coordinator.
pub struct ThermalStudyCoordinator {
    closures: Vec<ThermalStudyResultClosure>,
    preparation: ThermalStudyPreparation,
    definition_digest: Digest,
    phase: ThermalStudyPhase,
    solve_task: Option<PreparedThermalTask>,
    accuracy_task: Option<PreparedAccuracy>,
    state_task: Option<PreparedSemanticStatePublication>,
    publication_task: Option<PreparedResultPublication>,
    result: Option<ImportedThermalResult>,
    accuracy: Option<ImportedAccuracy>,
    states: Vec<ImportedSemanticState>,
    usage: Option<PreparedResourceUsage>,
    baseline: Option<PreparedResourceUsage>,
    terminal: Option<PreparedResourceUsage>,
    publication: Option<ImportedPublishedResult>,
    observations: crate::sampled_history_observations::SampledHistoryWorkflow,
}

impl ThermalStudyCoordinator {
    pub fn new(preparation: ThermalStudyPreparation) -> AnalysisBridgeResult<Self> {
        preparation
            .product
            .compiled
            .validate_against(&preparation.mesh)
            .map_err(|error| invalid(error.to_string()))?;
        preparation.validate_workload()?;
        let definition_digest = preparation
            .product
            .compiled
            .definition_digest()
            .map_err(|error| invalid(error.to_string()))?;
        Ok(Self {
            closures: Vec::new(),
            preparation,
            definition_digest,
            phase: ThermalStudyPhase::ReadyForSolve,
            solve_task: None,
            accuracy_task: None,
            state_task: None,
            publication_task: None,
            result: None,
            accuracy: None,
            states: Vec::new(),
            usage: None,
            baseline: None,
            terminal: None,
            publication: None,
            observations: crate::sampled_history_observations::SampledHistoryWorkflow::default(),
        })
    }

    pub fn prepare_solve(&mut self) -> AnalysisBridgeResult<&PreparedThermalTask> {
        self.require(ThermalStudyPhase::ReadyForSolve)?;
        let task = prepare_thermal_task(ThermalTaskPreparation {
            maximum_resident_bytes: self.preparation.maximum_resident_bytes,
            local_execution: self.preparation.local_execution,
            field_source: self.preparation.field_source.clone(),
            authored_study_digest: self.definition_digest,
            frozen_project_digest: self.preparation.frozen_project_digest,
            program_revision: self.preparation.program_revision.clone(),
            artifact_access: self.preparation.artifact_access.clone(),
            workload: self.preparation.product.workload.clone(),
            mesh: self.preparation.mesh.clone(),
            output_policy: self.preparation.output_policy,
            required_capabilities: self.preparation.required_capabilities.clone(),
        })?;
        self.solve_task = Some(task);
        self.phase = ThermalStudyPhase::AwaitingSolve;
        Ok(self.solve_task.as_ref().expect("just prepared"))
    }

    /// The caller must preserve the scheduler's response/accounting association.
    pub fn complete_solve(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(ThermalStudyPhase::AwaitingSolve)?;
        let task = self.solve_task.as_ref().expect("awaited solve");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = ThermalStudyResultClosure::new(
            ThermalStudyResultPhase::Solve,
            &task.host,
            &response,
            &accounting,
        );
        let completion = import_thermal_completion(
            source,
            response,
            accounting,
            task,
            self.preparation.output_policy,
        )?;
        self.result = Some(completion.result);
        self.usage = Some(completion.solve_usage);
        self.closures.push(closure);
        self.phase = if self.preparation.selected_step().observations.is_empty() {
            ThermalStudyPhase::ReadyForAccuracy
        } else {
            ThermalStudyPhase::ReadyForObservations
        };
        Ok(())
    }

    /// Prepare exactly one owned observation phase. None means exact authored coverage is closed.
    pub fn prepare_next_observation(
        &mut self,
        source: &impl CacheImport,
    ) -> AnalysisBridgeResult<Option<SampledObservationTask<'_>>> {
        self.require(ThermalStudyPhase::ReadyForObservations)?;
        let context = observations::Context {
            setup: &self.preparation,
            solve: self.solve_task.as_ref().expect("completed solve"),
            result: self.result.as_ref().expect("completed result"),
        };
        let task = self.observations.prepare(source, &context)?;
        self.phase = if task.is_some() {
            ThermalStudyPhase::AwaitingObservation
        } else {
            ThermalStudyPhase::ReadyForAccuracy
        };
        Ok(task)
    }

    pub fn complete_observation(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(ThermalStudyPhase::AwaitingObservation)?;
        let task = self.observations.pending()?;
        validate_accounting(task.host, task.resource_plan, &response, &accounting)?;
        let closure = ThermalStudyResultClosure::new(
            ThermalStudyResultPhase::Observation,
            task.host,
            &response,
            &accounting,
        );
        let usage = self.extend_usage(task.resource_plan, accounting)?;
        let context = observations::Context {
            setup: &self.preparation,
            solve: self.solve_task.as_ref().expect("completed solve"),
            result: self.result.as_ref().expect("completed result"),
        };
        self.observations.complete(source, response, &context)?;
        self.closures.push(closure);
        self.usage = Some(usage);
        self.phase = ThermalStudyPhase::ReadyForObservations;
        Ok(())
    }

    pub fn prepare_accuracy(&mut self) -> AnalysisBridgeResult<&PreparedAccuracy> {
        self.require(ThermalStudyPhase::ReadyForAccuracy)?;
        let result = self.result.as_ref().expect("completed solve");
        let workload = &self.preparation.product.workload;
        let task =
            prepare_accuracy(AccuracyPreparation {
                solve_host: result.host.clone(),
                source_kind: AccuracySourceKind::Thermal,
                source_numerical_receipt: result.numerical_receipt_objects.clone(),
                source_observations: self.observations.completed().catalog.clone(),
                source_context_inputs: self
                    .solve_task
                    .as_ref()
                    .expect("completed solve")
                    .inputs
                    .iter()
                    .filter(|input| {
                        matches!(input.manifest.kind,
                    runmat_analysis_fea::execution::AnalysisNumericalObjectKind::ThermalWorkload
                    | runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Mesh)
                    })
                    .cloned()
                    .collect(),
                sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
                step: self.preparation.selected_step().clone(),
                program_revision: self.preparation.program_revision.clone(),
                chunk_bytes: self.preparation.output_policy.chunk_bytes(),
                inventory_limits: self.preparation.output_policy.inventory_limits(),
            })?;
        self.accuracy_task = Some(task);
        self.phase = ThermalStudyPhase::AwaitingAccuracy;
        Ok(self.accuracy_task.as_ref().expect("just prepared"))
    }

    pub fn complete_accuracy(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(ThermalStudyPhase::AwaitingAccuracy)?;
        let task = self.accuracy_task.as_ref().expect("awaited accuracy");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = ThermalStudyResultClosure::new(
            ThermalStudyResultPhase::Accuracy,
            &task.host,
            &response,
            &accounting,
        );
        let accuracy = import_accuracy(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        self.accuracy = Some(accuracy);
        self.closures.push(closure);
        self.usage = Some(usage);
        self.advance_state();
        Ok(())
    }

    /// Publish every declaration, in authored order, including multiple thermal states.
    pub fn prepare_next_state(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedSemanticStatePublication> {
        let ordinal = self.states.len();
        self.require(ThermalStudyPhase::ReadyForState { ordinal })?;
        let result = self.result.as_ref().expect("completed solve");
        let declaration = self.preparation.selected_step().state_outputs[ordinal].clone();
        let task = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: result.host.clone(),
            source_result: result.objects.clone(),
            source_profiles: result.state_publication_profile_objects.clone(),
            source: SemanticStatePublicationSource::ThermalPrimaryField,
            declaration,
            topology_digest: result.manifest.topology_digest,
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.output_policy.chunk_bytes(),
            inventory_limits: self.preparation.output_policy.inventory_limits(),
        })?;
        self.state_task = Some(task);
        self.phase = ThermalStudyPhase::AwaitingState { ordinal };
        Ok(self.state_task.as_ref().expect("just prepared"))
    }

    pub fn complete_next_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(ThermalStudyPhase::AwaitingState {
            ordinal: self.states.len(),
        })?;
        let task = self.state_task.as_ref().expect("awaited state");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = ThermalStudyResultClosure::new(
            ThermalStudyResultPhase::State {
                ordinal: self.states.len(),
            },
            &task.host,
            &response,
            &accounting,
        );
        let state = import_semantic_state_completion(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        self.states.push(state);
        self.closures.push(closure);
        self.usage = Some(usage);
        self.state_task = None;
        self.advance_state();
        Ok(())
    }

    /// Freeze the immutable solve/accuracy/all-states baseline named by the receipt.
    pub fn freeze_baseline(&mut self) -> AnalysisBridgeResult<&PreparedResourceUsage> {
        self.require(ThermalStudyPhase::ReadyForBaseline)?;
        self.baseline = self.usage.clone();
        self.phase = ThermalStudyPhase::ReadyForPublication;
        Ok(self.baseline.as_ref().expect("completed usage"))
    }

    pub fn prepare_publication(&mut self) -> AnalysisBridgeResult<&PreparedResultPublication> {
        self.require(ThermalStudyPhase::ReadyForPublication)?;
        let task = prepare_result_publication(ResultPublicationPreparation {
            accuracy_host: self
                .accuracy_task
                .as_ref()
                .expect("completed accuracy task")
                .host
                .clone(),
            source_accuracy: self
                .accuracy
                .as_ref()
                .expect("completed accuracy")
                .objects
                .clone(),
            source_resource_usage: self
                .baseline
                .as_ref()
                .expect("frozen baseline")
                .objects
                .clone(),
            source_observations: self.observations.completed().catalog.clone(),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.output_policy.chunk_bytes(),
            inventory_limits: self.preparation.output_policy.inventory_limits(),
        })?;
        self.publication_task = Some(task);
        self.phase = ThermalStudyPhase::AwaitingPublication;
        Ok(self.publication_task.as_ref().expect("just prepared"))
    }

    /// Commit receipt and terminal accounting atomically; neither is visible on failure.
    pub fn complete_publication(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(ThermalStudyPhase::AwaitingPublication)?;
        let task = self.publication_task.as_ref().expect("awaited publication");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = ThermalStudyResultClosure::new(
            ThermalStudyResultPhase::Publication,
            &task.host,
            &response,
            &accounting,
        );
        let publication = import_published_result(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let terminal = complete_result_publication_resource_usage(
            task,
            &publication,
            self.baseline.as_ref().expect("frozen baseline"),
            &accounting,
            self.preparation.output_policy,
        )?;
        self.publication = Some(publication);
        self.closures.push(closure);
        self.terminal = Some(terminal);
        self.phase = ThermalStudyPhase::Complete;
        Ok(())
    }

    pub fn phase(&self) -> ThermalStudyPhase {
        self.phase
    }

    pub fn into_completed(self) -> AnalysisBridgeResult<CompletedThermalStudy> {
        self.require(ThermalStudyPhase::Complete)?;
        Ok(CompletedThermalStudy {
            replay: completion::ThermalStudyReplayContext {
                maximum_resident_bytes: self.preparation.maximum_resident_bytes,
                local_execution: self.preparation.local_execution,
                field_source: self.preparation.field_source,
                scope: self.preparation.scope,
                mesh: self.preparation.mesh,
                frozen_project_digest: self.preparation.frozen_project_digest,
                artifact_access: self.preparation.artifact_access,
                program_revision: self.preparation.program_revision,
                output_policy: self.preparation.output_policy,
                required_capabilities: self.preparation.required_capabilities,
            },
            closures: self.closures,
            observations: self.observations.into_completed(),
            compiled: self.preparation.product.compiled,
            workload: self.preparation.product.workload,
            result: self.result.expect("completed solve"),
            accuracy: self.accuracy.expect("completed accuracy"),
            states: self.states,
            baseline_usage: self.baseline.expect("frozen baseline"),
            terminal_usage: self.terminal.expect("completed terminal accounting"),
            publication: self.publication.expect("completed publication"),
        })
    }
    pub fn is_complete(&self) -> bool {
        self.phase == ThermalStudyPhase::Complete
    }
    pub fn scope(&self) -> ThermalStudyScope {
        self.preparation.scope
    }
    pub fn selected_step(&self) -> &runmat_analysis_core::ProcedureStep {
        self.preparation.selected_step()
    }
    pub fn compiled_study(&self) -> &CompiledAnalysisStudy {
        &self.preparation.product.compiled
    }
    /// Retain authored aliases, field/history selection and reductions for result access.
    pub fn authored_outputs(&self) -> &[AuthoredOutput] {
        &self.preparation.product.compiled.outputs
    }
    pub fn result(&self) -> Option<&ImportedThermalResult> {
        self.result.as_ref()
    }
    pub fn accuracy(&self) -> Option<&ImportedAccuracy> {
        self.accuracy.as_ref()
    }
    pub fn observations(&self) -> &CompletedSampledObservations {
        self.observations.completed()
    }
    pub fn states(&self) -> &[ImportedSemanticState] {
        &self.states
    }
    pub fn baseline_usage(&self) -> Option<&PreparedResourceUsage> {
        self.baseline.as_ref()
    }
    pub fn terminal_usage(&self) -> Option<&PreparedResourceUsage> {
        self.terminal.as_ref()
    }
    pub fn publication(&self) -> Option<&ImportedPublishedResult> {
        self.publication.as_ref()
    }

    fn require(&self, expected: ThermalStudyPhase) -> AnalysisBridgeResult<()> {
        if self.phase != expected {
            return Err(invalid(format!(
                "thermal study phase {:?} does not permit {:?}",
                self.phase, expected
            )));
        }
        Ok(())
    }

    fn advance_state(&mut self) {
        self.phase = if self.states.len() < self.preparation.selected_step().state_outputs.len() {
            ThermalStudyPhase::ReadyForState {
                ordinal: self.states.len(),
            }
        } else {
            ThermalStudyPhase::ReadyForBaseline
        };
    }

    fn extend_usage(
        &self,
        plan: &FeaResourcePlan,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<PreparedResourceUsage> {
        let graph = accounting.resource_graph(plan)?;
        extend_resource_usage(
            self.usage.as_ref().expect("completed solve usage"),
            &graph,
            &self.preparation.artifact_access,
            self.preparation.output_policy,
        )
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
