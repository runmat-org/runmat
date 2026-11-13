//! Governed lifecycle of one explicitly keyed entry of an authored acoustic frequency plan.

use crate::phase_accounting::validate_analysis_phase_accounting as validate_accounting;
use crate::AnalysisPhaseExecutionEvidence;
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, SemanticStatePublicationSource, SequenceIdentity},
    planning::resource::FeaResourcePlan,
    study::{AcousticStudyPlanningPolicy, CompiledAnalysisStudy},
};
use runmat_execution::Digest;
use runmat_execution_artifact::{cache::CacheImport, ProgramExecutionResponse};

use crate::{
    complete_result_publication_resource_usage, extend_resource_usage, import_accuracy,
    import_acoustic_harmonic_result, import_published_result, import_semantic_state_completion,
    prepare_accuracy, prepare_acoustic_harmonic_task, prepare_result_publication,
    prepare_semantic_state_publication, AccuracyPreparation, AcousticHarmonicTaskPreparation,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, ImportedAccuracy,
    ImportedAcousticHarmonicResult, ImportedPublishedResult, ImportedSemanticState,
    PlannedResourceUsageAttempt, PreparedAccuracy, PreparedAcousticHarmonicTask,
    PreparedResourceUsage, PreparedResultPublication, PreparedSemanticStatePublication,
    ResultPublicationPreparation, SemanticStatePublicationPreparation,
};

use super::completion::{AcousticStudyResultClosure, AcousticStudyResultPhase};
use super::AcousticStudyPreparation;

mod observations;
mod task;
pub(super) use crate::sampled_study_observations::SampledStudyObservations as AcousticObservations;

/// One exact authored numerical entry; states remain scoped to this coordinate and declaration.
pub struct CompletedAcousticStudyEntry {
    pub entry: runmat_analysis_fea::frequency::FrequencyResponsePlanEntry,
    pub workload: runmat_analysis_fea::execution::AcousticHarmonicWorkload,
    pub result: ImportedAcousticHarmonicResult,
    /// Retained native pressure and particle velocity at this frequency.
    pub fields: crate::ImportedRecoveredFields,
    /// The complete transport closure of `fields`, for publication catalogs.
    pub field_objects: crate::PreparedAnalysisObjects,
    pub observations: Option<crate::ImportedObservations>,
    pub observation_objects: Option<crate::PreparedAnalysisObjects>,
    pub observation_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    pub accuracy: ImportedAccuracy,
    pub states: Vec<ImportedSemanticState>,
    pub baseline_usage: PreparedResourceUsage,
    pub terminal_usage: PreparedResourceUsage,
    pub publication: ImportedPublishedResult,
    pub(super) closures: Vec<AcousticStudyResultClosure>,
}
impl CompletedAcousticStudyEntry {
    pub fn result_closures(&self) -> &[AcousticStudyResultClosure] {
        &self.closures
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AcousticStudyPhase {
    ReadyForSolve,
    AwaitingSolve,
    ReadyForFieldRecovery,
    AwaitingFieldRecovery,
    ReadyForObservationFanout,
    ReadyForObservationPartition { ordinal: usize },
    AwaitingObservationPartition { ordinal: usize },
    ReadyForObservationReduction { ordinal: usize },
    AwaitingObservationReduction { ordinal: usize },
    ReadyForObservationAssembly,
    AwaitingObservationAssembly,
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
pub struct AcousticStudyEntryCoordinator {
    closures: Vec<AcousticStudyResultClosure>,
    preparation: std::sync::Arc<AcousticStudyPreparation>,
    entry: runmat_analysis_fea::frequency::FrequencyResponsePlanEntry,
    attempts: Vec<PlannedResourceUsageAttempt>,
    definition_digest: Digest,
    phase: AcousticStudyPhase,
    solve_task: Option<PreparedAcousticHarmonicTask>,
    recovery_task: Option<crate::PreparedAcousticFieldRecovery>,
    fields: Option<crate::ImportedRecoveredFields>,
    field_objects: Option<crate::PreparedAnalysisObjects>,
    observation_objects: Option<crate::PreparedAnalysisObjects>,
    observation_state: AcousticObservations,
    accuracy_task: Option<PreparedAccuracy>,
    state_task: Option<PreparedSemanticStatePublication>,
    publication_task: Option<PreparedResultPublication>,
    result: Option<ImportedAcousticHarmonicResult>,
    accuracy: Option<ImportedAccuracy>,
    states: Vec<ImportedSemanticState>,
    usage: Option<PreparedResourceUsage>,
    baseline: Option<PreparedResourceUsage>,
    terminal: Option<PreparedResourceUsage>,
    publication: Option<ImportedPublishedResult>,
}

impl AcousticStudyEntryCoordinator {
    pub(super) fn new(
        preparation: std::sync::Arc<AcousticStudyPreparation>,
        entry: runmat_analysis_fea::frequency::FrequencyResponsePlanEntry,
    ) -> AnalysisBridgeResult<Self> {
        preparation
            .product
            .plan
            .frequency_plan()
            .validate_entry(entry)
            .map_err(|error| invalid(error.to_string()))?;
        let planned = preparation
            .product
            .plan
            .frequencies()
            .get(entry.ordinal() as usize)
            .ok_or_else(|| invalid("missing planned acoustic entry"))?;
        if planned.entry() != entry {
            return Err(invalid("acoustic entry differs from exact authored plan"));
        }
        let definition_digest = preparation
            .product
            .compiled
            .definition_digest()
            .map_err(|error| invalid(error.to_string()))?;
        Ok(Self {
            entry,
            attempts: Vec::new(),
            closures: Vec::new(),
            preparation,
            definition_digest,
            phase: AcousticStudyPhase::ReadyForSolve,
            solve_task: None,
            recovery_task: None,
            fields: None,
            field_objects: None,
            observation_objects: None,
            observation_state: AcousticObservations::default(),
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
        })
    }

    pub fn prepare_solve(&mut self) -> AnalysisBridgeResult<&PreparedAcousticHarmonicTask> {
        self.require(AcousticStudyPhase::ReadyForSolve)?;
        let task = prepare_acoustic_harmonic_task(AcousticHarmonicTaskPreparation {
            local_execution: self.preparation.local_execution,
            authored_study_digest: self.definition_digest,
            frozen_project_digest: self.preparation.frozen_project_digest,
            program_revision: self.preparation.program_revision.clone(),
            artifact_access: self.preparation.artifact_access.clone(),
            workload: self.workload().clone(),
            mesh: self.preparation.mesh.clone(),
            output_policy: self.preparation.output_policy,
            required_capabilities: self.preparation.required_capabilities.clone(),
        })?;
        self.solve_task = Some(task);
        self.phase = AcousticStudyPhase::AwaitingSolve;
        Ok(self.solve_task.as_ref().expect("just prepared"))
    }

    /// The caller must preserve the scheduler's response/accounting association.
    pub fn complete_solve(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(AcousticStudyPhase::AwaitingSolve)?;
        let task = self.solve_task.as_ref().expect("awaited solve");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        let closure = AcousticStudyResultClosure::new(
            AcousticStudyResultPhase::Solve,
            &task.host,
            &response,
            &accounting,
        );
        let result = import_acoustic_harmonic_result(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let graph = accounting.resource_graph(&task.resource_plan)?;
        let usage = crate::prepare_planned_resource_graph_usage(
            &task.resource_plan,
            task.host.workload.identity.digest(),
            result.result.source_result_identity()?.digest(),
            &graph,
            &task.host.artifact_access,
            self.preparation.output_policy.chunk_bytes(),
            self.preparation.output_policy.inventory_limits(),
        )?;
        self.result = Some(result);
        self.usage = Some(usage);
        self.closures.push(closure);
        self.attempts.extend(task_attempts);
        self.phase = AcousticStudyPhase::ReadyForFieldRecovery;
        Ok(())
    }

    pub fn prepare_accuracy(&mut self) -> AnalysisBridgeResult<&PreparedAccuracy> {
        self.require(AcousticStudyPhase::ReadyForAccuracy)?;
        let result = self.result.as_ref().expect("completed solve");
        let workload = self.workload();
        let task =
            prepare_accuracy(AccuracyPreparation {
                solve_host: self.solve_task.as_ref().expect("completed solve").host.clone(),
                source_kind: AccuracySourceKind::AcousticHarmonic,
                source_numerical_receipt: result.numerical_receipt_objects.clone(),
                source_observations: self.observation_state.catalog.clone(),
                source_context_inputs: self
                    .solve_task
                    .as_ref()
                    .expect("completed solve")
                    .inputs
                    .iter()
                    .filter(|input| {
                        matches!(input.manifest.kind,
                    runmat_analysis_fea::execution::AnalysisNumericalObjectKind::AcousticHarmonicWorkload
                    | runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Mesh)
                    })
                    .cloned()
                    .chain(std::iter::once(result.objects.clone()))
                    .collect(),
                sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
                step: self.preparation.selected_step().clone(),
                program_revision: self.preparation.program_revision.clone(),
                chunk_bytes: self.preparation.output_policy.chunk_bytes(),
                inventory_limits: self.preparation.output_policy.inventory_limits(),
            })?;
        self.accuracy_task = Some(task);
        self.phase = AcousticStudyPhase::AwaitingAccuracy;
        Ok(self.accuracy_task.as_ref().expect("just prepared"))
    }

    pub fn complete_accuracy(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(AcousticStudyPhase::AwaitingAccuracy)?;
        let task = self.accuracy_task.as_ref().expect("awaited accuracy");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        let closure = AcousticStudyResultClosure::new(
            AcousticStudyResultPhase::Accuracy,
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
        self.attempts.extend(task_attempts);
        self.usage = Some(usage);
        self.advance_state();
        Ok(())
    }

    /// Publish the exact complete complex state declared by the standalone acoustic step.
    pub fn prepare_next_state(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedSemanticStatePublication> {
        let ordinal = self.states.len();
        self.require(AcousticStudyPhase::ReadyForState { ordinal })?;
        let result = self.result.as_ref().expect("completed solve");
        let declaration = self.preparation.selected_step().state_outputs[ordinal].clone();
        let task = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: self
                .solve_task
                .as_ref()
                .expect("completed solve")
                .host
                .clone(),
            source_result: result.objects.clone(),
            source_profiles: result.source_profiles.clone(),
            source: SemanticStatePublicationSource::AcousticNativeFields,
            declaration,
            topology_digest: Digest::from_bytes(*self.preparation.mesh.canonical_digest.bytes()),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.output_policy.chunk_bytes(),
            inventory_limits: self.preparation.output_policy.inventory_limits(),
        })?;
        self.state_task = Some(task);
        self.phase = AcousticStudyPhase::AwaitingState { ordinal };
        Ok(self.state_task.as_ref().expect("just prepared"))
    }

    pub fn complete_next_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(AcousticStudyPhase::AwaitingState {
            ordinal: self.states.len(),
        })?;
        let task = self.state_task.as_ref().expect("awaited state");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        let closure = AcousticStudyResultClosure::new(
            AcousticStudyResultPhase::State {
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
        self.attempts.extend(task_attempts);
        self.usage = Some(usage);
        self.state_task = None;
        self.advance_state();
        Ok(())
    }

    /// Freeze the immutable solve/accuracy/all-states baseline named by the receipt.
    pub fn freeze_baseline(&mut self) -> AnalysisBridgeResult<&PreparedResourceUsage> {
        self.require(AcousticStudyPhase::ReadyForBaseline)?;
        self.baseline = self.usage.clone();
        self.phase = AcousticStudyPhase::ReadyForPublication;
        Ok(self.baseline.as_ref().expect("completed usage"))
    }

    pub fn prepare_publication(&mut self) -> AnalysisBridgeResult<&PreparedResultPublication> {
        self.require(AcousticStudyPhase::ReadyForPublication)?;
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
            source_observations: self.observation_state.catalog.clone(),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.output_policy.chunk_bytes(),
            inventory_limits: self.preparation.output_policy.inventory_limits(),
        })?;
        self.publication_task = Some(task);
        self.phase = AcousticStudyPhase::AwaitingPublication;
        Ok(self.publication_task.as_ref().expect("just prepared"))
    }

    /// Commit receipt and terminal accounting atomically; neither is visible on failure.
    pub fn complete_publication(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(AcousticStudyPhase::AwaitingPublication)?;
        let task = self.publication_task.as_ref().expect("awaited publication");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let task_attempts = accounting.resource_graph(&task.resource_plan)?.attempts;
        let closure = AcousticStudyResultClosure::new(
            AcousticStudyResultPhase::Publication,
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
        self.attempts.extend(task_attempts);
        self.terminal = Some(terminal);
        self.phase = AcousticStudyPhase::Complete;
        Ok(())
    }

    pub fn phase(&self) -> AcousticStudyPhase {
        self.phase
    }

    pub fn into_completed(self) -> AnalysisBridgeResult<CompletedAcousticStudyEntry> {
        self.require(AcousticStudyPhase::Complete)?;
        let workload = self.workload().clone();
        Ok(CompletedAcousticStudyEntry {
            entry: self.entry,
            closures: self.closures,
            workload,
            result: self.result.expect("completed solve"),
            fields: self.fields.expect("completed retained fields"),
            field_objects: self
                .field_objects
                .expect("completed retained field objects"),
            observations: self.observation_state.completed,
            observation_objects: self.observation_objects,
            observation_catalog: self.observation_state.catalog,
            accuracy: self.accuracy.expect("completed accuracy"),
            states: self.states,
            baseline_usage: self.baseline.expect("frozen baseline"),
            terminal_usage: self.terminal.expect("completed terminal accounting"),
            publication: self.publication.expect("completed publication"),
        })
    }
    pub fn is_complete(&self) -> bool {
        self.phase == AcousticStudyPhase::Complete
    }
    pub fn policy(&self) -> AcousticStudyPlanningPolicy {
        self.preparation.policy
    }
    pub fn selected_step(&self) -> &runmat_analysis_core::ProcedureStep {
        self.preparation.selected_step()
    }
    pub fn compiled_study(&self) -> &CompiledAnalysisStudy {
        &self.preparation.product.compiled
    }
    pub fn result(&self) -> Option<&ImportedAcousticHarmonicResult> {
        self.result.as_ref()
    }
    pub fn fields(&self) -> Option<&crate::ImportedRecoveredFields> {
        self.fields.as_ref()
    }
    pub fn observations(&self) -> Option<&crate::ImportedObservations> {
        self.observation_state.completed.as_ref()
    }
    pub fn accuracy(&self) -> Option<&ImportedAccuracy> {
        self.accuracy.as_ref()
    }
    pub fn states(&self) -> &[ImportedSemanticState] {
        &self.states
    }
    pub(super) fn resource_attempts(&self) -> &[PlannedResourceUsageAttempt] {
        &self.attempts
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

    pub const fn entry(&self) -> runmat_analysis_fea::frequency::FrequencyResponsePlanEntry {
        self.entry
    }
    pub fn workload(&self) -> &runmat_analysis_fea::execution::AcousticHarmonicWorkload {
        self.preparation.product.plan.frequencies()[self.entry.ordinal() as usize].workload()
    }

    fn require(&self, expected: AcousticStudyPhase) -> AnalysisBridgeResult<()> {
        if self.phase != expected {
            return Err(invalid(format!(
                "acoustic study phase {:?} does not permit {:?}",
                self.phase, expected
            )));
        }
        Ok(())
    }

    fn advance_state(&mut self) {
        self.phase = if self.states.len() < self.preparation.selected_step().state_outputs.len() {
            AcousticStudyPhase::ReadyForState {
                ordinal: self.states.len(),
            }
        } else {
            AcousticStudyPhase::ReadyForBaseline
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
