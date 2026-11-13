//! Shared placement-neutral terminal authority for fresh standalone physics studies.
//! Solve adapters supply original authenticated sources; this engine never solves or recovers.
use crate::phase_accounting::validate_analysis_phase_accounting as validate_accounting;
use crate::AnalysisPhaseExecutionEvidence;
use crate::{
    complete_result_publication_resource_usage, extend_resource_usage, import_accuracy,
    import_published_result, import_semantic_state_completion, prepare_accuracy,
    prepare_result_publication, prepare_semantic_state_publication, AccuracyPreparation,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, AnalysisOutputPolicy,
    ImportedAccuracy, ImportedPublishedResult, ImportedSemanticState, PreparedAccuracy,
    PreparedAnalysisObjects, PreparedResourceUsage, PreparedResultPublication,
    PreparedSemanticStatePublication, ResultPublicationPreparation,
    SemanticStatePublicationPreparation,
};
use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, SemanticStatePublicationSource, SequenceIdentity},
    planning::resource::FeaResourcePlan,
    results::observation::ObservationPublicationCatalog,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, ProgramExecutionResponse};
mod storage;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SteadyStudyPhase {
    ReadyForAccuracy,
    AwaitingAccuracy,
    ReadyForState { ordinal: usize },
    AwaitingState { ordinal: usize },
    ReadyForBaseline,
    ReadyForPublication,
    AwaitingPublication,
    Complete,
}

pub(crate) struct SteadyStudySource {
    pub host: AnalysisHostWorkload,
    pub result: PreparedAnalysisObjects,
    pub state: TerminalStatePublicationSource,
    pub receipt: PreparedAnalysisObjects,
    pub context_inputs: Vec<PreparedAnalysisObjects>,
    pub accuracy_kind: AccuracySourceKind,
    pub sequence: SequenceIdentity,
    pub step: ProcedureStep,
    pub observations: Option<ObservationPublicationCatalog>,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
}

/// The adapter retains original source authority, never synthetic state profiles.
pub(crate) enum TerminalStatePublicationSource {
    Profiles {
        profiles: Box<PreparedAnalysisObjects>,
        kind: SemanticStatePublicationSource,
        topology: Digest,
    },
    FluidOriginal,
    NonlinearOriginal,
}

pub(crate) struct SteadyStudyTail {
    source: SteadyStudySource,
    retained_context_bytes: u64,
    original_import_workspace_bytes: Option<u64>,
    pub phase: SteadyStudyPhase,
    pub accuracy_task: Option<PreparedAccuracy>,
    completed_accuracy_host: Option<AnalysisHostWorkload>,
    pub state_task: Option<PreparedSemanticStatePublication>,
    pub publication_task: Option<PreparedResultPublication>,
    pub accuracy: Option<ImportedAccuracy>,
    pub states: Vec<ImportedSemanticState>,
    pub usage: Option<PreparedResourceUsage>,
    pub baseline: Option<PreparedResourceUsage>,
    pub terminal: Option<PreparedResourceUsage>,
    pub publication: Option<ImportedPublishedResult>,
}

impl SteadyStudyTail {
    pub fn new(source: SteadyStudySource, usage: PreparedResourceUsage) -> Self {
        Self {
            source,
            retained_context_bytes: 0,
            original_import_workspace_bytes: None,
            phase: SteadyStudyPhase::ReadyForAccuracy,
            accuracy_task: None,
            completed_accuracy_host: None,
            state_task: None,
            publication_task: None,
            accuracy: None,
            states: Vec::new(),
            usage: Some(usage),
            baseline: None,
            terminal: None,
            publication: None,
        }
    }
    pub fn prepare_accuracy(&mut self) -> AnalysisBridgeResult<&PreparedAccuracy> {
        self.require(SteadyStudyPhase::ReadyForAccuracy)?;

        let task = prepare_accuracy(AccuracyPreparation {
            solve_host: self.source.host.clone(),
            source_kind: self.source.accuracy_kind,
            source_numerical_receipt: self.source.receipt.clone(),
            source_observations: self.source.observations.clone(),
            // Accuracy admits workload/mesh/result source authority. The original
            // solve plan remains retained for state publication, not this phase.
            source_context_inputs: self
                .source
                .context_inputs
                .iter()
                .filter(|input| {
                    input.manifest.kind
                        != runmat_analysis_fea::execution::AnalysisNumericalObjectKind::ResourcePlan
                })
                .cloned()
                .collect(),
            sequence: self.source.sequence,
            step: self.source.step.clone(),
            program_revision: self.source.program_revision.clone(),
            chunk_bytes: self.source.output_policy.chunk_bytes(),
            inventory_limits: self.source.output_policy.inventory_limits(),
        })?;
        self.require_retained_budget(&task.resource_plan)?;
        self.accuracy_task = Some(task);
        self.phase = SteadyStudyPhase::AwaitingAccuracy;
        Ok(self.accuracy_task.as_ref().expect("just prepared"))
    }

    pub fn complete_accuracy(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(SteadyStudyPhase::AwaitingAccuracy)?;
        let task = self.accuracy_task.as_ref().expect("awaited accuracy");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let accuracy = import_accuracy(
            source,
            response,
            task,
            self.source.output_policy.inventory_limits(),
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        let host = task.host.clone();
        self.accuracy = Some(accuracy);
        self.usage = Some(usage);
        // Publication needs exact producer lineage, not completed task input/program copies.
        self.completed_accuracy_host = Some(host);
        self.accuracy_task = None;
        if matches!(
            self.source.state,
            TerminalStatePublicationSource::NonlinearOriginal
        ) {
            // Accuracy authenticated workload/mesh/result together. Subsequent original-state
            // tasks use only workload plus source.result; publication uses accuracy lineage.
            // Release the completed accuracy's mesh and duplicate result input buffers.
            self.source.context_inputs.retain(|input| {
                input.manifest.kind
                    == runmat_analysis_fea::execution::AnalysisNumericalObjectKind::NonlinearSolidWorkload
            });
        }
        self.advance_state();
        Ok(())
    }

    /// Publish the exact complete static state declared by the original authored step.
    pub fn prepare_next_state(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedSemanticStatePublication> {
        let ordinal = self.states.len();
        self.require(SteadyStudyPhase::ReadyForState { ordinal })?;
        let declaration = self.source.step.state_outputs[ordinal].clone();
        let task = match &self.source.state {
            TerminalStatePublicationSource::Profiles {
                profiles,
                kind,
                topology,
            } => prepare_semantic_state_publication(SemanticStatePublicationPreparation {
                source_host: self.source.host.clone(),
                source_result: self.source.result.clone(),
                    source_profiles: profiles.as_ref().clone(),
                source: *kind,
                declaration,
                topology_digest: *topology,
                program_revision: self.source.program_revision.clone(),
                chunk_bytes: self.source.output_policy.chunk_bytes(),
                inventory_limits: self.source.output_policy.inventory_limits(),
            })?,
            TerminalStatePublicationSource::FluidOriginal => {
                crate::prepare_fluid_state_publication(crate::FluidStatePublicationPreparation {
                    source_host: self.source.host.clone(),
                    source_inputs: self.source.context_inputs.iter().filter(|input|
                        input.manifest.kind != runmat_analysis_fea::execution::AnalysisNumericalObjectKind::SolverResult)
                        .cloned().collect(),
                    source_result: self.source.result.clone(),
                    declaration,
                    program_revision: self.source.program_revision.clone(),
                    chunk_bytes: self.source.output_policy.chunk_bytes(),
                    inventory_limits: self.source.output_policy.inventory_limits(),
                })?
            }
            TerminalStatePublicationSource::NonlinearOriginal => {
                let workload = self.source.context_inputs.iter().find(|i| i.manifest.kind
                    == runmat_analysis_fea::execution::AnalysisNumericalObjectKind::NonlinearSolidWorkload)
                    .ok_or_else(|| invalid("original nonlinear workload missing"))?;
                crate::prepare_nonlinear_solid_state_publication(crate::NonlinearSolidStatePublicationPreparation {
                    source_host: self.source.host.clone(), workload: workload.clone(), block: self.source.result.clone(),
                    declaration, program_revision: self.source.program_revision.clone(), output_policy: self.source.output_policy,
                })?
            }
        };
        self.require_retained_budget(&task.resource_plan)?;
        self.state_task = Some(task);
        self.phase = SteadyStudyPhase::AwaitingState { ordinal };
        Ok(self.state_task.as_ref().expect("just prepared"))
    }

    pub fn complete_next_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(SteadyStudyPhase::AwaitingState {
            ordinal: self.states.len(),
        })?;
        let task = self.state_task.as_ref().expect("awaited state");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let mut state = import_semantic_state_completion(
            source,
            response,
            task,
            self.source.output_policy.inventory_limits(),
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        for previous in &self.states {
            state
                .state
                .share_original_nonlinear_payload(&previous.state);
        }
        self.states.push(state);
        self.usage = Some(usage);
        self.state_task = None;
        self.advance_state();
        Ok(())
    }

    /// Freeze the immutable solve/accuracy/all-states baseline named by the receipt.
    pub fn freeze_baseline(&mut self) -> AnalysisBridgeResult<&PreparedResourceUsage> {
        self.require(SteadyStudyPhase::ReadyForBaseline)?;
        self.baseline = self.usage.clone();
        self.phase = SteadyStudyPhase::ReadyForPublication;
        Ok(self.baseline.as_ref().expect("completed usage"))
    }

    pub fn prepare_publication(&mut self) -> AnalysisBridgeResult<&PreparedResultPublication> {
        self.require(SteadyStudyPhase::ReadyForPublication)?;
        let task = prepare_result_publication(ResultPublicationPreparation {
            accuracy_host: self
                .completed_accuracy_host
                .as_ref()
                .expect("completed accuracy host")
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
            source_observations: self.source.observations.clone(),
            program_revision: self.source.program_revision.clone(),
            chunk_bytes: self.source.output_policy.chunk_bytes(),
            inventory_limits: self.source.output_policy.inventory_limits(),
        })?;
        self.require_retained_budget(&task.resource_plan)?;
        self.publication_task = Some(task);
        self.phase = SteadyStudyPhase::AwaitingPublication;
        Ok(self.publication_task.as_ref().expect("just prepared"))
    }

    /// Commit receipt and terminal accounting atomically; neither is visible on failure.
    pub fn complete_publication(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(SteadyStudyPhase::AwaitingPublication)?;
        let task = self.publication_task.as_ref().expect("awaited publication");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let publication = import_published_result(
            source,
            response,
            task,
            self.source.output_policy.inventory_limits(),
        )?;
        let terminal = complete_result_publication_resource_usage(
            task,
            &publication,
            self.baseline.as_ref().expect("frozen baseline"),
            &accounting,
            self.source.output_policy,
        )?;
        self.publication = Some(publication);
        self.terminal = Some(terminal);
        self.publication_task = None;
        self.phase = SteadyStudyPhase::Complete;
        Ok(())
    }

    fn require(&self, expected: SteadyStudyPhase) -> AnalysisBridgeResult<()> {
        if self.phase != expected {
            return Err(invalid(format!(
                "steady study phase {:?} does not permit {:?}",
                self.phase, expected
            )));
        }
        Ok(())
    }

    /// Additional caller-owned context profiled by a coordinating adapter. This is a
    /// conservative admission component, not process-RSS evidence or a new worker plan.
    pub(crate) fn set_retained_context_bytes(
        &mut self,
        bytes: u64,
        original_import_workspace_bytes: Option<u64>,
    ) {
        self.retained_context_bytes = bytes;
        self.original_import_workspace_bytes = original_import_workspace_bytes;
    }

    fn require_retained_budget(&self, plan: &FeaResourcePlan) -> AnalysisBridgeResult<()> {
        if self.retained_context_bytes == 0 {
            return Ok(());
        }
        let additional = match self.original_import_workspace_bytes {
            Some(workspace) if workspace <= self.retained_context_bytes => {
                // The coordinating importer already includes the entire original
                // closure's transport/decode workspace in retained_context_bytes.
                // It covers this saved phase; no source program executes here.
                0
            }
            Some(_) => return Err(invalid("original phase import lacks retained workspace")),
            None => plan.request.memory_bytes,
        };
        let bytes = self
            .retained_context_bytes
            .checked_add(additional)
            .ok_or_else(|| invalid("terminal retained context overflow"))?;
        if bytes > self.source.step.accuracy.resources.resident_byte_budget() {
            return Err(
                runmat_analysis_fea::planning::resource::ResourcePlanningError::AccuracyCeiling {
                    resource: "terminal retained resident bytes",
                    required: bytes,
                    maximum: self.source.step.accuracy.resources.resident_byte_budget(),
                }
                .into(),
            );
        }
        Ok(())
    }

    fn advance_state(&mut self) {
        self.phase = if self.states.len() < self.source.step.state_outputs.len() {
            SteadyStudyPhase::ReadyForState {
                ordinal: self.states.len(),
            }
        } else {
            SteadyStudyPhase::ReadyForBaseline
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
            &self.source.host.artifact_access,
            self.source.output_policy,
        )
    }
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
