//! Shared placement-neutral terminal authority for fresh static electromagnetic studies.
//! Solve adapters supply original authenticated sources; this engine never solves or recovers.
use crate::{
    complete_result_publication_resource_usage, extend_resource_usage, import_accuracy,
    import_published_result, import_semantic_state_completion, prepare_accuracy,
    prepare_result_publication, prepare_semantic_state_publication, AccuracyPreparation,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, AnalysisOutputPolicy,
    ImportedAccuracy, ImportedPublishedResult, ImportedSemanticState, PlannedResourceUsageAttempt,
    PlannedResourceUsageGraph, PreparedAccuracy, PreparedAnalysisObjects, PreparedResourceUsage,
    PreparedResultPublication, PreparedSemanticStatePublication, ResultPublicationPreparation,
    SemanticStatePublicationPreparation,
};
use runmat_analysis_core::{FormulationRequirement, ProcedureStep};
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, SemanticStatePublicationSource, SequenceIdentity},
    planning::resource::FeaResourcePlan,
    results::observation::ObservationPublicationCatalog,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, ProgramExecutionResponse};
use runmat_execution_runner::AttemptResourceAccounting;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StaticEmStudyPhase {
    ReadyForAccuracy,
    AwaitingAccuracy,
    ReadyForState { ordinal: usize },
    AwaitingState { ordinal: usize },
    ReadyForBaseline,
    ReadyForPublication,
    AwaitingPublication,
    Complete,
}

pub(crate) struct StaticEmStudySource {
    pub host: AnalysisHostWorkload,
    pub result: PreparedAnalysisObjects,
    pub profiles: PreparedAnalysisObjects,
    pub receipt: PreparedAnalysisObjects,
    pub context_inputs: Vec<PreparedAnalysisObjects>,
    pub accuracy_kind: AccuracySourceKind,
    pub state_kind: SemanticStatePublicationSource,
    pub topology: Digest,
    pub sequence: SequenceIdentity,
    pub step: ProcedureStep,
    pub formulation_requirement: FormulationRequirement,
    pub observations: Option<ObservationPublicationCatalog>,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
}

pub(crate) struct StaticEmStudyTail {
    source: StaticEmStudySource,
    pub phase: StaticEmStudyPhase,
    pub accuracy_task: Option<PreparedAccuracy>,
    pub state_task: Option<PreparedSemanticStatePublication>,
    pub publication_task: Option<PreparedResultPublication>,
    pub accuracy: Option<ImportedAccuracy>,
    pub states: Vec<ImportedSemanticState>,
    pub usage: Option<PreparedResourceUsage>,
    pub baseline: Option<PreparedResourceUsage>,
    pub terminal: Option<PreparedResourceUsage>,
    pub publication: Option<ImportedPublishedResult>,
}

impl StaticEmStudyTail {
    pub fn new(source: StaticEmStudySource, usage: PreparedResourceUsage) -> Self {
        Self {
            source,
            phase: StaticEmStudyPhase::ReadyForAccuracy,
            accuracy_task: None,
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
        self.require(StaticEmStudyPhase::ReadyForAccuracy)?;

        let task = prepare_accuracy(AccuracyPreparation {
            solve_host: self.source.host.clone(),
            source_kind: self.source.accuracy_kind,
            source_numerical_receipt: self.source.receipt.clone(),
            source_observations: self.source.observations.clone(),
            source_context_inputs: self.source.context_inputs.clone(),
            sequence: self.source.sequence,
            step: self.source.step.clone(),
            formulation_requirement: self.source.formulation_requirement,
            program_revision: self.source.program_revision.clone(),
            chunk_bytes: self.source.output_policy.chunk_bytes(),
            inventory_limits: self.source.output_policy.inventory_limits(),
        })?;
        self.accuracy_task = Some(task);
        self.phase = StaticEmStudyPhase::AwaitingAccuracy;
        Ok(self.accuracy_task.as_ref().expect("just prepared"))
    }

    pub fn complete_accuracy(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        self.require(StaticEmStudyPhase::AwaitingAccuracy)?;
        let task = self.accuracy_task.as_ref().expect("awaited accuracy");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let accuracy = import_accuracy(
            source,
            response,
            task,
            self.source.output_policy.inventory_limits(),
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        self.accuracy = Some(accuracy);
        self.usage = Some(usage);
        self.advance_state();
        Ok(())
    }

    /// Publish the exact complete static state declared by the original authored step.
    pub fn prepare_next_state(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedSemanticStatePublication> {
        let ordinal = self.states.len();
        self.require(StaticEmStudyPhase::ReadyForState { ordinal })?;
        let declaration = self.source.step.state_outputs[ordinal].clone();
        let task = prepare_semantic_state_publication(SemanticStatePublicationPreparation {
            source_host: self.source.host.clone(),
            source_result: self.source.result.clone(),
            source_profiles: self.source.profiles.clone(),
            source: self.source.state_kind,
            declaration,
            topology_digest: self.source.topology,
            program_revision: self.source.program_revision.clone(),
            chunk_bytes: self.source.output_policy.chunk_bytes(),
            inventory_limits: self.source.output_policy.inventory_limits(),
        })?;
        self.state_task = Some(task);
        self.phase = StaticEmStudyPhase::AwaitingState { ordinal };
        Ok(self.state_task.as_ref().expect("just prepared"))
    }

    pub fn complete_next_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        self.require(StaticEmStudyPhase::AwaitingState {
            ordinal: self.states.len(),
        })?;
        let task = self.state_task.as_ref().expect("awaited state");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let state = import_semantic_state_completion(
            source,
            response,
            task,
            self.source.output_policy.inventory_limits(),
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        self.states.push(state);
        self.usage = Some(usage);
        self.state_task = None;
        self.advance_state();
        Ok(())
    }

    /// Freeze the immutable solve/accuracy/all-states baseline named by the receipt.
    pub fn freeze_baseline(&mut self) -> AnalysisBridgeResult<&PreparedResourceUsage> {
        self.require(StaticEmStudyPhase::ReadyForBaseline)?;
        self.baseline = self.usage.clone();
        self.phase = StaticEmStudyPhase::ReadyForPublication;
        Ok(self.baseline.as_ref().expect("completed usage"))
    }

    pub fn prepare_publication(&mut self) -> AnalysisBridgeResult<&PreparedResultPublication> {
        self.require(StaticEmStudyPhase::ReadyForPublication)?;
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
            source_observations: self.source.observations.clone(),
            program_revision: self.source.program_revision.clone(),
            chunk_bytes: self.source.output_policy.chunk_bytes(),
            inventory_limits: self.source.output_policy.inventory_limits(),
        })?;
        self.publication_task = Some(task);
        self.phase = StaticEmStudyPhase::AwaitingPublication;
        Ok(self.publication_task.as_ref().expect("just prepared"))
    }

    /// Commit receipt and terminal accounting atomically; neither is visible on failure.
    pub fn complete_publication(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        self.require(StaticEmStudyPhase::AwaitingPublication)?;
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
            &PlannedResourceUsageAttempt {
                plan: task.resource_plan.clone(),
                accounting,
            },
            self.source.output_policy,
        )?;
        self.publication = Some(publication);
        self.terminal = Some(terminal);
        self.phase = StaticEmStudyPhase::Complete;
        Ok(())
    }

    fn require(&self, expected: StaticEmStudyPhase) -> AnalysisBridgeResult<()> {
        if self.phase != expected {
            return Err(invalid(format!(
                "static EM study phase {:?} does not permit {:?}",
                self.phase, expected
            )));
        }
        Ok(())
    }

    fn advance_state(&mut self) {
        self.phase = if self.states.len() < self.source.step.state_outputs.len() {
            StaticEmStudyPhase::ReadyForState {
                ordinal: self.states.len(),
            }
        } else {
            StaticEmStudyPhase::ReadyForBaseline
        };
    }

    fn extend_usage(
        &self,
        plan: &FeaResourcePlan,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<PreparedResourceUsage> {
        let wall_millis = accounting.resources.usage.wall_millis;
        let graph = PlannedResourceUsageGraph::new(
            vec![PlannedResourceUsageAttempt {
                plan: plan.clone(),
                accounting,
            }],
            wall_millis,
        )?;
        extend_resource_usage(
            self.usage.as_ref().expect("completed solve usage"),
            &graph,
            &self.source.host.artifact_access,
            self.source.output_policy,
        )
    }
}

pub(crate) fn validate_accounting(
    host: &AnalysisHostWorkload,
    plan: &FeaResourcePlan,
    response: &ProgramExecutionResponse,
    accounting: &AttemptResourceAccounting,
) -> AnalysisBridgeResult<()> {
    accounting
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } = response else {
        return Err(invalid(
            "static EM study requires externalized phase completion",
        ));
    };
    let bytes = result_objects
        .iter()
        .try_fold(0_u64, |sum, object| sum.checked_add(object.encoded_length))
        .ok_or_else(|| invalid("static EM phase output byte count overflow"))?;
    if bytes != accounting.resources.artifact_bytes
        || !runmat_analysis_fea::planning::resource::allocation_matches_plan(
            &accounting.resources.allocation,
            &plan.request,
        )
        || !host
            .workload
            .generic_capabilities()?
            .is_subset(&accounting.resources.allocation.required_capabilities)
    {
        return Err(AnalysisBridgeError::Identity(
            "static EM phase accounting differs from its admitted request and output closure",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
