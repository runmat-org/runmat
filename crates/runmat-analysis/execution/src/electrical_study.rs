//! Placement-neutral orchestration of a fresh authored standalone static electrical step.
//!
//! Prepared tasks remain owned here: callers execute their immutable requests and return the
//! associated scheduler response/accounting pair. Failed imports never advance the lifecycle.
use std::collections::BTreeSet;

use runmat_analysis_fea::{
    execution::{AccuracySourceKind, SemanticStatePublicationSource, SequenceIdentity},
    planning::resource::FeaResourcePlan,
    study::{
        plan_electrical_study, CompiledAnalysisStudy, ElectricalStudyPlanningPolicy,
        ElectricalStudyProduct,
    },
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, ArtifactAccess, ProgramExecutionResponse};
use runmat_execution_runner::AttemptResourceAccounting;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    complete_result_publication_resource_usage, extend_resource_usage, import_accuracy,
    import_electrical_static_result, import_published_result, import_semantic_state_completion,
    prepare_accuracy, prepare_electrical_static_task, prepare_result_publication,
    prepare_semantic_state_publication, AccuracyPreparation, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisOutputPolicy, ElectricalStaticTaskPreparation, ImportedAccuracy,
    ImportedElectricalStaticResult, ImportedPublishedResult, ImportedSemanticState,
    PlannedResourceUsageAttempt, PlannedResourceUsageGraph, PreparedAccuracy,
    PreparedElectricalStaticTask, PreparedResourceUsage, PreparedResultPublication,
    PreparedSemanticStatePublication, ResultPublicationPreparation,
    SemanticStatePublicationPreparation,
};

mod completion;
mod root;
#[cfg(test)]
mod tests;
pub use completion::{
    import_electrical_study_completion, ElectricalStudyResultClosure, ElectricalStudyResultPhase,
};
pub use root::{
    import_electrical_study_root, prepare_electrical_study_root, PreparedElectricalStudyRoot,
};

/// Owned terminal artifacts for the exact standalone authored static envelope.
/// Consuming the coordinator does not clone numerical storage.
pub struct CompletedElectricalStudy {
    replay: completion::ElectricalStudyReplayContext,
    closures: Vec<ElectricalStudyResultClosure>,
    pub compiled: CompiledAnalysisStudy,
    pub workload: runmat_analysis_fea::execution::ElectricalStaticWorkload,
    pub result: ImportedElectricalStaticResult,
    pub accuracy: ImportedAccuracy,
    pub states: Vec<ImportedSemanticState>,
    pub baseline_usage: PreparedResourceUsage,
    pub terminal_usage: PreparedResourceUsage,
    pub publication: ImportedPublishedResult,
}

pub struct ElectricalStudyPreparation {
    pub policy: ElectricalStudyPlanningPolicy,
    pub product: ElectricalStudyProduct,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElectricalStudyPhase {
    ReadyForSolve,
    AwaitingSolve,
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
pub struct ElectricalStudyCoordinator {
    closures: Vec<ElectricalStudyResultClosure>,
    preparation: ElectricalStudyPreparation,
    definition_digest: Digest,
    phase: ElectricalStudyPhase,
    solve_task: Option<PreparedElectricalStaticTask>,
    accuracy_task: Option<PreparedAccuracy>,
    state_task: Option<PreparedSemanticStatePublication>,
    publication_task: Option<PreparedResultPublication>,
    result: Option<ImportedElectricalStaticResult>,
    accuracy: Option<ImportedAccuracy>,
    states: Vec<ImportedSemanticState>,
    usage: Option<PreparedResourceUsage>,
    baseline: Option<PreparedResourceUsage>,
    terminal: Option<PreparedResourceUsage>,
    publication: Option<ImportedPublishedResult>,
}

impl ElectricalStudyCoordinator {
    pub fn new(preparation: ElectricalStudyPreparation) -> AnalysisBridgeResult<Self> {
        preparation
            .product
            .compiled
            .validate_against(&preparation.mesh)
            .map_err(|error| invalid(error.to_string()))?;
        let expected = plan_electrical_study(&preparation.product.compiled, preparation.policy)
            .map_err(|error| invalid(error.to_string()))?;
        if expected != preparation.product.workload {
            return Err(AnalysisBridgeError::Identity(
                "electrical workload differs from the compiled authored study",
            ));
        }
        let definition_digest = preparation
            .product
            .compiled
            .definition_digest()
            .map_err(|error| invalid(error.to_string()))?;
        Ok(Self {
            closures: Vec::new(),
            preparation,
            definition_digest,
            phase: ElectricalStudyPhase::ReadyForSolve,
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
        })
    }

    pub fn prepare_solve(&mut self) -> AnalysisBridgeResult<&PreparedElectricalStaticTask> {
        self.require(ElectricalStudyPhase::ReadyForSolve)?;
        let task = prepare_electrical_static_task(ElectricalStaticTaskPreparation {
            authored_study_digest: self.definition_digest,
            frozen_project_digest: self.preparation.frozen_project_digest,
            program_revision: self.preparation.program_revision.clone(),
            artifact_access: self.preparation.artifact_access.clone(),
            workload: self.preparation.workload().clone(),
            mesh: self.preparation.mesh.clone(),
            output_policy: self.preparation.output_policy,
            required_capabilities: self.preparation.required_capabilities.clone(),
        })?;
        self.solve_task = Some(task);
        self.phase = ElectricalStudyPhase::AwaitingSolve;
        Ok(self.solve_task.as_ref().expect("just prepared"))
    }

    /// The caller must preserve the scheduler's response/accounting association.
    pub fn complete_solve(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        self.require(ElectricalStudyPhase::AwaitingSolve)?;
        let task = self.solve_task.as_ref().expect("awaited solve");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = ElectricalStudyResultClosure::new(
            ElectricalStudyResultPhase::Solve,
            &task.host,
            &response,
            &accounting,
        );
        let result = import_electrical_static_result(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let graph = PlannedResourceUsageGraph::new(
            vec![PlannedResourceUsageAttempt {
                plan: task.resource_plan.clone(),
                accounting: accounting.clone(),
            }],
            accounting.resources.usage.wall_millis,
        )?;
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
        self.phase = ElectricalStudyPhase::ReadyForAccuracy;
        Ok(())
    }

    pub fn prepare_accuracy(&mut self) -> AnalysisBridgeResult<&PreparedAccuracy> {
        self.require(ElectricalStudyPhase::ReadyForAccuracy)?;
        let result = self.result.as_ref().expect("completed solve");
        let workload = self.preparation.workload();
        let task =
            prepare_accuracy(AccuracyPreparation {
                solve_host: self.solve_task.as_ref().expect("completed solve").host.clone(),
                source_kind: AccuracySourceKind::ElectricalStatic,
                source_numerical_receipt: result.numerical_receipt_objects.clone(),
                source_observations: None,
                source_context_inputs: self
                    .solve_task
                    .as_ref()
                    .expect("completed solve")
                    .inputs
                    .iter()
                    .filter(|input| {
                        matches!(input.manifest.kind,
                    runmat_analysis_fea::execution::AnalysisNumericalObjectKind::ElectricalStaticWorkload
                    | runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Mesh)
                    })
                    .cloned()
                    .chain(std::iter::once(result.objects.clone()))
                    .collect(),
                sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
                step: self.preparation.selected_step().clone(),
                formulation_requirement: workload.formulation_requirement,
                program_revision: self.preparation.program_revision.clone(),
                chunk_bytes: self.preparation.output_policy.chunk_bytes(),
                inventory_limits: self.preparation.output_policy.inventory_limits(),
            })?;
        self.accuracy_task = Some(task);
        self.phase = ElectricalStudyPhase::AwaitingAccuracy;
        Ok(self.accuracy_task.as_ref().expect("just prepared"))
    }

    pub fn complete_accuracy(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        self.require(ElectricalStudyPhase::AwaitingAccuracy)?;
        let task = self.accuracy_task.as_ref().expect("awaited accuracy");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = ElectricalStudyResultClosure::new(
            ElectricalStudyResultPhase::Accuracy,
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

    /// Publish the exact complete static state declared by the standalone electrical step.
    pub fn prepare_next_state(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedSemanticStatePublication> {
        let ordinal = self.states.len();
        self.require(ElectricalStudyPhase::ReadyForState { ordinal })?;
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
            source: SemanticStatePublicationSource::StaticElectricalNativeFields,
            declaration,
            topology_digest: Digest::from_bytes(*self.preparation.mesh.canonical_digest.bytes()),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.output_policy.chunk_bytes(),
            inventory_limits: self.preparation.output_policy.inventory_limits(),
        })?;
        self.state_task = Some(task);
        self.phase = ElectricalStudyPhase::AwaitingState { ordinal };
        Ok(self.state_task.as_ref().expect("just prepared"))
    }

    pub fn complete_next_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        self.require(ElectricalStudyPhase::AwaitingState {
            ordinal: self.states.len(),
        })?;
        let task = self.state_task.as_ref().expect("awaited state");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = ElectricalStudyResultClosure::new(
            ElectricalStudyResultPhase::State {
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
        self.require(ElectricalStudyPhase::ReadyForBaseline)?;
        self.baseline = self.usage.clone();
        self.phase = ElectricalStudyPhase::ReadyForPublication;
        Ok(self.baseline.as_ref().expect("completed usage"))
    }

    pub fn prepare_publication(&mut self) -> AnalysisBridgeResult<&PreparedResultPublication> {
        self.require(ElectricalStudyPhase::ReadyForPublication)?;
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
            source_observations: None,
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: self.preparation.output_policy.chunk_bytes(),
            inventory_limits: self.preparation.output_policy.inventory_limits(),
        })?;
        self.publication_task = Some(task);
        self.phase = ElectricalStudyPhase::AwaitingPublication;
        Ok(self.publication_task.as_ref().expect("just prepared"))
    }

    /// Commit receipt and terminal accounting atomically; neither is visible on failure.
    pub fn complete_publication(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        self.require(ElectricalStudyPhase::AwaitingPublication)?;
        let task = self.publication_task.as_ref().expect("awaited publication");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = ElectricalStudyResultClosure::new(
            ElectricalStudyResultPhase::Publication,
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
            &PlannedResourceUsageAttempt {
                plan: task.resource_plan.clone(),
                accounting,
            },
            self.preparation.output_policy,
        )?;
        self.publication = Some(publication);
        self.closures.push(closure);
        self.terminal = Some(terminal);
        self.phase = ElectricalStudyPhase::Complete;
        Ok(())
    }

    pub fn phase(&self) -> ElectricalStudyPhase {
        self.phase
    }

    pub fn into_completed(self) -> AnalysisBridgeResult<CompletedElectricalStudy> {
        self.require(ElectricalStudyPhase::Complete)?;
        let workload = self.preparation.workload().clone();
        Ok(CompletedElectricalStudy {
            replay: completion::ElectricalStudyReplayContext {
                policy: self.preparation.policy,
                mesh: self.preparation.mesh,
                frozen_project_digest: self.preparation.frozen_project_digest,
                artifact_access: self.preparation.artifact_access,
                program_revision: self.preparation.program_revision,
                output_policy: self.preparation.output_policy,
                required_capabilities: self.preparation.required_capabilities,
            },
            closures: self.closures,
            compiled: self.preparation.product.compiled,
            workload,
            result: self.result.expect("completed solve"),
            accuracy: self.accuracy.expect("completed accuracy"),
            states: self.states,
            baseline_usage: self.baseline.expect("frozen baseline"),
            terminal_usage: self.terminal.expect("completed terminal accounting"),
            publication: self.publication.expect("completed publication"),
        })
    }
    pub fn is_complete(&self) -> bool {
        self.phase == ElectricalStudyPhase::Complete
    }
    pub fn policy(&self) -> ElectricalStudyPlanningPolicy {
        self.preparation.policy
    }
    pub fn selected_step(&self) -> &runmat_analysis_core::ProcedureStep {
        self.preparation.selected_step()
    }
    pub fn compiled_study(&self) -> &CompiledAnalysisStudy {
        &self.preparation.product.compiled
    }
    pub fn result(&self) -> Option<&ImportedElectricalStaticResult> {
        self.result.as_ref()
    }
    pub fn accuracy(&self) -> Option<&ImportedAccuracy> {
        self.accuracy.as_ref()
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

    fn require(&self, expected: ElectricalStudyPhase) -> AnalysisBridgeResult<()> {
        if self.phase != expected {
            return Err(invalid(format!(
                "electrical study phase {:?} does not permit {:?}",
                self.phase, expected
            )));
        }
        Ok(())
    }

    fn advance_state(&mut self) {
        self.phase = if self.states.len() < self.preparation.selected_step().state_outputs.len() {
            ElectricalStudyPhase::ReadyForState {
                ordinal: self.states.len(),
            }
        } else {
            ElectricalStudyPhase::ReadyForBaseline
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
            &self.preparation.artifact_access,
            self.preparation.output_policy,
        )
    }
}

fn validate_accounting(
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
            "electrical study requires externalized phase completion",
        ));
    };
    let bytes = result_objects
        .iter()
        .try_fold(0_u64, |sum, object| sum.checked_add(object.encoded_length))
        .ok_or_else(|| invalid("electrical phase output byte count overflow"))?;
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
            "electrical phase accounting differs from its admitted request and output closure",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}

impl ElectricalStudyPreparation {
    pub(super) fn workload(&self) -> &runmat_analysis_fea::execution::ElectricalStaticWorkload {
        &self.product.workload
    }
    pub(super) fn selected_step(&self) -> &runmat_analysis_core::ProcedureStep {
        &self.product.compiled.model.sequence.steps[0]
    }
}
