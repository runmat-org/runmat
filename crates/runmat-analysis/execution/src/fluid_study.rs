//! Placement-neutral orchestration of a fresh authored standalone steady fluid step.
//!
//! Prepared tasks remain owned here: callers execute their immutable requests and return the
//! associated scheduler response/accounting pair. Failed imports never advance the lifecycle.
use crate::steady_study::{
    validate_accounting, SteadyStudyPhase, SteadyStudySource, SteadyStudyTail,
};
use std::collections::BTreeSet;

use runmat_analysis_fea::{
    execution::{AccuracySourceKind, SequenceIdentity},
    study::{
        plan_fluid_study_product, CompiledAnalysisStudy, FluidStudyPlanningPolicy,
        FluidStudyProduct,
    },
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, ArtifactAccess, ProgramExecutionResponse};
use runmat_execution_runner::AttemptResourceAccounting;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    import_fluid_steady_result, prepare_fluid_steady_task, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisOutputPolicy, FluidSteadyTaskPreparation, ImportedAccuracy, ImportedFluidSteadyResult,
    ImportedPublishedResult, ImportedSemanticState, PlannedResourceUsageAttempt,
    PlannedResourceUsageGraph, PreparedAccuracy, PreparedFluidSteadyTask, PreparedResourceUsage,
    PreparedResultPublication, PreparedSemanticStatePublication,
};

mod completion;
mod observations;
mod recovery;
mod root;
#[cfg(test)]
pub(crate) mod tests;
pub use completion::{
    import_fluid_study_completion, FluidStudyResultClosure, FluidStudyResultPhase,
};
pub use root::{import_fluid_study_root, prepare_fluid_study_root, PreparedFluidStudyRoot};

/// Owned terminal artifacts for the exact standalone authored steady envelope.
/// Consuming the coordinator does not clone numerical storage.
pub struct CompletedFluidStudy {
    replay: completion::FluidStudyReplayContext,
    closures: Vec<FluidStudyResultClosure>,
    pub compiled: CompiledAnalysisStudy,
    pub workload: runmat_analysis_fea::execution::FluidSteadyWorkload,
    pub result: ImportedFluidSteadyResult,
    pub fields: crate::ImportedRecoveredFields,
    pub observations: Option<crate::ImportedObservations>,
    pub observation_catalog:
        Option<runmat_analysis_fea::results::observation::ObservationPublicationCatalog>,
    pub accuracy: ImportedAccuracy,
    pub states: Vec<ImportedSemanticState>,
    pub baseline_usage: PreparedResourceUsage,
    pub terminal_usage: PreparedResourceUsage,
    pub publication: ImportedPublishedResult,
}

pub struct FluidStudyPreparation {
    pub policy: FluidStudyPlanningPolicy,
    pub product: FluidStudyProduct,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FluidStudyPhase {
    ReadyForSolve,
    AwaitingSolve,
    ReadyForFieldRecovery,
    AwaitingFieldRecovery,
    ReadyForObservationFanout,
    AwaitingObservationPartitions,
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
pub struct FluidStudyCoordinator {
    closures: Vec<FluidStudyResultClosure>,
    preparation: FluidStudyPreparation,
    definition_digest: Digest,
    phase: FluidStudyPhase,
    solve_task: Option<PreparedFluidSteadyTask>,
    result: Option<ImportedFluidSteadyResult>,
    tail: Option<SteadyStudyTail>,
    recovery_task: Option<crate::PreparedFluidFieldRecovery>,
    fields: Option<crate::ImportedRecoveredFields>,
    observation_state: observations::FluidObservations,
    usage: Option<PreparedResourceUsage>,
}

impl FluidStudyCoordinator {
    pub fn new(preparation: FluidStudyPreparation) -> AnalysisBridgeResult<Self> {
        preparation
            .product
            .compiled
            .validate_against(&preparation.mesh)
            .map_err(|error| invalid(error.to_string()))?;
        let expected = plan_fluid_study_product(
            preparation.product.compiled.clone(),
            &preparation.mesh,
            preparation.policy,
        )
        .map_err(|error| invalid(error.to_string()))?;
        if expected != preparation.product {
            return Err(AnalysisBridgeError::Identity(
                "fluid workload differs from the compiled authored study",
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
            phase: FluidStudyPhase::ReadyForSolve,
            solve_task: None,
            result: None,
            tail: None,
            recovery_task: None,
            fields: None,
            observation_state: observations::FluidObservations::default(),
            usage: None,
        })
    }

    pub fn prepare_solve(&mut self) -> AnalysisBridgeResult<&PreparedFluidSteadyTask> {
        self.require(FluidStudyPhase::ReadyForSolve)?;
        let task = prepare_fluid_steady_task(FluidSteadyTaskPreparation {
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
        self.phase = FluidStudyPhase::AwaitingSolve;
        Ok(self.solve_task.as_ref().expect("just prepared"))
    }

    /// The caller must preserve the scheduler's response/accounting association.
    pub fn complete_solve(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        self.require(FluidStudyPhase::AwaitingSolve)?;
        let task = self.solve_task.as_ref().expect("awaited solve");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = FluidStudyResultClosure::new(
            FluidStudyResultPhase::Solve,
            &task.host,
            &response,
            &accounting,
        );
        let result = import_fluid_steady_result(
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
            result.result.identity()?.digest(),
            &graph,
            &task.host.artifact_access,
            self.preparation.output_policy.chunk_bytes(),
            self.preparation.output_policy.inventory_limits(),
        )?;
        self.usage = Some(usage);
        self.result = Some(result);
        self.closures.push(closure);
        self.phase = FluidStudyPhase::ReadyForFieldRecovery;
        Ok(())
    }

    pub fn prepare_accuracy(&mut self) -> AnalysisBridgeResult<&PreparedAccuracy> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("fluid solve has not completed"))?
            .prepare_accuracy()
    }
    pub fn complete_accuracy(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        let phase = FluidStudyResultPhase::Accuracy;
        let tail = self
            .tail
            .as_mut()
            .ok_or_else(|| invalid("fluid solve has not completed"))?;
        let task = tail
            .accuracy_task
            .as_ref()
            .ok_or_else(|| invalid("fluid phase has not been prepared"))?;
        let closure = FluidStudyResultClosure::new(phase, &task.host, &response, &accounting);
        tail.complete_accuracy(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
    pub fn prepare_next_state(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedSemanticStatePublication> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("fluid solve has not completed"))?
            .prepare_next_state()
    }
    pub fn complete_next_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        let phase = FluidStudyResultPhase::State {
            ordinal: self.states().len(),
        };
        let tail = self
            .tail
            .as_mut()
            .ok_or_else(|| invalid("fluid solve has not completed"))?;
        let task = tail
            .state_task
            .as_ref()
            .ok_or_else(|| invalid("fluid phase has not been prepared"))?;
        let closure = FluidStudyResultClosure::new(phase, &task.host, &response, &accounting);
        tail.complete_next_state(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
    pub fn prepare_publication(&mut self) -> AnalysisBridgeResult<&PreparedResultPublication> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("fluid solve has not completed"))?
            .prepare_publication()
    }
    pub fn complete_publication(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        let phase = FluidStudyResultPhase::Publication;
        let tail = self
            .tail
            .as_mut()
            .ok_or_else(|| invalid("fluid solve has not completed"))?;
        let task = tail
            .publication_task
            .as_ref()
            .ok_or_else(|| invalid("fluid phase has not been prepared"))?;
        let closure = FluidStudyResultClosure::new(phase, &task.host, &response, &accounting);
        tail.complete_publication(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
    pub fn freeze_baseline(&mut self) -> AnalysisBridgeResult<&PreparedResourceUsage> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("fluid solve has not completed"))?
            .freeze_baseline()
    }

    pub fn phase(&self) -> FluidStudyPhase {
        self.tail
            .as_ref()
            .map_or(self.phase, |tail| match tail.phase {
                SteadyStudyPhase::ReadyForAccuracy => FluidStudyPhase::ReadyForAccuracy,
                SteadyStudyPhase::AwaitingAccuracy => FluidStudyPhase::AwaitingAccuracy,
                SteadyStudyPhase::ReadyForState { ordinal } => {
                    FluidStudyPhase::ReadyForState { ordinal }
                }
                SteadyStudyPhase::AwaitingState { ordinal } => {
                    FluidStudyPhase::AwaitingState { ordinal }
                }
                SteadyStudyPhase::ReadyForBaseline => FluidStudyPhase::ReadyForBaseline,
                SteadyStudyPhase::ReadyForPublication => FluidStudyPhase::ReadyForPublication,
                SteadyStudyPhase::AwaitingPublication => FluidStudyPhase::AwaitingPublication,
                SteadyStudyPhase::Complete => FluidStudyPhase::Complete,
            })
    }

    pub fn into_completed(self) -> AnalysisBridgeResult<CompletedFluidStudy> {
        self.require(FluidStudyPhase::Complete)?;
        let workload = self.preparation.workload().clone();
        let tail = self.tail.expect("completed shared terminal lifecycle");
        Ok(CompletedFluidStudy {
            replay: completion::FluidStudyReplayContext {
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
            fields: self.fields.expect("completed retained field publication"),
            observations: self.observation_state.completed,
            observation_catalog: self.observation_state.catalog,
            accuracy: tail.accuracy.expect("completed accuracy"),
            states: tail.states,
            baseline_usage: tail.baseline.expect("frozen baseline"),
            terminal_usage: tail.terminal.expect("completed terminal accounting"),
            publication: tail.publication.expect("completed publication"),
        })
    }
    pub fn is_complete(&self) -> bool {
        self.phase() == FluidStudyPhase::Complete
    }
    pub fn policy(&self) -> FluidStudyPlanningPolicy {
        self.preparation.policy
    }
    pub fn selected_step(&self) -> &runmat_analysis_core::ProcedureStep {
        self.preparation.selected_step()
    }
    pub fn compiled_study(&self) -> &CompiledAnalysisStudy {
        &self.preparation.product.compiled
    }
    pub fn result(&self) -> Option<&ImportedFluidSteadyResult> {
        self.result.as_ref()
    }
    pub fn fields(&self) -> Option<&crate::ImportedRecoveredFields> {
        self.fields.as_ref()
    }
    pub fn observations(&self) -> Option<&crate::ImportedObservations> {
        self.observation_state.completed.as_ref()
    }
    pub fn observation_catalog(
        &self,
    ) -> Option<&runmat_analysis_fea::results::observation::ObservationPublicationCatalog> {
        self.observation_state.catalog.as_ref()
    }
    pub fn accuracy(&self) -> Option<&ImportedAccuracy> {
        self.tail.as_ref().and_then(|tail| tail.accuracy.as_ref())
    }
    pub fn states(&self) -> &[ImportedSemanticState] {
        self.tail
            .as_ref()
            .map_or(&[], |tail| tail.states.as_slice())
    }
    pub fn baseline_usage(&self) -> Option<&PreparedResourceUsage> {
        self.tail.as_ref().and_then(|tail| tail.baseline.as_ref())
    }
    pub fn terminal_usage(&self) -> Option<&PreparedResourceUsage> {
        self.tail.as_ref().and_then(|tail| tail.terminal.as_ref())
    }
    pub fn publication(&self) -> Option<&ImportedPublishedResult> {
        self.tail
            .as_ref()
            .and_then(|tail| tail.publication.as_ref())
    }

    fn require(&self, expected: FluidStudyPhase) -> AnalysisBridgeResult<()> {
        if self.phase() != expected {
            return Err(invalid(format!(
                "fluid study phase {:?} does not permit {:?}",
                self.phase(),
                expected
            )));
        }
        Ok(())
    }
}

pub(super) fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}

impl FluidStudyPreparation {
    pub(super) fn workload(&self) -> &runmat_analysis_fea::execution::FluidSteadyWorkload {
        &self.product.workload
    }
    pub(super) fn selected_step(&self) -> &runmat_analysis_core::ProcedureStep {
        &self.product.compiled.model.sequence.steps[0]
    }
}
