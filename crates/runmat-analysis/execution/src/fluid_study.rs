//! Placement-neutral orchestration of a fresh authored standalone steady fluid step.
//!
//! Prepared tasks remain owned here: callers execute their immutable requests and return the
//! associated scheduler response/accounting pair. Failed imports never advance the lifecycle.
use crate::phase_accounting::validate_analysis_phase_accounting as validate_accounting;
use crate::steady_study::{SteadyStudyPhase, SteadyStudySource, SteadyStudyTail};
use std::collections::BTreeSet;

use crate::AnalysisPhaseExecutionEvidence;
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, SequenceIdentity},
    study::{
        plan_fluid_study_product, CompiledAnalysisStudy, FluidStudyPlanningPolicy,
        FluidStudyProduct,
    },
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, ArtifactAccess, ProgramExecutionResponse};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    import_fluid_steady_result, prepare_fluid_steady_task, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisOutputPolicy, FluidSteadyTaskPreparation, ImportedAccuracy, ImportedFluidSteadyResult,
    ImportedPublishedResult, ImportedSemanticState, PreparedAccuracy, PreparedFluidSteadyTask,
    PreparedResourceUsage, PreparedResultPublication, PreparedSemanticStatePublication,
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
pub use root::{
    import_fluid_study_root, prepare_fluid_study_root, prepare_fluid_study_root_for,
    PreparedFluidStudyRoot,
};

/// Owned terminal artifacts for the exact standalone authored steady envelope.
/// Consuming the coordinator does not clone numerical storage.
pub struct CompletedFluidStudy {
    replay: completion::FluidStudyReplayContext,
    closures: Vec<FluidStudyResultClosure>,
    /// The field-recovery admission the coordinator derived when it prepared that phase.
    field_recovery_plan: runmat_analysis_fea::planning::resource::FeaResourcePlan,
    /// The compiled study and workload planned on the study's mesh (sealed).
    pub product: FluidStudyProduct,
    pub result: ImportedFluidSteadyResult,
    pub fields: crate::ImportedRecoveredFields,
    /// Field outputs assembled per retained sample, histories finalized over them, and the
    /// publication catalog of both.
    pub observations: crate::CompletedSampledObservations,
    pub accuracy: ImportedAccuracy,
    pub states: Vec<ImportedSemanticState>,
    pub baseline_usage: PreparedResourceUsage,
    pub terminal_usage: PreparedResourceUsage,
    pub publication: ImportedPublishedResult,
}

/// A study this process's own coordinator drove to completion: each phase response was
/// admitted and validated by `FluidStudyCoordinator` as it arrived, in this process.
/// Only [`FluidStudyCoordinator::into_verified`] constructs it, so holding one is the proof
/// that no replay is needed. Completions decoded from bytes stay [`CompletedFluidStudy`]
/// and are authenticated by a full replay.
pub struct VerifiedFluidStudy(CompletedFluidStudy);

impl VerifiedFluidStudy {
    /// The coordinator's own preparation: `FluidStudyCoordinator::new` validated it and
    /// every phase was admitted against it, so it is not validated or re-planned again.
    pub fn preparation(&self) -> FluidStudyPreparation {
        self.0.assemble_preparation()
    }

    /// Give up the in-process proof (for example, to treat the study as a cold source).
    pub fn into_completed(self) -> CompletedFluidStudy {
        self.0
    }
}

impl std::ops::Deref for VerifiedFluidStudy {
    type Target = CompletedFluidStudy;
    fn deref(&self) -> &CompletedFluidStudy {
        &self.0
    }
}

#[derive(Clone)]
pub struct FluidStudyPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub policy: FluidStudyPlanningPolicy,
    pub product: FluidStudyProduct,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

/// Everything a fluid study preparation holds except the planned product.
pub struct FluidStudyPreparationParts {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub policy: FluidStudyPlanningPolicy,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

/// A preparation whose product is the canonical plan of its own compiled study on its own
/// mesh under its own policy. Only planning constructs it ([`Self::plan`] in process, or
/// [`Self::verify`], which binds a preparation assembled elsewhere to its sealed product),
/// so [`FluidStudyCoordinator::planned`] takes it without planning the product a second time.
#[derive(Clone)]
pub struct PlannedFluidStudyPreparation(FluidStudyPreparation);

impl PlannedFluidStudyPreparation {
    /// Plan `compiled` on `parts.mesh` under `parts.policy`; the product stays with them.
    pub fn plan(
        compiled: CompiledAnalysisStudy,
        parts: FluidStudyPreparationParts,
    ) -> AnalysisBridgeResult<Self> {
        // Planning validates the compiled study against the mesh first.
        let product = plan_fluid_study_product(compiled, &parts.mesh, parts.policy)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(Self(FluidStudyPreparation {
            local_execution: parts.local_execution,
            policy: parts.policy,
            product,
            mesh: parts.mesh,
            frozen_project_digest: parts.frozen_project_digest,
            artifact_access: parts.artifact_access,
            program_revision: parts.program_revision,
            output_policy: parts.output_policy,
            required_capabilities: parts.required_capabilities,
        }))
    }

    /// Bind a preparation assembled elsewhere to its product. The product is sealed (only
    /// planning constructs it), so this authenticates the preparation's mesh content and
    /// requires the product to have been planned on that mesh under the preparation's policy.
    pub fn verify(preparation: FluidStudyPreparation) -> AnalysisBridgeResult<Self> {
        runmat_meshing_core::CanonicalMeshingContract::validate_canonical(&preparation.mesh)
            .map_err(|error| invalid(error.to_string()))?;
        preparation
            .product
            .validate_planned_on(&preparation.mesh, preparation.policy)
            .map_err(|_| {
                AnalysisBridgeError::Identity(
                    "fluid workload differs from the compiled authored study",
                )
            })?;
        Ok(Self(preparation))
    }

    pub fn preparation(&self) -> &FluidStudyPreparation {
        &self.0
    }

    pub fn into_preparation(self) -> FluidStudyPreparation {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FluidStudyPhase {
    ReadyForSolve,
    AwaitingSolve,
    ReadyForFieldRecovery,
    AwaitingFieldRecovery,
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
    observation_state: crate::sampled_history_observations::SampledHistoryWorkflow,
    usage: Option<PreparedResourceUsage>,
}

impl FluidStudyCoordinator {
    /// For a preparation assembled elsewhere: binds it first ([`PlannedFluidStudyPreparation::verify`]).
    pub fn new(preparation: FluidStudyPreparation) -> AnalysisBridgeResult<Self> {
        Self::planned(PlannedFluidStudyPreparation::verify(preparation)?)
    }

    /// For a preparation planned in process: its product is canonical by construction.
    pub fn planned(preparation: PlannedFluidStudyPreparation) -> AnalysisBridgeResult<Self> {
        let preparation = preparation.into_preparation();
        let definition_digest = preparation
            .product
            .compiled()
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
            observation_state: crate::sampled_history_observations::SampledHistoryWorkflow::default(
            ),
            usage: None,
        })
    }

    pub fn prepare_solve(&mut self) -> AnalysisBridgeResult<&PreparedFluidSteadyTask> {
        self.require(FluidStudyPhase::ReadyForSolve)?;
        let task = prepare_fluid_steady_task(FluidSteadyTaskPreparation {
            local_execution: self.preparation.local_execution,
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
        accounting: AnalysisPhaseExecutionEvidence,
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
        let graph = accounting.resource_graph(&task.resource_plan)?;
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
        accounting: AnalysisPhaseExecutionEvidence,
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
        accounting: AnalysisPhaseExecutionEvidence,
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
        accounting: AnalysisPhaseExecutionEvidence,
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

    /// The completed study as this process's own verified result.
    pub fn into_verified(self) -> AnalysisBridgeResult<VerifiedFluidStudy> {
        self.into_completed().map(VerifiedFluidStudy)
    }

    pub fn into_completed(self) -> AnalysisBridgeResult<CompletedFluidStudy> {
        self.require(FluidStudyPhase::Complete)?;
        let tail = self.tail.expect("completed shared terminal lifecycle");
        Ok(CompletedFluidStudy {
            replay: completion::FluidStudyReplayContext {
                local_execution: self.preparation.local_execution,
                policy: self.preparation.policy,
                mesh: self.preparation.mesh,
                frozen_project_digest: self.preparation.frozen_project_digest,
                artifact_access: self.preparation.artifact_access,
                program_revision: self.preparation.program_revision,
                output_policy: self.preparation.output_policy,
                required_capabilities: self.preparation.required_capabilities,
            },
            closures: self.closures,
            field_recovery_plan: self
                .recovery_task
                .expect("completed field recovery")
                .resource_plan,
            product: self.preparation.product,
            result: self.result.expect("completed solve"),
            fields: self.fields.expect("completed retained field publication"),
            observations: self.observation_state.into_completed(),
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
        self.preparation.product.compiled()
    }
    pub fn result(&self) -> Option<&ImportedFluidSteadyResult> {
        self.result.as_ref()
    }
    pub fn fields(&self) -> Option<&crate::ImportedRecoveredFields> {
        self.fields.as_ref()
    }
    pub fn observations(&self) -> &crate::CompletedSampledObservations {
        self.observation_state.completed()
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
        self.product.workload()
    }
    pub(super) fn selected_step(&self) -> &runmat_analysis_core::ProcedureStep {
        &self.product.compiled().model.sequence.steps[0]
    }
}
