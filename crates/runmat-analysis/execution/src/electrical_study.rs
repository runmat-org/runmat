//! Placement-neutral orchestration of a fresh authored standalone static electrical step.
//!
//! Prepared tasks remain owned here: callers execute their immutable requests and return the
//! associated scheduler response/accounting pair. Failed imports never advance the lifecycle.
use crate::phase_accounting::validate_analysis_phase_accounting as validate_accounting;
use crate::steady_study::{SteadyStudyPhase, SteadyStudySource, SteadyStudyTail};
use std::collections::BTreeSet;

use crate::AnalysisPhaseExecutionEvidence;
use runmat_analysis_fea::{
    execution::{AccuracySourceKind, SemanticStatePublicationSource, SequenceIdentity},
    study::{
        plan_electrical_study, CompiledAnalysisStudy, ElectricalStudyPlanningPolicy,
        ElectricalStudyProduct,
    },
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, ArtifactAccess, ProgramExecutionResponse};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    import_electrical_static_result, prepare_electrical_static_task, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisOutputPolicy, ElectricalStaticTaskPreparation, ImportedAccuracy,
    ImportedElectricalStaticResult, ImportedPublishedResult, ImportedSemanticState,
    PreparedAccuracy, PreparedElectricalStaticTask, PreparedResourceUsage,
    PreparedResultPublication, PreparedSemanticStatePublication,
};

mod completion;
mod root;
#[cfg(test)]
pub(crate) mod tests;
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
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
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
    result: Option<ImportedElectricalStaticResult>,
    tail: Option<SteadyStudyTail>,
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
            result: None,
            tail: None,
        })
    }

    pub fn prepare_solve(&mut self) -> AnalysisBridgeResult<&PreparedElectricalStaticTask> {
        self.require(ElectricalStudyPhase::ReadyForSolve)?;
        let task = prepare_electrical_static_task(ElectricalStaticTaskPreparation {
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
        self.phase = ElectricalStudyPhase::AwaitingSolve;
        Ok(self.solve_task.as_ref().expect("just prepared"))
    }

    /// The caller must preserve the scheduler's response/accounting association.
    pub fn complete_solve(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
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
        let workload = self.preparation.workload();
        self.tail = Some(SteadyStudyTail::new(SteadyStudySource {
            host: task.host.clone(),
            result: result.objects.clone(),
            state: crate::steady_study::TerminalStatePublicationSource::Profiles {
                profiles: Box::new(result.source_profiles.clone()),
                kind: SemanticStatePublicationSource::StaticElectricalNativeFields,
                topology: Digest::from_bytes(*self.preparation.mesh.canonical_digest.bytes()),
            },
            receipt: result.numerical_receipt_objects.clone(),
            context_inputs: task.inputs.iter().filter(|input| matches!(input.manifest.kind,
                runmat_analysis_fea::execution::AnalysisNumericalObjectKind::ElectricalStaticWorkload
                | runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Mesh))
                .cloned().chain(std::iter::once(result.objects.clone())).collect(),
            accuracy_kind: AccuracySourceKind::ElectricalStatic,
            sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
            step: self.preparation.selected_step().clone(),
            formulation_requirement: workload.formulation_requirement,
            observations: None,
            program_revision: self.preparation.program_revision.clone(),
            output_policy: self.preparation.output_policy,
        }, usage));
        self.result = Some(result);
        self.closures.push(closure);
        self.phase = ElectricalStudyPhase::ReadyForAccuracy;
        Ok(())
    }

    pub fn prepare_accuracy(&mut self) -> AnalysisBridgeResult<&PreparedAccuracy> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("electrical solve has not completed"))?
            .prepare_accuracy()
    }
    pub fn complete_accuracy(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let phase = ElectricalStudyResultPhase::Accuracy;
        let tail = self
            .tail
            .as_mut()
            .ok_or_else(|| invalid("electrical solve has not completed"))?;
        let task = tail
            .accuracy_task
            .as_ref()
            .ok_or_else(|| invalid("electrical phase has not been prepared"))?;
        let closure = ElectricalStudyResultClosure::new(phase, &task.host, &response, &accounting);
        tail.complete_accuracy(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
    pub fn prepare_next_state(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedSemanticStatePublication> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("electrical solve has not completed"))?
            .prepare_next_state()
    }
    pub fn complete_next_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let phase = ElectricalStudyResultPhase::State {
            ordinal: self.states().len(),
        };
        let tail = self
            .tail
            .as_mut()
            .ok_or_else(|| invalid("electrical solve has not completed"))?;
        let task = tail
            .state_task
            .as_ref()
            .ok_or_else(|| invalid("electrical phase has not been prepared"))?;
        let closure = ElectricalStudyResultClosure::new(phase, &task.host, &response, &accounting);
        tail.complete_next_state(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
    pub fn prepare_publication(&mut self) -> AnalysisBridgeResult<&PreparedResultPublication> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("electrical solve has not completed"))?
            .prepare_publication()
    }
    pub fn complete_publication(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let phase = ElectricalStudyResultPhase::Publication;
        let tail = self
            .tail
            .as_mut()
            .ok_or_else(|| invalid("electrical solve has not completed"))?;
        let task = tail
            .publication_task
            .as_ref()
            .ok_or_else(|| invalid("electrical phase has not been prepared"))?;
        let closure = ElectricalStudyResultClosure::new(phase, &task.host, &response, &accounting);
        tail.complete_publication(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
    pub fn freeze_baseline(&mut self) -> AnalysisBridgeResult<&PreparedResourceUsage> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("electrical solve has not completed"))?
            .freeze_baseline()
    }

    pub fn phase(&self) -> ElectricalStudyPhase {
        self.tail
            .as_ref()
            .map_or(self.phase, |tail| match tail.phase {
                SteadyStudyPhase::ReadyForAccuracy => ElectricalStudyPhase::ReadyForAccuracy,
                SteadyStudyPhase::AwaitingAccuracy => ElectricalStudyPhase::AwaitingAccuracy,
                SteadyStudyPhase::ReadyForState { ordinal } => {
                    ElectricalStudyPhase::ReadyForState { ordinal }
                }
                SteadyStudyPhase::AwaitingState { ordinal } => {
                    ElectricalStudyPhase::AwaitingState { ordinal }
                }
                SteadyStudyPhase::ReadyForBaseline => ElectricalStudyPhase::ReadyForBaseline,
                SteadyStudyPhase::ReadyForPublication => ElectricalStudyPhase::ReadyForPublication,
                SteadyStudyPhase::AwaitingPublication => ElectricalStudyPhase::AwaitingPublication,
                SteadyStudyPhase::Complete => ElectricalStudyPhase::Complete,
            })
    }

    pub fn into_completed(self) -> AnalysisBridgeResult<CompletedElectricalStudy> {
        self.require(ElectricalStudyPhase::Complete)?;
        let workload = self.preparation.workload().clone();
        let tail = self.tail.expect("completed shared terminal lifecycle");
        Ok(CompletedElectricalStudy {
            replay: completion::ElectricalStudyReplayContext {
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
            compiled: self.preparation.product.compiled,
            workload,
            result: self.result.expect("completed solve"),
            accuracy: tail.accuracy.expect("completed accuracy"),
            states: tail.states,
            baseline_usage: tail.baseline.expect("frozen baseline"),
            terminal_usage: tail.terminal.expect("completed terminal accounting"),
            publication: tail.publication.expect("completed publication"),
        })
    }
    pub fn is_complete(&self) -> bool {
        self.phase() == ElectricalStudyPhase::Complete
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

    fn require(&self, expected: ElectricalStudyPhase) -> AnalysisBridgeResult<()> {
        if self.phase() != expected {
            return Err(invalid(format!(
                "electrical study phase {:?} does not permit {:?}",
                self.phase(),
                expected
            )));
        }
        Ok(())
    }
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
