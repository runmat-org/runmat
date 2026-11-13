//! Placement-neutral orchestration of a fresh authored standalone static acoustic response step.
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
        plan_acoustic_response_study, AcousticResponseStudyPlanningPolicy,
        AcousticResponseStudyProduct, CompiledAnalysisStudy,
    },
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, ArtifactAccess, ProgramExecutionResponse};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    import_acoustic_response_result, prepare_acoustic_response_task,
    AcousticResponseTaskPreparation, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisOutputPolicy, ImportedAccuracy,
    ImportedAcousticResponseResult, ImportedPublishedResult, ImportedSemanticState,
    PreparedAccuracy, PreparedAcousticResponseTask, PreparedResourceUsage,
    PreparedResultPublication, PreparedSemanticStatePublication,
};

mod completion;
mod observations;
mod root;
pub use completion::{
    import_acoustic_response_study_completion, AcousticResponseStudyResultClosure,
    AcousticResponseStudyResultPhase,
};
pub use root::{
    import_acoustic_response_study_root, prepare_acoustic_response_study_root,
    PreparedAcousticResponseStudyRoot,
};

/// Owned terminal artifacts for the exact standalone authored static envelope.
/// Consuming the coordinator does not clone numerical storage.
pub struct CompletedAcousticResponseStudy {
    replay: completion::AcousticResponseStudyReplayContext,
    closures: Vec<AcousticResponseStudyResultClosure>,
    pub compiled: CompiledAnalysisStudy,
    pub workload: runmat_analysis_fea::execution::AcousticResponseWorkload,
    pub result: ImportedAcousticResponseResult,
    pub fields: Option<crate::ImportedRecoveredFields>,
    pub observations: crate::CompletedSampledObservations,
    pub accuracy: ImportedAccuracy,
    pub states: Vec<ImportedSemanticState>,
    pub baseline_usage: PreparedResourceUsage,
    pub terminal_usage: PreparedResourceUsage,
    pub publication: ImportedPublishedResult,
}

pub struct AcousticResponseStudyPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub policy: AcousticResponseStudyPlanningPolicy,
    pub product: AcousticResponseStudyProduct,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AcousticResponseStudyPhase {
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
pub struct AcousticResponseStudyCoordinator {
    closures: Vec<AcousticResponseStudyResultClosure>,
    preparation: AcousticResponseStudyPreparation,
    definition_digest: Digest,
    phase: AcousticResponseStudyPhase,
    solve_task: Option<PreparedAcousticResponseTask>,
    result: Option<ImportedAcousticResponseResult>,
    recovery_task: Option<crate::PreparedAcousticResponseFieldRecovery>,
    fields: Option<crate::ImportedRecoveredFields>,
    observations: crate::sampled_history_observations::SampledHistoryWorkflow,
    usage: Option<PreparedResourceUsage>,
    tail: Option<SteadyStudyTail>,
}

impl AcousticResponseStudyCoordinator {
    pub fn new(preparation: AcousticResponseStudyPreparation) -> AnalysisBridgeResult<Self> {
        preparation
            .product
            .compiled
            .validate_against(&preparation.mesh)
            .map_err(|error| invalid(error.to_string()))?;
        let expected =
            plan_acoustic_response_study(&preparation.product.compiled, preparation.policy)
                .map_err(|error| invalid(error.to_string()))?;
        if expected != preparation.product.workload {
            return Err(AnalysisBridgeError::Identity(
                "acoustic response workload differs from the compiled authored study",
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
            phase: AcousticResponseStudyPhase::ReadyForSolve,
            solve_task: None,
            result: None,
            recovery_task: None,
            fields: None,
            observations: crate::sampled_history_observations::SampledHistoryWorkflow::default(),
            usage: None,
            tail: None,
        })
    }

    pub fn prepare_solve(&mut self) -> AnalysisBridgeResult<&PreparedAcousticResponseTask> {
        self.require(AcousticResponseStudyPhase::ReadyForSolve)?;
        let task = prepare_acoustic_response_task(AcousticResponseTaskPreparation {
            authored_study_digest: self.definition_digest,
            frozen_project_digest: self.preparation.frozen_project_digest,
            program_revision: self.preparation.program_revision.clone(),
            artifact_access: self.preparation.artifact_access.clone(),
            workload: self.preparation.workload().clone(),
            mesh: self.preparation.mesh.clone(),
            output_policy: self.preparation.output_policy,
            required_capabilities: self.preparation.required_capabilities.clone(),
            local_execution: self.preparation.local_execution,
        })?;
        self.solve_task = Some(task);
        self.phase = AcousticResponseStudyPhase::AwaitingSolve;
        Ok(self.solve_task.as_ref().expect("just prepared"))
    }

    /// The caller must preserve the scheduler's response/accounting association.
    pub fn complete_solve(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        self.require(AcousticResponseStudyPhase::AwaitingSolve)?;
        let task = self.solve_task.as_ref().expect("awaited solve");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = AcousticResponseStudyResultClosure::new(
            AcousticResponseStudyResultPhase::Solve,
            &task.host,
            &response,
            &accounting,
        );
        let result = import_acoustic_response_result(
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
        self.usage = Some(usage);
        self.result = Some(result);
        if self.preparation.selected_step().observations.is_empty() {
            self.prepare_tail(None)?;
            self.phase = AcousticResponseStudyPhase::ReadyForAccuracy;
        } else {
            self.phase = AcousticResponseStudyPhase::ReadyForFieldRecovery;
        }
        self.closures.push(closure);
        Ok(())
    }

    fn prepare_tail(
        &mut self,
        observations: Option<
            runmat_analysis_fea::results::observation::ObservationPublicationCatalog,
        >,
    ) -> AnalysisBridgeResult<()> {
        let task = self.solve_task.as_ref().expect("completed solve");
        let result = self.result.as_ref().expect("completed result");
        let workload = self.preparation.workload();
        self.tail = Some(SteadyStudyTail::new(SteadyStudySource {
            host: task.host.clone(),
            result: result.objects.clone(),
            state: crate::steady_study::TerminalStatePublicationSource::Profiles {
                profiles: Box::new(result.source_profiles.clone()),
                kind: SemanticStatePublicationSource::AcousticResponseFields,
                topology: Digest::from_bytes(*self.preparation.mesh.canonical_digest.bytes()),
            },
            receipt: result.numerical_receipt_objects.clone(),
            context_inputs: task.inputs.iter().filter(|input| matches!(input.manifest.kind,
                runmat_analysis_fea::execution::AnalysisNumericalObjectKind::AcousticResponseWorkload
                | runmat_analysis_fea::execution::AnalysisNumericalObjectKind::Mesh))
                .cloned().chain(std::iter::once(result.objects.clone())).collect(),
            accuracy_kind: AccuracySourceKind::AcousticResponse,
            sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence)?,
            step: self.preparation.selected_step().clone(),
            observations,
            program_revision: self.preparation.program_revision.clone(),
            output_policy: self.preparation.output_policy,
        }, self.usage.clone().expect("completed solve usage")));
        Ok(())
    }

    pub fn prepare_accuracy(&mut self) -> AnalysisBridgeResult<&PreparedAccuracy> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("acoustic response solve has not completed"))?
            .prepare_accuracy()
    }
    pub fn complete_accuracy(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let phase = AcousticResponseStudyResultPhase::Accuracy;
        let tail = self
            .tail
            .as_mut()
            .ok_or_else(|| invalid("acoustic response solve has not completed"))?;
        let task = tail
            .accuracy_task
            .as_ref()
            .ok_or_else(|| invalid("acoustic response phase has not been prepared"))?;
        let closure =
            AcousticResponseStudyResultClosure::new(phase, &task.host, &response, &accounting);
        tail.complete_accuracy(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
    pub fn prepare_next_state(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedSemanticStatePublication> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("acoustic response solve has not completed"))?
            .prepare_next_state()
    }
    pub fn complete_next_state(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let phase = AcousticResponseStudyResultPhase::State {
            ordinal: self.states().len(),
        };
        let tail = self
            .tail
            .as_mut()
            .ok_or_else(|| invalid("acoustic response solve has not completed"))?;
        let task = tail
            .state_task
            .as_ref()
            .ok_or_else(|| invalid("acoustic response phase has not been prepared"))?;
        let closure =
            AcousticResponseStudyResultClosure::new(phase, &task.host, &response, &accounting);
        tail.complete_next_state(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
    pub fn prepare_publication(&mut self) -> AnalysisBridgeResult<&PreparedResultPublication> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("acoustic response solve has not completed"))?
            .prepare_publication()
    }
    pub fn complete_publication(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
    ) -> AnalysisBridgeResult<()> {
        let phase = AcousticResponseStudyResultPhase::Publication;
        let tail = self
            .tail
            .as_mut()
            .ok_or_else(|| invalid("acoustic response solve has not completed"))?;
        let task = tail
            .publication_task
            .as_ref()
            .ok_or_else(|| invalid("acoustic response phase has not been prepared"))?;
        let closure =
            AcousticResponseStudyResultClosure::new(phase, &task.host, &response, &accounting);
        tail.complete_publication(source, response, accounting)?;
        self.closures.push(closure);
        Ok(())
    }
    pub fn freeze_baseline(&mut self) -> AnalysisBridgeResult<&PreparedResourceUsage> {
        self.tail
            .as_mut()
            .ok_or_else(|| invalid("acoustic response solve has not completed"))?
            .freeze_baseline()
    }

    pub fn phase(&self) -> AcousticResponseStudyPhase {
        self.tail
            .as_ref()
            .map_or(self.phase, |tail| match tail.phase {
                SteadyStudyPhase::ReadyForAccuracy => AcousticResponseStudyPhase::ReadyForAccuracy,
                SteadyStudyPhase::AwaitingAccuracy => AcousticResponseStudyPhase::AwaitingAccuracy,
                SteadyStudyPhase::ReadyForState { ordinal } => {
                    AcousticResponseStudyPhase::ReadyForState { ordinal }
                }
                SteadyStudyPhase::AwaitingState { ordinal } => {
                    AcousticResponseStudyPhase::AwaitingState { ordinal }
                }
                SteadyStudyPhase::ReadyForBaseline => AcousticResponseStudyPhase::ReadyForBaseline,
                SteadyStudyPhase::ReadyForPublication => {
                    AcousticResponseStudyPhase::ReadyForPublication
                }
                SteadyStudyPhase::AwaitingPublication => {
                    AcousticResponseStudyPhase::AwaitingPublication
                }
                SteadyStudyPhase::Complete => AcousticResponseStudyPhase::Complete,
            })
    }

    pub fn into_completed(self) -> AnalysisBridgeResult<CompletedAcousticResponseStudy> {
        self.require(AcousticResponseStudyPhase::Complete)?;
        let workload = self.preparation.workload().clone();
        let tail = self.tail.expect("completed shared terminal lifecycle");
        Ok(CompletedAcousticResponseStudy {
            replay: completion::AcousticResponseStudyReplayContext {
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
            fields: self.fields,
            observations: self.observations.into_completed(),
            accuracy: tail.accuracy.expect("completed accuracy"),
            states: tail.states,
            baseline_usage: tail.baseline.expect("frozen baseline"),
            terminal_usage: tail.terminal.expect("completed terminal accounting"),
            publication: tail.publication.expect("completed publication"),
        })
    }
    pub fn is_complete(&self) -> bool {
        self.phase() == AcousticResponseStudyPhase::Complete
    }
    pub fn policy(&self) -> AcousticResponseStudyPlanningPolicy {
        self.preparation.policy
    }
    pub fn selected_step(&self) -> &runmat_analysis_core::ProcedureStep {
        self.preparation.selected_step()
    }
    pub fn compiled_study(&self) -> &CompiledAnalysisStudy {
        &self.preparation.product.compiled
    }
    pub fn result(&self) -> Option<&ImportedAcousticResponseResult> {
        self.result.as_ref()
    }
    pub fn fields(&self) -> Option<&crate::ImportedRecoveredFields> {
        self.fields.as_ref()
    }
    pub fn observations(&self) -> &crate::CompletedSampledObservations {
        self.observations.completed()
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

    fn require(&self, expected: AcousticResponseStudyPhase) -> AnalysisBridgeResult<()> {
        if self.phase() != expected {
            return Err(invalid(format!(
                "acoustic response study phase {:?} does not permit {:?}",
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

impl AcousticResponseStudyPreparation {
    pub(super) fn workload(&self) -> &runmat_analysis_fea::execution::AcousticResponseWorkload {
        &self.product.workload
    }
    pub(super) fn selected_step(&self) -> &runmat_analysis_core::ProcedureStep {
        &self.product.compiled.model.sequence.steps[0]
    }
}
