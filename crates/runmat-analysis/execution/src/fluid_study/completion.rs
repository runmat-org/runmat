//! Ordered, metered completion replay without solver or worker-task re-execution.
//! Existing import-time source and numerical receipt consistency checks still run.
use super::*;
use crate::AnalysisWorkloadRequest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum FluidStudyResultPhase {
    Solve,
    FieldRecovery,
    Observation,
    Accuracy,
    State { ordinal: usize },
    Publication,
}

/// Preserves the original scheduler association; this is not a new scheduler attestation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidStudyResultClosure {
    pub phase: FluidStudyResultPhase,
    pub workload: AnalysisWorkloadRequest,
    pub response: ProgramExecutionResponse,
    pub execution: AnalysisPhaseExecutionEvidence,
}

impl FluidStudyResultClosure {
    pub(super) fn new(
        phase: FluidStudyResultPhase,
        host: &AnalysisHostWorkload,
        response: &ProgramExecutionResponse,
        accounting: &AnalysisPhaseExecutionEvidence,
    ) -> Self {
        Self {
            phase,
            workload: host.workload.clone(),
            response: response.clone(),
            execution: accounting.clone(),
        }
    }

    fn validate_for(
        &self,
        phase: FluidStudyResultPhase,
        host: &AnalysisHostWorkload,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != phase || self.workload != host.workload {
            return Err(AnalysisBridgeError::Identity(
                "fluid completion phase or workload differs from its reconstructed task",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct FluidStudyReplayContext {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub policy: FluidStudyPlanningPolicy,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

impl CompletedFluidStudy {
    /// The original field-recovery admission. The coordinator that completed this study
    /// (live, or replaying a decoded one) derived it from the validated solve; it is kept,
    /// never re-derived by replaying the solve import.
    pub fn field_recovery_resource_plan(
        &self,
    ) -> &runmat_analysis_fea::planning::resource::FeaResourcePlan {
        &self.field_recovery_plan
    }

    pub fn policy(&self) -> FluidStudyPlanningPolicy {
        self.replay.policy
    }

    pub fn result_closures(&self) -> &[FluidStudyResultClosure] {
        &self.closures
    }

    /// Original validated preparation, retained for durable artifact packaging.
    pub fn preparation(&self) -> AnalysisBridgeResult<FluidStudyPreparation> {
        let preparation = self.assemble_preparation();
        // The sealed product binds its compiled study and workload to the mesh it was
        // planned on; authenticate the retained mesh and that binding.
        let preparation = PlannedFluidStudyPreparation::verify(preparation)?.into_preparation();
        self.result.numerical_receipt.validate_against(
            self.product.workload(),
            &self.result.result,
            &self.replay.mesh,
        )?;
        Ok(preparation)
    }

    /// The preparation from this study's retained parts, without validating them.
    pub(super) fn assemble_preparation(&self) -> FluidStudyPreparation {
        FluidStudyPreparation {
            local_execution: self.replay.local_execution,
            policy: self.replay.policy,
            product: self.product.clone(),
            mesh: self.replay.mesh.clone(),
            frozen_project_digest: self.replay.frozen_project_digest,
            artifact_access: self.replay.artifact_access.clone(),
            program_revision: self.replay.program_revision.clone(),
            output_policy: self.replay.output_policy,
            required_capabilities: self.replay.required_capabilities.clone(),
        }
    }
}

/// Reconstruct every task from authored inputs and consume its saved response/accounting pair.
/// Completion reuses the source imports, admission and terminal usage checks.
pub fn import_fluid_study_completion(
    source: &impl CacheImport,
    preparation: FluidStudyPreparation,
    closures: Vec<FluidStudyResultClosure>,
) -> AnalysisBridgeResult<CompletedFluidStudy> {
    let mut study = FluidStudyCoordinator::new(preparation)?;
    let mut closures = closures.into_iter();
    loop {
        if study.phase() == FluidStudyPhase::ReadyForBaseline {
            study.freeze_baseline()?;
        }
        if study.phase() == FluidStudyPhase::Complete {
            break;
        }
        let (phase, host) = match study.phase() {
            FluidStudyPhase::ReadyForSolve => {
                (FluidStudyResultPhase::Solve, &study.prepare_solve()?.host)
            }
            FluidStudyPhase::ReadyForFieldRecovery => (
                FluidStudyResultPhase::FieldRecovery,
                &study.prepare_field_recovery()?.host,
            ),
            FluidStudyPhase::ReadyForObservations => {
                let Some(task) = study.prepare_next_observation(source)? else {
                    continue;
                };
                (FluidStudyResultPhase::Observation, task.host)
            }
            FluidStudyPhase::ReadyForAccuracy => (
                FluidStudyResultPhase::Accuracy,
                &study.prepare_accuracy()?.host,
            ),
            FluidStudyPhase::ReadyForState { ordinal } => (
                FluidStudyResultPhase::State { ordinal },
                &study.prepare_next_state()?.host,
            ),
            FluidStudyPhase::ReadyForPublication => (
                FluidStudyResultPhase::Publication,
                &study.prepare_publication()?.host,
            ),
            _ => return Err(invalid("fluid completion replay reached an awaiting phase")),
        };
        let closure = closures
            .next()
            .ok_or_else(|| invalid("fluid completion omits an authored phase"))?;
        closure.validate_for(phase, host)?;
        match phase {
            FluidStudyResultPhase::Solve => {
                study.complete_solve(source, closure.response, closure.execution)?
            }
            FluidStudyResultPhase::FieldRecovery => {
                study.complete_field_recovery(source, closure.response, closure.execution)?
            }
            FluidStudyResultPhase::Observation => {
                study.complete_observation(source, closure.response, closure.execution)?
            }
            FluidStudyResultPhase::Accuracy => {
                study.complete_accuracy(source, closure.response, closure.execution)?
            }
            FluidStudyResultPhase::State { .. } => {
                study.complete_next_state(source, closure.response, closure.execution)?
            }
            FluidStudyResultPhase::Publication => {
                study.complete_publication(source, closure.response, closure.execution)?
            }
        }
    }
    if closures.next().is_some() {
        return Err(invalid("fluid completion contains an extra phase"));
    }
    study.into_completed()
}
