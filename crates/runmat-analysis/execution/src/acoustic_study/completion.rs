//! Ordered, metered completion replay without solver or worker-task re-execution.
//! Existing import-time source and numerical receipt consistency checks still run.
use super::*;
use crate::AnalysisWorkloadRequest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum AcousticStudyResultPhase {
    Solve,
    FieldRecovery,
    ObservationPartition { ordinal: usize },
    ObservationReduction { ordinal: usize },
    ObservationAssembly,
    Accuracy,
    State { ordinal: usize },
    Publication,
    /// One checkpoint or finalization task of the study-level history phase, in order.
    History { ordinal: usize },
}

/// Preserves the original scheduler association; this is not a new scheduler attestation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticStudyResultClosure {
    pub phase: AcousticStudyResultPhase,
    pub workload: AnalysisWorkloadRequest,
    pub response: ProgramExecutionResponse,
    pub execution: AnalysisPhaseExecutionEvidence,
}

/// Canonical frequency coordinate scopes every saved entry phase.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticStudyEntryClosure {
    pub entry: runmat_analysis_fea::frequency::FrequencyResponsePlanEntry,
    pub closures: Vec<AcousticStudyResultClosure>,
}

/// Original metered aggregate task response; never substitutes a numerical receipt.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticStudyAssemblyClosure {
    pub workload: AnalysisWorkloadRequest,
    pub response: ProgramExecutionResponse,
    pub execution: AnalysisPhaseExecutionEvidence,
}

impl AcousticStudyResultClosure {
    pub(super) fn new(
        phase: AcousticStudyResultPhase,
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
        phase: AcousticStudyResultPhase,
        host: &AnalysisHostWorkload,
    ) -> AnalysisBridgeResult<()> {
        if self.phase != phase || self.workload != host.workload {
            return Err(AnalysisBridgeError::Identity(
                "acoustic completion phase or workload differs from its reconstructed task",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct AcousticStudyReplayContext {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub policy: AcousticStudyPlanningPolicy,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

impl CompletedAcousticStudy {
    pub fn policy(&self) -> AcousticStudyPlanningPolicy {
        self.replay.policy
    }

    pub fn entry_closures(&self) -> Vec<AcousticStudyEntryClosure> {
        self.entries
            .iter()
            .map(|entry| AcousticStudyEntryClosure {
                entry: entry.entry,
                closures: entry.closures.clone(),
            })
            .collect()
    }

    /// Reconstruct exact authored planning and validate every original entry receipt.
    pub fn preparation(&self) -> AnalysisBridgeResult<AcousticStudyPreparation> {
        let entry_attempts = self.entries.iter().try_fold(0_usize, |total, entry| {
            total
                .checked_add(entry.closures.len())
                .ok_or_else(|| invalid("acoustic entry attempt inventory overflow"))
        })?;
        if entry_attempts
            .checked_add(self.history_closures.len())
            .and_then(|total| total.checked_add(1))
            != Some(self.resource_usage.evidence.attempts.len())
        {
            return Err(invalid("acoustic aggregate resource inventory omits or adds an actual entry/assembly attempt"));
        }
        self.compiled
            .validate_against(&self.replay.mesh)
            .map_err(|e| invalid(e.to_string()))?;
        let plan = plan_acoustic_study(&self.compiled, self.replay.policy)
            .map_err(|e| invalid(e.to_string()))?;
        if plan.frequency_plan() != self.workload.plan()
            || plan.frequencies().len() != self.entries.len()
        {
            return Err(invalid(
                "completed acoustic frequencies differ from exact authored planning",
            ));
        }
        for (planned, entry) in plan.frequencies().iter().zip(&self.entries) {
            if planned.entry() != entry.entry || planned.workload() != &entry.workload {
                return Err(invalid(
                    "completed acoustic entry differs from its exact authored workload",
                ));
            }
            entry.result.numerical_receipt.validate_against(
                &entry.workload,
                &entry.result.result,
                &self.replay.mesh,
            )?;
        }
        self.assembly
            .result
            .validate_for(&self.workload, &self.replay.mesh)?;
        self.assembly
            .publication
            .validate_for(&self.assembly.result.publication_authority()?)?;
        let preparation = AcousticStudyPreparation {
            local_execution: self.replay.local_execution,
            policy: self.replay.policy,
            product: AcousticStudyProduct {
                compiled: self.compiled.clone(),
                plan,
            },
            mesh: self.replay.mesh.clone(),
            frozen_project_digest: self.replay.frozen_project_digest,
            artifact_access: self.replay.artifact_access.clone(),
            program_revision: self.replay.program_revision.clone(),
            output_policy: self.replay.output_policy,
            required_capabilities: self.replay.required_capabilities.clone(),
        };
        if preparation.frequency_workload()? != self.workload {
            return Err(invalid(
                "completed acoustic aggregate workload differs from its authored preparation",
            ));
        }
        Ok(preparation)
    }
}

/// Reconstruct every entry task and aggregate task from authored inputs. Saved scheduler
/// response/accounting associations are imported, never reexecuted or relabelled.
pub fn import_acoustic_study_completion(
    source: &impl CacheImport,
    preparation: AcousticStudyPreparation,
    entries: Vec<AcousticStudyEntryClosure>,
    histories: Vec<AcousticStudyResultClosure>,
    assembly: AcousticStudyAssemblyClosure,
    wave_wall_millis: u64,
) -> AnalysisBridgeResult<CompletedAcousticStudy> {
    let mut study = AcousticStudyCoordinator::new(preparation)?;
    if study.entries().len() != entries.len() {
        return Err(invalid(
            "acoustic completion omits or adds frequency entries",
        ));
    }
    for (ordinal, saved) in entries.into_iter().enumerate() {
        let entry = study.entry_mut(ordinal)?;
        if entry.entry() != saved.entry {
            return Err(invalid(
                "acoustic completion frequency order or coordinate differs from exact plan",
            ));
        }
        let mut closures = saved.closures.into_iter();
        loop {
            if entry.phase() == AcousticStudyPhase::Complete {
                break;
            }
            entry.prepare_next(source)?;
            let phase = match entry.phase() {
                AcousticStudyPhase::AwaitingSolve => AcousticStudyResultPhase::Solve,
                AcousticStudyPhase::AwaitingFieldRecovery => {
                    AcousticStudyResultPhase::FieldRecovery
                }
                AcousticStudyPhase::AwaitingObservationPartition { ordinal } => {
                    AcousticStudyResultPhase::ObservationPartition { ordinal }
                }
                AcousticStudyPhase::AwaitingObservationReduction { ordinal } => {
                    AcousticStudyResultPhase::ObservationReduction { ordinal }
                }
                AcousticStudyPhase::AwaitingObservationAssembly => {
                    AcousticStudyResultPhase::ObservationAssembly
                }
                AcousticStudyPhase::AwaitingAccuracy => AcousticStudyResultPhase::Accuracy,
                AcousticStudyPhase::AwaitingState { ordinal } => {
                    AcousticStudyResultPhase::State { ordinal }
                }
                AcousticStudyPhase::AwaitingPublication => AcousticStudyResultPhase::Publication,
                _ => {
                    return Err(invalid(
                        "acoustic completion replay reached a phase without a task",
                    ))
                }
            };
            let closure = closures
                .next()
                .ok_or_else(|| invalid("acoustic entry completion omits an authored phase"))?;
            closure.validate_for(phase, entry.pending_task()?.0)?;
            entry.complete_next(source, closure.response, closure.execution)?;
        }
        if closures.next().is_some() {
            return Err(invalid("acoustic entry completion contains an extra phase"));
        }
    }
    let mut histories = histories.into_iter();
    let mut ordinal = 0;
    loop {
        let hosts = study
            .prepare_history()?
            .iter()
            .map(|task| task.host.clone())
            .collect::<Vec<_>>();
        if hosts.is_empty() {
            break;
        }
        let mut wave = Vec::with_capacity(hosts.len());
        for host in &hosts {
            let closure = histories
                .next()
                .ok_or_else(|| invalid("acoustic completion omits a history phase"))?;
            closure.validate_for(AcousticStudyResultPhase::History { ordinal }, host)?;
            wave.push((closure.response, closure.execution));
            ordinal += 1;
        }
        study.complete_history(source, wave)?;
    }
    if histories.next().is_some() {
        return Err(invalid("acoustic completion contains an extra history phase"));
    }
    if study.prepare_assembly()?.host.workload != assembly.workload {
        return Err(invalid(
            "acoustic assembly closure differs from reconstructed aggregate task",
        ));
    }
    study.complete_assembly(
        source,
        assembly.response,
        assembly.execution,
        wave_wall_millis,
    )?;
    study.into_completed()
}
