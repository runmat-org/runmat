//! Placement-neutral orchestration of an exact authored acoustic frequency plan.
//! Each entry owns its numerical evidence; aggregation is a separately governed task.
use crate::phase_accounting::validate_analysis_phase_accounting as validate_accounting;
pub use crate::sampled_study_observations::history::CompletedEntryHistories;
use crate::sampled_study_observations::history::{EntryHistoryContext, EntryHistoryPhase};
use crate::AnalysisPhaseExecutionEvidence;
use crate::{
    import_acoustic_frequency_assembly, prepare_acoustic_frequency_assembly_task,
    AcousticFrequencyAssemblyPreparation, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisOutputPolicy,
    ImportedAcousticFrequencyAssembly, PlannedResourceUsageGraph,
    PreparedAcousticFrequencyAssemblyTask, PreparedResourceUsage,
};
use runmat_analysis_fea::{
    execution::AcousticFrequencyWorkload,
    study::{
        plan_acoustic_study, AcousticStudyPlanningPolicy, AcousticStudyProduct,
        CompiledAnalysisStudy,
    },
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{cache::CacheImport, ArtifactAccess, ProgramExecutionResponse};
use runmat_meshing_core::SolverMeshArtifact;
use std::{collections::BTreeSet, sync::Arc};
mod completion;
mod entry;
mod root;
pub use completion::{
    import_acoustic_study_completion, AcousticStudyAssemblyClosure, AcousticStudyEntryClosure,
    AcousticStudyResultClosure, AcousticStudyResultPhase,
};
pub use entry::{AcousticStudyEntryCoordinator, AcousticStudyPhase, CompletedAcousticStudyEntry};
pub use root::{
    import_acoustic_study_root, prepare_acoustic_study_root, PreparedAcousticStudyRoot,
};
#[cfg(test)]
mod tests;

pub struct CompletedAcousticStudy {
    replay: completion::AcousticStudyReplayContext,
    pub compiled: CompiledAnalysisStudy,
    pub workload: AcousticFrequencyWorkload,
    pub entries: Vec<CompletedAcousticStudyEntry>,
    /// Histories over the frequencies of the sweep, finalized after the last entry.
    pub histories: Option<CompletedEntryHistories>,
    /// The original checkpoint and finalization task responses of `histories`, in order.
    pub history_closures: Vec<AcousticStudyResultClosure>,
    pub assembly: ImportedAcousticFrequencyAssembly,
    pub resource_usage: PreparedResourceUsage,
    pub assembly_closure: AcousticStudyAssemblyClosure,
    pub wave_wall_millis: u64,
}

pub struct AcousticStudyPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub policy: AcousticStudyPlanningPolicy,
    pub product: AcousticStudyProduct,
    pub mesh: SolverMeshArtifact,
    pub frozen_project_digest: Option<Digest>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

/// Lifecycle of the study-level history phase that runs after every entry completes.
enum AcousticStudyHistoryState {
    NotStarted,
    Running(Box<EntryHistoryPhase>),
    Complete(Option<CompletedEntryHistories>),
}

/// The coordinator admits complete entry evidence before preparing immutable fan-in.
pub struct AcousticStudyCoordinator {
    preparation: Arc<AcousticStudyPreparation>,
    workload: AcousticFrequencyWorkload,
    entries: Vec<AcousticStudyEntryCoordinator>,
    history: AcousticStudyHistoryState,
    history_closures: Vec<AcousticStudyResultClosure>,
    history_attempts: Vec<crate::PlannedResourceUsageAttempt>,
    assembly_task: Option<PreparedAcousticFrequencyAssemblyTask>,
    assembly: Option<ImportedAcousticFrequencyAssembly>,
    assembly_closure: Option<AcousticStudyAssemblyClosure>,
    resource_usage: Option<PreparedResourceUsage>,
    wave_wall_millis: Option<u64>,
}
impl AcousticStudyCoordinator {
    pub fn new(preparation: AcousticStudyPreparation) -> AnalysisBridgeResult<Self> {
        preparation
            .product
            .compiled
            .validate_against(&preparation.mesh)
            .map_err(|error| invalid(error.to_string()))?;
        let expected = plan_acoustic_study(&preparation.product.compiled, preparation.policy)
            .map_err(|error| invalid(error.to_string()))?;
        if expected != preparation.product.plan {
            return Err(invalid(
                "acoustic workload differs from the compiled authored study",
            ));
        }
        let workload = preparation.frequency_workload()?;
        let preparation = Arc::new(preparation);
        let entries = preparation
            .product
            .plan
            .frequencies()
            .iter()
            .map(|planned| {
                AcousticStudyEntryCoordinator::new(Arc::clone(&preparation), planned.entry())
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        Ok(Self {
            preparation,
            workload,
            entries,
            history: AcousticStudyHistoryState::NotStarted,
            history_closures: Vec::new(),
            history_attempts: Vec::new(),
            assembly_task: None,
            assembly: None,
            assembly_closure: None,
            resource_usage: None,
            wave_wall_millis: None,
        })
    }
    pub fn entries(&self) -> &[AcousticStudyEntryCoordinator] {
        &self.entries
    }
    pub fn entries_mut(&mut self) -> AnalysisBridgeResult<&mut [AcousticStudyEntryCoordinator]> {
        if self.assembly_task.is_some()
            || !matches!(self.history, AcousticStudyHistoryState::NotStarted)
        {
            return Err(invalid("entry lifecycle is frozen for assembly"));
        }
        Ok(&mut self.entries)
    }
    pub fn entry_mut(
        &mut self,
        ordinal: usize,
    ) -> AnalysisBridgeResult<&mut AcousticStudyEntryCoordinator> {
        self.entries_mut()?
            .get_mut(ordinal)
            .ok_or_else(|| invalid("acoustic entry ordinal is outside the plan"))
    }
    pub fn workload(&self) -> &AcousticFrequencyWorkload {
        &self.workload
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

    fn history_context<'a>(
        preparation: &'a AcousticStudyPreparation,
        entries: &[AcousticStudyEntryCoordinator],
    ) -> AnalysisBridgeResult<EntryHistoryContext<'a>> {
        let first = entries
            .first()
            .ok_or_else(|| invalid("acoustic study has no entries"))?;
        Ok(EntryHistoryContext {
            producer: first.producer()?,
            step: preparation.selected_step(),
            source: crate::ObservationHistoryFinalizationSource::VolumeMesh {
                mesh: first.mesh_objects()?,
            },
            history_context:
                runmat_analysis_fea::execution::ObservationHistorySourceContext::VolumeMesh {
                    mesh: &preparation.mesh,
                },
            program_revision: &preparation.program_revision,
            chunk_bytes: preparation.output_policy.chunk_bytes(),
            inventory_limits: preparation.output_policy.inventory_limits(),
        })
    }

    /// Prepare the next wave of the study-level history phase, which checkpoints every
    /// history across the complete entries in frequency order and then finalizes it. The
    /// tasks of a wave are independent; each wave depends on the one before. An empty wave
    /// once every history is published (at once when the step declares none).
    pub fn prepare_history(
        &mut self,
    ) -> AnalysisBridgeResult<Vec<crate::SampledObservationTask<'_>>> {
        if self.assembly_task.is_some() || self.entries.iter().any(|entry| !entry.is_complete()) {
            return Err(invalid(
                "acoustic histories require complete, unassembled entry evidence",
            ));
        }
        let preparation = Arc::clone(&self.preparation);
        let context = Self::history_context(&preparation, &self.entries)?;
        if matches!(self.history, AcousticStudyHistoryState::NotStarted) {
            let samples = self
                .entries
                .iter()
                .map(AcousticStudyEntryCoordinator::history_sample)
                .collect::<AnalysisBridgeResult<Vec<_>>>()?;
            self.history = match EntryHistoryPhase::new(&context, samples)? {
                Some(phase) => AcousticStudyHistoryState::Running(Box::new(phase)),
                None => AcousticStudyHistoryState::Complete(None),
            };
        }
        let AcousticStudyHistoryState::Running(phase) = &mut self.history else {
            return Ok(Vec::new());
        };
        if phase.completed().is_none() && !phase.prepare(&context)?.is_empty() {
            return Ok(self.pending_history_tasks());
        }
        let AcousticStudyHistoryState::Running(phase) =
            std::mem::replace(&mut self.history, AcousticStudyHistoryState::NotStarted)
        else {
            unreachable!("history phase is running");
        };
        self.history = AcousticStudyHistoryState::Complete(Some(phase.into_completed()?));
        Ok(Vec::new())
    }

    /// The prepared history wave, in task order.
    pub fn pending_history_tasks(&self) -> Vec<crate::SampledObservationTask<'_>> {
        match &self.history {
            AcousticStudyHistoryState::Running(phase) => phase.pending(),
            _ => Vec::new(),
        }
    }

    /// Complete the prepared history wave with one response and its accounting per task, in
    /// task order.
    pub fn complete_history(
        &mut self,
        source: &impl CacheImport,
        responses: Vec<(ProgramExecutionResponse, AnalysisPhaseExecutionEvidence)>,
    ) -> AnalysisBridgeResult<()> {
        let preparation = Arc::clone(&self.preparation);
        let context = Self::history_context(&preparation, &self.entries)?;
        let AcousticStudyHistoryState::Running(phase) = &mut self.history else {
            return Err(invalid("acoustic history phase is not running"));
        };
        let tasks = phase.pending();
        if tasks.is_empty() || tasks.len() != responses.len() {
            return Err(invalid(
                "acoustic history responses differ from the prepared wave",
            ));
        }
        let mut closures = Vec::with_capacity(tasks.len());
        let mut attempts = Vec::new();
        for (task, (response, accounting)) in tasks.iter().zip(&responses) {
            validate_accounting(task.host, task.resource_plan, response, accounting)?;
            attempts.extend(accounting.resource_graph(task.resource_plan)?.attempts);
            closures.push(AcousticStudyResultClosure::new(
                AcousticStudyResultPhase::History {
                    ordinal: self.history_closures.len() + closures.len(),
                },
                task.host,
                response,
                accounting,
            ));
        }
        drop(tasks);
        phase.complete(
            source,
            responses
                .into_iter()
                .map(|(response, _)| response)
                .collect(),
            &context,
        )?;
        self.history_closures.extend(closures);
        self.history_attempts.extend(attempts);
        Ok(())
    }

    pub fn prepare_assembly(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedAcousticFrequencyAssemblyTask> {
        if !matches!(self.history, AcousticStudyHistoryState::Complete(_)) {
            return Err(invalid(
                "acoustic assembly requires the finalized histories",
            ));
        }
        if self.assembly_task.is_some() || self.entries.iter().any(|entry| !entry.is_complete()) {
            return Err(invalid(
                "acoustic assembly requires complete, unassembled entry evidence",
            ));
        }
        if self.entries.len() != self.preparation.product.plan.frequencies().len()
            || self
                .entries
                .iter()
                .zip(self.preparation.product.plan.frequencies())
                .any(|(entry, planned)| {
                    entry.entry() != planned.entry() || entry.workload() != planned.workload()
                })
        {
            return Err(invalid(
                "acoustic assembly entry inventory differs from exact authored order",
            ));
        }
        let inputs = self
            .entries
            .iter()
            .map(|entry| {
                entry
                    .result()
                    .expect("completed entry result")
                    .objects
                    .clone()
            })
            .collect();
        let request = runmat_analysis_fea::execution::AcousticFrequencyAssemblyRequest::new(
            self.entries
                .iter()
                .map(|entry| entry.observations().map(|value| value.manifest.clone()))
                .collect(),
        )?;
        let task =
            prepare_acoustic_frequency_assembly_task(AcousticFrequencyAssemblyPreparation {
                workload: self.workload.clone(),
                mesh: self.preparation.mesh.clone(),
                entries: inputs,
                request,
                artifact_access: self.preparation.artifact_access.clone(),
                program_revision: self.preparation.program_revision.clone(),
                output_policy: self.preparation.output_policy,
                authored_study_digest: self
                    .preparation
                    .product
                    .compiled
                    .definition_digest()
                    .map_err(|error| invalid(error.to_string()))?,
                frozen_project_digest: self.preparation.frozen_project_digest,
            })?;
        self.assembly_task = Some(task);
        Ok(self.assembly_task.as_ref().expect("just prepared assembly"))
    }

    pub fn complete_assembly(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AnalysisPhaseExecutionEvidence,
        wave_wall_millis: u64,
    ) -> AnalysisBridgeResult<()> {
        if self.assembly.is_some() {
            return Err(invalid("acoustic assembly is already complete"));
        }
        let task = self
            .assembly_task
            .as_ref()
            .ok_or_else(|| invalid("acoustic assembly was not prepared"))?;
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = AcousticStudyAssemblyClosure {
            workload: task.host.workload.clone(),
            response: response.clone(),
            execution: accounting.clone(),
        };
        let assembly = import_acoustic_frequency_assembly(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        assembly
            .result
            .validate_for(&self.workload, &self.preparation.mesh)?;
        let mut attempts: Vec<_> = self
            .entries
            .iter()
            .flat_map(|entry| entry.resource_attempts().iter().cloned())
            .collect();
        attempts.extend(self.history_attempts.iter().cloned());
        attempts.extend(accounting.resource_graph(&task.resource_plan)?.attempts);
        let graph = PlannedResourceUsageGraph::new(attempts, wave_wall_millis)?;
        let usage = crate::prepare_planned_resource_graph_usage(
            &task.resource_plan,
            task.host.workload.identity.digest(),
            assembly.result.source_result_identity()?.digest(),
            &graph,
            &self.preparation.artifact_access,
            self.preparation.output_policy.chunk_bytes(),
            self.preparation.output_policy.inventory_limits(),
        )?;
        self.assembly = Some(assembly);
        self.assembly_closure = Some(closure);
        self.resource_usage = Some(usage);
        self.wave_wall_millis = Some(wave_wall_millis);
        Ok(())
    }
    pub fn is_complete(&self) -> bool {
        self.assembly.is_some()
    }
    pub fn into_completed(self) -> AnalysisBridgeResult<CompletedAcousticStudy> {
        if !self.is_complete() {
            return Err(invalid("acoustic study is not complete"));
        }
        let entries = self
            .entries
            .into_iter()
            .map(AcousticStudyEntryCoordinator::into_completed)
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        let preparation = Arc::try_unwrap(self.preparation)
            .map_err(|_| invalid("acoustic preparation still has lifecycle owners"))?;
        Ok(CompletedAcousticStudy {
            replay: completion::AcousticStudyReplayContext {
                local_execution: preparation.local_execution,
                policy: preparation.policy,
                mesh: preparation.mesh,
                frozen_project_digest: preparation.frozen_project_digest,
                artifact_access: preparation.artifact_access,
                program_revision: preparation.program_revision,
                output_policy: preparation.output_policy,
                required_capabilities: preparation.required_capabilities,
            },
            compiled: preparation.product.compiled,
            workload: self.workload,
            entries,
            histories: match self.history {
                AcousticStudyHistoryState::Complete(histories) => histories,
                _ => return Err(invalid("acoustic histories are not finalized")),
            },
            history_closures: self.history_closures,
            assembly: self.assembly.expect("completed assembly"),
            resource_usage: self.resource_usage.expect("completed assembly accounting"),
            assembly_closure: self.assembly_closure.expect("completed assembly closure"),
            wave_wall_millis: self.wave_wall_millis.expect("completed wave accounting"),
        })
    }
}
impl AcousticStudyPreparation {
    pub(super) fn frequency_workload(&self) -> AnalysisBridgeResult<AcousticFrequencyWorkload> {
        let template = self
            .product
            .plan
            .frequencies()
            .iter()
            .find(|entry| {
                entry.entry().frequency_hz()
                    == self.product.plan.frequency_plan().reference_frequency_hz()
            })
            .ok_or_else(|| invalid("acoustic frequency plan omits its authored reference"))?
            .workload()
            .clone();
        let workload = AcousticFrequencyWorkload::new(template)?;
        if workload.plan() != self.product.plan.frequency_plan() {
            return Err(invalid(
                "aggregate acoustic frequency plan differs from authored planning",
            ));
        }
        for planned in self.product.plan.frequencies() {
            if workload.entry(planned.entry())? != *planned.workload() {
                return Err(invalid(
                    "acoustic entry workload differs from authored plan",
                ));
            }
        }
        Ok(workload)
    }
    pub(super) fn selected_step(&self) -> &runmat_analysis_core::ProcedureStep {
        &self.product.compiled.model.sequence.steps[0]
    }
}
fn invalid(message: impl Into<String>) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
