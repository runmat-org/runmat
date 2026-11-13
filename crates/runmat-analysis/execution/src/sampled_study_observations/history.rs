//! Study-level phase of a sampled study with several entries (one solve per frequency of a
//! sweep, for example). Each entry evaluates the observations its sample schedules and
//! publishes its field outputs in its own `Entry` catalog. A history spans entries, so no
//! entry can discharge it: this phase checkpoints every history across the entries in sample
//! order, finalizes it after the last entry against an entry-history field catalog, and
//! publishes the finalized histories in one `EntryHistories` catalog. Work runs in waves:
//! one wave per entry holds that entry's checkpoints (one per history it schedules), and a
//! last wave finalizes every history. Tasks in a wave are independent; each wave depends on
//! the one before. Coordination only; numerical work runs in the shared checkpoint and
//! finalization tasks.
use super::invalid;
use crate::sampled_history_observations::SampledObservationTask;
use crate::{
    import_observation_history_checkpoint_task, import_observation_history_finalization_task,
    prepare_analysis_objects, prepare_observation_history_checkpoint_task,
    prepare_observation_history_finalization_task, AnalysisBridgeResult, AnalysisHostWorkload,
    ImportedObservationHistoryCheckpointTask, ImportedObservationHistoryFinalization,
    ImportedReducedObservation, ObservationHistoryCheckpointPreparation,
    ObservationHistoryFinalizationPreparation, ObservationHistoryFinalizationSource,
    PreparedAnalysisObjects, PreparedObservationHistoryCheckpointTask,
    PreparedObservationHistoryFinalizationTask,
};
use runmat_analysis_core::{ObservationId, ObservationRequest, ProcedureStep};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind as Kind, ObservationHistoryFinalizationRequest,
        ObservationHistorySourceContext, StepIdentity,
    },
    results::observation::{
        ObservationFieldCatalog, ObservationPublicationCatalog, ObservationPublicationEntry,
        ObservationPublicationScope, ResolvedObservationPlan,
    },
};
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ObjectNamespace, ProgramExecutionResponse,
};
use std::collections::{BTreeMap, VecDeque};

/// One complete entry as the history phase sees it, in increasing sample order.
pub struct EntryHistorySample {
    /// The entry's single-sample field catalog.
    pub catalog: ObservationFieldCatalog,
    /// The plan the entry's observations were resolved against; `None` when the entry
    /// scheduled no observation.
    pub plan: Option<ResolvedObservationPlan>,
    /// The entry's reduced observations with the reduction task that produced each.
    pub reduced: Vec<(ImportedReducedObservation, AnalysisHostWorkload)>,
}

/// The study's source and policy for checkpoint and finalization tasks.
pub struct EntryHistoryContext<'a> {
    pub producer: StepIdentity,
    pub step: &'a ProcedureStep,
    pub source: ObservationHistoryFinalizationSource,
    pub history_context: ObservationHistorySourceContext<'a>,
    pub program_revision: &'a ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

/// The finalized histories of a sampled study and the catalog that publishes them.
#[derive(Clone, Debug, PartialEq)]
pub struct CompletedEntryHistories {
    pub catalog: ObservationPublicationCatalog,
    pub histories: Vec<ImportedObservationHistoryFinalization>,
    /// The terminal checkpoint of each history, aligned with `histories`.
    pub checkpoints: Vec<ImportedObservationHistoryCheckpointTask>,
    /// The transported entry-history field catalog the finalizations read.
    pub source_objects: PreparedAnalysisObjects,
}

enum PendingEntryHistoryTask {
    Checkpoint(Box<PreparedObservationHistoryCheckpointTask>),
    Finalization(Box<PreparedObservationHistoryFinalizationTask>),
}

impl PendingEntryHistoryTask {
    fn view(&self) -> SampledObservationTask<'_> {
        let (host, program, resource_plan) = match self {
            Self::Checkpoint(task) => (&task.host, &task.program, &task.resource_plan),
            Self::Finalization(task) => (&task.host, &task.program, &task.resource_plan),
        };
        SampledObservationTask {
            host,
            program,
            resource_plan,
        }
    }
}

type Checkpointed = (
    ImportedObservationHistoryCheckpointTask,
    AnalysisHostWorkload,
);

pub struct EntryHistoryPhase {
    plan: ResolvedObservationPlan,
    catalog: ObservationFieldCatalog,
    catalog_objects: PreparedAnalysisObjects,
    /// History samples still to checkpoint: one wave per entry, in sample order.
    waves: VecDeque<Vec<(ImportedReducedObservation, AnalysisHostWorkload)>>,
    /// Every history, in authored order; finalized together after the last checkpoint wave.
    histories: Vec<ObservationId>,
    checkpoints: BTreeMap<ObservationId, Checkpointed>,
    pending: Vec<PendingEntryHistoryTask>,
    completed: Option<CompletedEntryHistories>,
}

impl EntryHistoryPhase {
    /// `None` when the authored step declares no history.
    pub fn new(
        context: &EntryHistoryContext<'_>,
        samples: Vec<EntryHistorySample>,
    ) -> AnalysisBridgeResult<Option<Self>> {
        let histories = context
            .step
            .observations
            .iter()
            .filter_map(|request| match request {
                ObservationRequest::History(spec) => Some(spec.observation_id.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if histories.is_empty() {
            return Ok(None);
        }
        let mut plan: Option<ResolvedObservationPlan> = None;
        let mut catalogs = Vec::with_capacity(samples.len());
        let mut waves = VecDeque::new();
        for sample in samples {
            if let Some(entry_plan) = sample.plan {
                match &plan {
                    Some(plan) if *plan != entry_plan => {
                        return Err(invalid("sampled study entries resolved different plans"))
                    }
                    Some(_) => {}
                    None => plan = Some(entry_plan),
                }
            }
            let wave = sample
                .reduced
                .into_iter()
                .filter(|(reduced, _)| histories.contains(&reduced.result.observation_id))
                .collect::<Vec<_>>();
            if !wave.is_empty() {
                waves.push_back(wave);
            }
            catalogs.push(sample.catalog);
        }
        let plan = plan.ok_or_else(|| invalid("no sampled study entry evaluated a history"))?;
        if let Some(id) = histories.iter().find(|id| {
            !waves
                .iter()
                .flatten()
                .any(|(reduced, _)| &reduced.result.observation_id == *id)
        }) {
            return Err(invalid(&format!(
                "history `{}` produced no scheduled sample in any entry",
                id.0
            )));
        }
        let catalog = ObservationFieldCatalog::entry_history(catalogs)?;
        let catalog_objects = prepare_analysis_objects(
            Kind::ObservationFieldCatalog,
            catalog.identity(),
            &catalog.canonical_bytes()?,
            context.chunk_bytes,
            ObjectNamespace::InputValue,
            context.inventory_limits,
        )?;
        Ok(Some(Self {
            plan,
            catalog,
            catalog_objects,
            waves,
            histories,
            checkpoints: BTreeMap::new(),
            pending: Vec::new(),
            completed: None,
        }))
    }

    pub fn completed(&self) -> Option<&CompletedEntryHistories> {
        self.completed.as_ref()
    }
    pub fn into_completed(self) -> AnalysisBridgeResult<CompletedEntryHistories> {
        self.completed
            .ok_or_else(|| invalid("sampled study histories are not finalized"))
    }

    /// The prepared wave, in task order.
    pub fn pending(&self) -> Vec<SampledObservationTask<'_>> {
        self.pending
            .iter()
            .map(PendingEntryHistoryTask::view)
            .collect()
    }

    /// Prepare the next wave of checkpoints or the finalization wave; an empty wave once every
    /// history is published.
    pub fn prepare(
        &mut self,
        context: &EntryHistoryContext<'_>,
    ) -> AnalysisBridgeResult<Vec<SampledObservationTask<'_>>> {
        if !self.pending.is_empty() {
            return Err(invalid("entry-history wave already prepared"));
        }
        if let Some(wave) = self.waves.front() {
            let mut tasks = Vec::with_capacity(wave.len());
            for (sample, host) in wave {
                let id = sample.result.observation_id.clone();
                tasks.push(PendingEntryHistoryTask::Checkpoint(Box::new(
                    prepare_observation_history_checkpoint_task(
                        ObservationHistoryCheckpointPreparation {
                            source_host: host.clone(),
                            plan: self.plan.clone(),
                            observation_id: id.clone(),
                            sample: sample.clone(),
                            previous: self
                                .checkpoints
                                .get(&id)
                                .map(|(checkpoint, _)| checkpoint.objects.clone()),
                            program_revision: context.program_revision.clone(),
                            chunk_bytes: context.chunk_bytes,
                            inventory_limits: context.inventory_limits,
                        },
                    )?,
                )));
            }
            self.pending = tasks;
            return Ok(self.pending());
        }
        if self.completed.is_some() {
            return Ok(Vec::new());
        }
        let mut tasks = Vec::with_capacity(self.histories.len());
        for id in &self.histories {
            let (checkpoint, host) = self
                .checkpoints
                .get(id)
                .ok_or_else(|| invalid("entry history has no terminal checkpoint"))?;
            let request = ObservationHistoryFinalizationRequest::new(
                self.plan.clone(),
                context.producer,
                id.clone(),
                &checkpoint.checkpoint,
                &self.catalog,
                context.history_context,
            )?;
            tasks.push(PendingEntryHistoryTask::Finalization(Box::new(
                prepare_observation_history_finalization_task(
                    ObservationHistoryFinalizationPreparation {
                        source_host: host.clone(),
                        request,
                        checkpoint: checkpoint.objects.clone(),
                        catalog: self.catalog_objects.clone(),
                        source: context.source.clone(),
                        program_revision: context.program_revision.clone(),
                        chunk_bytes: context.chunk_bytes,
                        inventory_limits: context.inventory_limits,
                    },
                )?,
            )));
        }
        self.pending = tasks;
        Ok(self.pending())
    }

    /// Import every response of the prepared wave, in task order. Progress commits only after
    /// the whole wave validates, so a failed import leaves the wave prepared.
    pub fn complete(
        &mut self,
        source: &impl CacheImport,
        responses: Vec<ProgramExecutionResponse>,
        context: &EntryHistoryContext<'_>,
    ) -> AnalysisBridgeResult<()> {
        if self.pending.is_empty() || responses.len() != self.pending.len() {
            return Err(invalid(
                "entry-history responses differ from the prepared wave",
            ));
        }
        let limits = context.inventory_limits;
        let finalizing = matches!(self.pending[0], PendingEntryHistoryTask::Finalization(_));
        if !finalizing {
            let mut imported = Vec::with_capacity(responses.len());
            for (task, response) in self.pending.iter().zip(responses) {
                let PendingEntryHistoryTask::Checkpoint(task) = task else {
                    return Err(invalid("entry-history wave mixes task kinds"));
                };
                imported.push((
                    task.request.observation_id.clone(),
                    import_observation_history_checkpoint_task(source, response, task, limits)?,
                    task.host.clone(),
                ));
            }
            for (id, checkpoint, host) in imported {
                self.checkpoints.insert(id, (checkpoint, host));
            }
            self.waves.pop_front();
            self.pending.clear();
            return Ok(());
        }
        let mut entries = Vec::with_capacity(responses.len());
        let mut histories = Vec::with_capacity(responses.len());
        for ((task, response), id) in self.pending.iter().zip(responses).zip(&self.histories) {
            let PendingEntryHistoryTask::Finalization(task) = task else {
                return Err(invalid("entry-history wave mixes task kinds"));
            };
            let history =
                import_observation_history_finalization_task(source, response, task, limits)?;
            if &history.history.manifest.observation_id != id {
                return Err(invalid("finalized history differs from its authored order"));
            }
            entries.push(ObservationPublicationEntry::History(Box::new(
                history.publication.clone(),
            )));
            histories.push(history);
        }
        let catalog = ObservationPublicationCatalog::scoped(
            ObservationPublicationScope::EntryHistories,
            context.producer,
            self.plan.clone(),
            self.catalog.clone(),
            context.step,
            entries,
            context.history_context,
        )?;
        let checkpoints = self
            .histories
            .iter()
            .map(|id| {
                self.checkpoints
                    .get(id)
                    .map(|(checkpoint, _)| checkpoint.clone())
                    .ok_or_else(|| invalid("entry history has no terminal checkpoint"))
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        self.completed = Some(CompletedEntryHistories {
            catalog,
            histories,
            checkpoints,
            source_objects: self.catalog_objects.clone(),
        });
        self.checkpoints.clear();
        self.pending.clear();
        Ok(())
    }
}
