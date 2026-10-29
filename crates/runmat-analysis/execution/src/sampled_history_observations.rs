//! Sample-by-sample authored observations over a producer's retained field samples: field
//! outputs assemble at the samples they select, histories checkpoint each sample and finalize
//! once, all through the shared observation tasks. Families supply their source through
//! [`SampledHistorySource`]; serial coordination only, never numerical evaluation. Each imported
//! phase is validated before committing progress; prepared work survives failure.
use crate::{
    import_observation_assembly, import_observation_history_checkpoint_task,
    import_observation_history_finalization_task, import_observation_partition_contribution,
    import_reduced_observation, prepare_analysis_objects, prepare_observation_assembly,
    prepare_observation_fanout, prepare_observation_history_checkpoint_task,
    prepare_observation_history_finalization_task, prepare_observation_reduction,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, AnalysisOutputPolicy,
    ImportedObservationHistoryCheckpointTask, ImportedObservationHistoryFinalization,
    ImportedObservationPartitionContribution, ImportedObservations, ImportedReducedObservation,
    ObservationAssemblyPreparation, ObservationHistoryCheckpointPreparation,
    ObservationHistoryFinalizationPreparation, ObservationHistoryFinalizationSource,
    ObservationPreparation, ObservationReductionPreparation, PreparedAnalysisObjects,
    PreparedObservationPartition,
};
use runmat_analysis_core::{ObservationId, ObservationRequest, ProcedureStep};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind as Kind, ObservationEvaluationRequest,
        ObservationHistoryFinalizationRequest, ObservationHistorySourceContext,
    },
    results::{
        field::RecoveredFieldSample,
        observation::{
            ObservationFieldCatalog, ObservationPartitionPlan, ObservationPublicationCatalog,
            ObservationPublicationEntry, ObservationSamplePublication,
            VolumeMeshObservationAuthority,
        },
    },
};
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::{cache::CacheImport, ObjectNamespace, ProgramExecutionResponse};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

mod task;
use task::PendingObservationTask;
pub use task::SampledObservationTask;

/// A producer's retained field samples as the observation workflow sees them.
pub(crate) trait SampledHistorySource {
    fn step(&self) -> &ProcedureStep;
    /// The solve task whose result owns the samples.
    fn solve_host(&self) -> &AnalysisHostWorkload;
    fn program_revision(&self) -> &ProgramRevision;
    fn output_policy(&self) -> AnalysisOutputPolicy;
    /// Retained samples in increasing independent-domain order.
    fn sample_count(&self) -> usize;
    /// The observation source of one retained sample and its coordinate.
    fn observation_preparation(
        &self,
        source: &impl CacheImport,
        ordinal: usize,
    ) -> AnalysisBridgeResult<(ObservationPreparation, RecoveredFieldSample)>;
    /// Field catalog of every retained sample, the authority histories finalize against.
    fn field_catalog(&self) -> AnalysisBridgeResult<ObservationFieldCatalog>;
    fn finalization_source(&self) -> AnalysisBridgeResult<ObservationHistoryFinalizationSource>;
    fn history_context(&self) -> ObservationHistorySourceContext<'_>;
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompletedSampledObservations {
    pub catalog: Option<ObservationPublicationCatalog>,
    pub samples: Vec<ImportedObservations>,
    pub histories: Vec<ImportedObservationHistoryFinalization>,
}

fn is_history(step: &ProcedureStep, id: &ObservationId) -> bool {
    step.observations.iter().any(|observation| {
        matches!(observation,
        ObservationRequest::History(spec) if &spec.observation_id == id)
    })
}

#[derive(Default)]
pub(crate) struct SampledHistoryWorkflow {
    sample_ordinal: usize,
    prepared_sample: Option<usize>,
    source_request: Option<ObservationEvaluationRequest>,
    catalog: Option<ObservationFieldCatalog>,
    /// A volume-mesh source's catalog sealed with the current request's plan.
    volume_mesh: Option<VolumeMeshObservationAuthority>,
    catalog_objects: Option<PreparedAnalysisObjects>,
    partitions: VecDeque<PreparedObservationPartition>,
    plans: BTreeMap<ObservationId, ObservationPartitionPlan>,
    contributions: BTreeMap<ObservationId, Vec<ImportedObservationPartitionContribution>>,
    reduced: BTreeMap<ObservationId, (ImportedReducedObservation, AnalysisHostWorkload)>,
    checkpointed: BTreeSet<ObservationId>,
    checkpoints: BTreeMap<
        ObservationId,
        (
            ImportedObservationHistoryCheckpointTask,
            AnalysisHostWorkload,
        ),
    >,
    pending: Option<PendingObservationTask>,
    entries: Vec<ObservationPublicationEntry>,
    completed: CompletedSampledObservations,
}

impl SampledHistoryWorkflow {
    pub(crate) fn completed(&self) -> &CompletedSampledObservations {
        &self.completed
    }
    pub(crate) fn into_completed(self) -> CompletedSampledObservations {
        self.completed
    }
    pub(crate) fn pending(&self) -> AnalysisBridgeResult<SampledObservationTask<'_>> {
        self.pending
            .as_ref()
            .map(PendingObservationTask::view)
            .ok_or(AnalysisBridgeError::Identity(
                "no pending observation phase",
            ))
    }

    pub(crate) fn prepare<C: SampledHistorySource>(
        &mut self,
        source: &impl CacheImport,
        context: &C,
    ) -> AnalysisBridgeResult<Option<SampledObservationTask<'_>>> {
        if self.pending.is_some() {
            return Err(AnalysisBridgeError::Identity(
                "observation phase already prepared",
            ));
        }
        if context.step().observations.is_empty() {
            return Ok(None);
        }
        let policy = context.output_policy();
        loop {
            if self.sample_ordinal < context.sample_count() {
                if self.prepared_sample != Some(self.sample_ordinal) {
                    self.prepare_sample(source, context)?;
                }
                if let Some(partition) = self.partitions.pop_front() {
                    self.pending = Some(PendingObservationTask::Partition(Box::new(partition)));
                    break;
                }
                if let Some((id, plan)) = self.plans.first_key_value() {
                    let task = prepare_observation_reduction(ObservationReductionPreparation {
                        source_host: context.solve_host().clone(),
                        source_request: self
                            .source_request
                            .as_ref()
                            .expect("sample request")
                            .clone(),
                        partition_plan: plan.clone(),
                        contributions: self.contributions.get(id).cloned().unwrap_or_default(),
                        program_revision: context.program_revision().clone(),
                        chunk_bytes: policy.chunk_bytes(),
                        inventory_limits: policy.inventory_limits(),
                    })?;
                    self.pending = Some(PendingObservationTask::Reduction(Box::new(task)));
                    break;
                }
                if let Some((id, (sample, host))) = self.reduced.iter().find(|(id, _)| {
                    is_history(context.step(), id) && !self.checkpointed.contains(*id)
                }) {
                    let task = prepare_observation_history_checkpoint_task(
                        ObservationHistoryCheckpointPreparation {
                            source_host: host.clone(),
                            plan: self
                                .source_request
                                .as_ref()
                                .expect("sample request")
                                .plan
                                .clone(),
                            observation_id: id.clone(),
                            sample: sample.clone(),
                            previous: self
                                .checkpoints
                                .get(id)
                                .map(|(checkpoint, _)| checkpoint.objects.clone()),
                            program_revision: context.program_revision().clone(),
                            chunk_bytes: policy.chunk_bytes(),
                            inventory_limits: policy.inventory_limits(),
                        },
                    )?;
                    self.pending = Some(PendingObservationTask::Checkpoint(Box::new(task)));
                    break;
                }
                if self
                    .reduced
                    .keys()
                    .any(|id| !is_history(context.step(), id))
                {
                    let task = prepare_observation_assembly(ObservationAssemblyPreparation {
                        source_host: context.solve_host().clone(),
                        source_request: self
                            .source_request
                            .as_ref()
                            .expect("sample request")
                            .clone(),
                        // Assembly preserves exact scheduled coverage, even in mixed field/history studies.
                        results: self
                            .reduced
                            .values()
                            .map(|(sample, _)| sample.clone())
                            .collect(),
                        program_revision: context.program_revision().clone(),
                        chunk_bytes: policy.chunk_bytes(),
                        inventory_limits: policy.inventory_limits(),
                    })?;
                    self.pending = Some(PendingObservationTask::Assembly(Box::new(task)));
                    break;
                }
                self.advance_sample();
                continue;
            }
            if let Some((id, (checkpoint, host))) = self.checkpoints.first_key_value() {
                let catalog = self.catalog.as_ref().expect("source catalog");
                let request = ObservationHistoryFinalizationRequest::new(
                    self.last_plan()?.clone(),
                    context.solve_host().workload.step,
                    id.clone(),
                    &checkpoint.checkpoint,
                    catalog,
                    context.history_context(),
                )?;
                let task = prepare_observation_history_finalization_task(
                    ObservationHistoryFinalizationPreparation {
                        source_host: host.clone(),
                        request,
                        checkpoint: checkpoint.objects.clone(),
                        catalog: self
                            .catalog_objects
                            .as_ref()
                            .expect("catalog objects")
                            .clone(),
                        source: context.finalization_source()?,
                        program_revision: context.program_revision().clone(),
                        chunk_bytes: policy.chunk_bytes(),
                        inventory_limits: policy.inventory_limits(),
                    },
                )?;
                self.pending = Some(PendingObservationTask::Finalization(Box::new(task)));
                break;
            }
            let catalog = match self.volume_mesh.clone() {
                Some(authority) => ObservationPublicationCatalog::on_volume_mesh(
                    context.solve_host().workload.step,
                    authority,
                    context.step(),
                    self.entries.clone(),
                    context.history_context(),
                )?,
                None => ObservationPublicationCatalog::new(
                    context.solve_host().workload.step,
                    self.last_plan()?.clone(),
                    self.catalog.as_ref().expect("source catalog").clone(),
                    context.step(),
                    self.entries.clone(),
                    context.history_context(),
                )?,
            };
            self.completed.catalog = Some(catalog);
            return Ok(None);
        }
        Ok(self.pending.as_ref().map(PendingObservationTask::view))
    }

    fn last_plan(
        &self,
    ) -> AnalysisBridgeResult<&runmat_analysis_fea::results::observation::ResolvedObservationPlan>
    {
        self.source_request
            .as_ref()
            .map(|request| &request.plan)
            .ok_or(AnalysisBridgeError::Identity("observation plan absent"))
    }

    fn advance_sample(&mut self) {
        self.sample_ordinal += 1;
        self.reduced.clear();
        self.checkpointed.clear();
        // Preserve the final request as the complete resolved-plan authority.
        // The next sample replaces its selection, never its authored plan.
    }

    fn prepare_sample<C: SampledHistorySource>(
        &mut self,
        source: &impl CacheImport,
        context: &C,
    ) -> AnalysisBridgeResult<()> {
        let policy = context.output_policy();
        let (preparation, coordinate) =
            context.observation_preparation(source, self.sample_ordinal)?;
        let fanout = prepare_observation_fanout(source, preparation, coordinate, None)?;
        // A volume-mesh admission resolved its plan from this same catalog; reuse it.
        let catalog = match &fanout.volume_mesh {
            Some(authority) => authority.catalog().clone(),
            None => context.field_catalog()?,
        };
        let catalog_objects = prepare_analysis_objects(
            Kind::ObservationFieldCatalog,
            catalog.identity(),
            &catalog.canonical_bytes()?,
            policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            policy.inventory_limits(),
        )?;
        self.plans = fanout
            .partitions
            .iter()
            .map(|partition| {
                (
                    partition.partition_plan.observation_id.clone(),
                    partition.partition_plan.clone(),
                )
            })
            .collect();
        self.partitions = fanout.partitions.into();
        self.source_request = Some(fanout.source_request);
        self.volume_mesh = fanout.volume_mesh;
        self.prepared_sample = Some(self.sample_ordinal);
        self.catalog = Some(catalog);
        self.catalog_objects = Some(catalog_objects);
        Ok(())
    }

    pub(crate) fn complete<C: SampledHistorySource>(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        context: &C,
    ) -> AnalysisBridgeResult<()> {
        let limits = context.output_policy().inventory_limits();
        let task = self.pending.as_ref().ok_or(AnalysisBridgeError::Identity(
            "no pending observation phase",
        ))?;
        match task {
            PendingObservationTask::Partition(task) => {
                let contribution =
                    import_observation_partition_contribution(source, response, task, limits)?;
                self.contributions
                    .entry(task.partition_plan.observation_id.clone())
                    .or_default()
                    .push(contribution);
            }
            PendingObservationTask::Reduction(task) => {
                let reduced = import_reduced_observation(source, response, task, limits)?;
                let id = reduced.result.observation_id.clone();
                self.plans.remove(&id);
                self.contributions.remove(&id);
                self.reduced.insert(id, (reduced, task.host.clone()));
            }
            PendingObservationTask::Checkpoint(task) => {
                let checkpoint =
                    import_observation_history_checkpoint_task(source, response, task, limits)?;
                let id = task.request.observation_id.clone();
                self.checkpointed.insert(id.clone());
                self.checkpoints.insert(id, (checkpoint, task.host.clone()));
            }
            PendingObservationTask::Assembly(task) => {
                let artifact = import_observation_assembly(source, response, task, limits)?;
                self.entries
                    .push(ObservationPublicationEntry::Sample(Box::new(
                        ObservationSamplePublication {
                            request: task.source_request.clone(),
                            manifest: artifact.manifest.clone(),
                            partition_evidence: artifact
                                .manifest
                                .results
                                .iter()
                                .map(|entry| {
                                    self.reduced
                                        .get(&entry.observation_id)
                                        .map(|(reduced, _)| {
                                            reduced.result.partition_evidence.clone()
                                        })
                                        .ok_or(AnalysisBridgeError::Identity(
                                            "assembled sample has no retained reduction evidence",
                                        ))
                                })
                                .collect::<AnalysisBridgeResult<Vec<_>>>()?,
                        },
                    )));
                self.completed.samples.push(artifact);
                self.advance_sample();
            }
            PendingObservationTask::Finalization(task) => {
                let history =
                    import_observation_history_finalization_task(source, response, task, limits)?;
                self.checkpoints
                    .remove(&history.history.manifest.observation_id);
                self.entries
                    .push(ObservationPublicationEntry::History(Box::new(
                        history.publication.clone(),
                    )));
                self.completed.histories.push(history);
            }
        }
        self.pending = None;
        Ok(())
    }
}
