//! Serial coordination of admitted shared observation phases, not numerical evaluation.
//! Each imported phase is validated before committing progress; prepared work survives failure.
use crate::{
    import_analysis_output_set, import_observation_assembly,
    import_observation_history_checkpoint_task, import_observation_history_finalization_task,
    import_observation_partition_contribution, import_reduced_observation,
    prepare_analysis_objects, prepare_observation_assembly, prepare_observation_fanout,
    prepare_observation_history_checkpoint_task, prepare_observation_history_finalization_task,
    prepare_observation_reduction, AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload,
    AnalysisOutputRoot, ImportedObservationHistoryCheckpointTask,
    ImportedObservationHistoryFinalization, ImportedObservationPartitionContribution,
    ImportedObservations, ImportedReducedObservation, ImportedThermalResult,
    ObservationAssemblyPreparation, ObservationHistoryCheckpointPreparation,
    ObservationHistoryFinalizationPreparation, ObservationHistoryFinalizationSource,
    ObservationPreparation, ObservationPublishedSource, ObservationReductionPreparation,
    ObservationWorkloadSource, PreparedAnalysisObjects, PreparedObservationPartition,
    PreparedThermalTask, ThermalStudyPreparation,
};
use runmat_analysis_core::{ObservationId, ObservationRequest};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind as Kind, ObservationEvaluationRequest,
        ObservationHistoryFinalizationRequest, ObservationHistorySourceContext, SequenceIdentity,
    },
    results::observation::{
        ObservationFieldCatalog, ObservationLimits, ObservationPartitionPlan,
        ObservationPublicationCatalog, ObservationPublicationEntry, ObservationSamplePublication,
    },
};
use runmat_execution_artifact::{cache::CacheImport, ObjectNamespace, ProgramExecutionResponse};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

mod task;
use task::PendingObservationTask;
pub use task::ThermalObservationTask;

#[derive(Default)]
pub struct CompletedThermalObservations {
    pub catalog: Option<ObservationPublicationCatalog>,
    pub samples: Vec<ImportedObservations>,
    pub histories: Vec<ImportedObservationHistoryFinalization>,
}

pub(super) struct Context<'a> {
    pub setup: &'a ThermalStudyPreparation,
    pub solve: &'a PreparedThermalTask,
    pub result: &'a ImportedThermalResult,
}
impl Context<'_> {
    fn input(&self, kind: Kind) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
        self.solve
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .cloned()
            .ok_or(AnalysisBridgeError::Identity(
                "thermal observation source input absent",
            ))
    }
    fn step(&self) -> &runmat_analysis_core::ProcedureStep {
        &self.setup.product.workload.model.sequence.steps[0]
    }
    fn source(&self) -> ObservationHistorySourceContext<'_> {
        ObservationHistorySourceContext::Thermal {
            workload: &self.setup.product.workload,
            mesh: &self.setup.mesh,
        }
    }
    fn is_history(&self, id: &ObservationId) -> bool {
        self.step().observations.iter().any(|observation| {
            matches!(observation,
            ObservationRequest::History(spec) if &spec.observation_id == id)
        })
    }
}

#[derive(Default)]
pub(super) struct Workflow {
    sample_ordinal: usize,
    prepared_sample: Option<usize>,
    source_request: Option<ObservationEvaluationRequest>,
    catalog: Option<ObservationFieldCatalog>,
    catalog_objects: Option<PreparedAnalysisObjects>,
    output: Option<AnalysisOutputRoot>,
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
    completed: CompletedThermalObservations,
}

impl Workflow {
    pub(super) fn completed(&self) -> &CompletedThermalObservations {
        &self.completed
    }
    pub(super) fn into_completed(self) -> CompletedThermalObservations {
        self.completed
    }
    pub(super) fn pending(&self) -> AnalysisBridgeResult<ThermalObservationTask<'_>> {
        self.pending
            .as_ref()
            .map(PendingObservationTask::view)
            .ok_or(AnalysisBridgeError::Identity(
                "no pending thermal observation phase",
            ))
    }

    pub(super) fn prepare(
        &mut self,
        source: &impl CacheImport,
        context: &Context<'_>,
    ) -> AnalysisBridgeResult<Option<ThermalObservationTask<'_>>> {
        if self.pending.is_some() {
            return Err(AnalysisBridgeError::Identity(
                "thermal observation phase already prepared",
            ));
        }
        if context.step().observations.is_empty() {
            return Ok(None);
        }
        let policy = context.setup.output_policy;
        loop {
            if self.sample_ordinal < context.result.manifest.fields.len() {
                if self.prepared_sample != Some(self.sample_ordinal) {
                    self.prepare_sample(source, context)?;
                }
                if let Some(partition) = self.partitions.pop_front() {
                    self.pending = Some(PendingObservationTask::Partition(Box::new(partition)));
                    break;
                }
                if let Some((id, plan)) = self.plans.first_key_value() {
                    let task = prepare_observation_reduction(ObservationReductionPreparation {
                        source_host: context.solve.host.clone(),
                        source_request: self
                            .source_request
                            .as_ref()
                            .expect("sample request")
                            .clone(),
                        partition_plan: plan.clone(),
                        contributions: self.contributions.get(id).cloned().unwrap_or_default(),
                        program_revision: context.setup.program_revision.clone(),
                        chunk_bytes: policy.chunk_bytes(),
                        inventory_limits: policy.inventory_limits(),
                    })?;
                    self.pending = Some(PendingObservationTask::Reduction(Box::new(task)));
                    break;
                }
                if let Some((id, (sample, host))) = self
                    .reduced
                    .iter()
                    .find(|(id, _)| context.is_history(id) && !self.checkpointed.contains(*id))
                {
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
                            program_revision: context.setup.program_revision.clone(),
                            chunk_bytes: policy.chunk_bytes(),
                            inventory_limits: policy.inventory_limits(),
                        },
                    )?;
                    self.pending = Some(PendingObservationTask::Checkpoint(Box::new(task)));
                    break;
                }
                if self.reduced.keys().any(|id| !context.is_history(id)) {
                    let task = prepare_observation_assembly(ObservationAssemblyPreparation {
                        source_host: context.solve.host.clone(),
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
                        program_revision: context.setup.program_revision.clone(),
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
                    context.solve.host.workload.step,
                    id.clone(),
                    &checkpoint.checkpoint,
                    catalog,
                    context.source(),
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
                        source: ObservationHistoryFinalizationSource::Thermal {
                            workload: context.input(Kind::ThermalWorkload)?,
                            mesh: context.input(Kind::Mesh)?,
                        },
                        program_revision: context.setup.program_revision.clone(),
                        chunk_bytes: policy.chunk_bytes(),
                        inventory_limits: policy.inventory_limits(),
                    },
                )?;
                self.pending = Some(PendingObservationTask::Finalization(Box::new(task)));
                break;
            }
            let catalog = ObservationPublicationCatalog::new(
                context.solve.host.workload.step,
                self.last_plan()?.clone(),
                self.catalog.as_ref().expect("source catalog").clone(),
                context.step(),
                self.entries.clone(),
                context.source(),
            )?;
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
            .ok_or(AnalysisBridgeError::Identity(
                "thermal observation plan absent",
            ))
    }

    fn advance_sample(&mut self) {
        self.sample_ordinal += 1;
        self.reduced.clear();
        self.checkpointed.clear();
        // Preserve the final request as the complete resolved-plan authority.
        // The next sample replaces its selection, never its authored plan.
    }

    fn prepare_sample(
        &mut self,
        source: &impl CacheImport,
        context: &Context<'_>,
    ) -> AnalysisBridgeResult<()> {
        if self.output.is_none() {
            let output = import_analysis_output_set(
                source,
                context.result.response.clone(),
                &context.solve.host,
            )?;
            self.output = output
                .manifest
                .outputs
                .into_iter()
                .find(|output| output.kind == Kind::SolverResult);
        }
        let fields = &context.result.manifest.fields[self.sample_ordinal];
        let coordinate = fields
            .first()
            .ok_or(AnalysisBridgeError::Identity(
                "thermal sample has no fields",
            ))?
            .sample;
        let policy = context.setup.output_policy;
        let fanout = prepare_observation_fanout(
            source,
            ObservationPreparation {
                source_host: context.solve.host.clone(),
                source: ObservationPublishedSource::Thermal {
                    manifest: context.result.manifest.clone(),
                    output: self.output.as_ref().expect("solver result output").clone(),
                    access: context.solve.host.artifact_access.clone(),
                    sample_ordinal: self.sample_ordinal,
                },
                source_workload: ObservationWorkloadSource::Thermal(
                    context.input(Kind::ThermalWorkload)?,
                ),
                mesh: Some(context.input(Kind::Mesh)?),
                accelerator_requests: Vec::new(),
                sequence: SequenceIdentity::from_solver_contract(
                    &context.setup.product.workload.model.sequence,
                )?,
                step: context.step().clone(),
                limits: ObservationLimits::default(),
                program_revision: context.setup.program_revision.clone(),
                chunk_bytes: policy.chunk_bytes(),
                inventory_limits: policy.inventory_limits(),
            },
            coordinate,
            None,
        )?;
        let catalog = ObservationFieldCatalog::thermal(context.result.manifest.clone())?;
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
        self.prepared_sample = Some(self.sample_ordinal);
        self.catalog = Some(catalog);
        self.catalog_objects = Some(catalog_objects);
        Ok(())
    }

    pub(super) fn complete(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        context: &Context<'_>,
    ) -> AnalysisBridgeResult<()> {
        let limits = context.setup.output_policy.inventory_limits();
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
