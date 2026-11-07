//! Source-bound sampled fanout, ordered reductions and publication assembly.
mod retention;
use crate::{
    import_observation_assembly, import_observation_partition_contribution,
    import_reduced_observation, prepare_observation_assembly, prepare_observation_fanout,
    prepare_observation_reduction, ImportedObservationPartitionContribution, ImportedObservations,
    ImportedReducedObservation, ObservationAssemblyPreparation, ObservationPreparation,
    ObservationReductionPreparation, PreparedObservationAssembly, PreparedObservationFanout,
    PreparedObservationPartition, PreparedObservationReduction,
};
use crate::{AnalysisBridgeError, AnalysisBridgeResult, AnalysisOutputPolicy};
use runmat_analysis_core::ProcedureStep;
use runmat_analysis_fea::{
    execution::{ObservationHistorySourceContext, StepIdentity},
    results::{
        field::RecoveredFieldSample,
        observation::{
            ObservationFieldCatalog, ObservationPublicationCatalog, ObservationPublicationEntry,
            ObservationSamplePublication, ResolvedObservation,
        },
    },
};
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::{cache::CacheImport, ProgramExecutionResponse};
use std::collections::BTreeMap;

#[derive(Default)]
pub(crate) struct SampledStudyObservations {
    pub fanout: Option<PreparedObservationFanout>,
    pub(crate) contributions: BTreeMap<usize, ImportedObservationPartitionContribution>,
    pub reduction: Option<PreparedObservationReduction>,
    pub(crate) reduced: Vec<ImportedReducedObservation>,
    pub assembly: Option<PreparedObservationAssembly>,
    pub completed: Option<ImportedObservations>,
    pub catalog: Option<ObservationPublicationCatalog>,
}

impl SampledStudyObservations {
    pub fn prepare_fanout(
        source: &impl CacheImport,
        preparation: ObservationPreparation,
    ) -> AnalysisBridgeResult<PreparedObservationFanout> {
        Self::prepare_fanout_at(source, preparation, RecoveredFieldSample::Final)
    }

    /// Fan out at the retained sample coordinate of a single-sample source (for example one
    /// authored frequency of a sweep).
    pub fn prepare_fanout_at(
        source: &impl CacheImport,
        preparation: ObservationPreparation,
        sample: RecoveredFieldSample,
    ) -> AnalysisBridgeResult<PreparedObservationFanout> {
        Self::prepare_fanout_if_sampled(source, preparation, sample)?
            .ok_or_else(|| invalid("sampled authored observations have no executable partitions"))
    }

    /// Fan out at one retained sample of a multi-sample source; `None` when no authored
    /// observation samples this coordinate (for example a final-only output at an earlier
    /// frequency of a sweep).
    pub fn prepare_fanout_if_sampled(
        source: &impl CacheImport,
        preparation: ObservationPreparation,
        sample: RecoveredFieldSample,
    ) -> AnalysisBridgeResult<Option<PreparedObservationFanout>> {
        let task = prepare_observation_fanout(source, preparation, sample, None)?;
        Ok((!task.partitions.is_empty()).then_some(task))
    }
    pub fn partition(&self, ordinal: usize) -> AnalysisBridgeResult<&PreparedObservationPartition> {
        if self.contributions.contains_key(&ordinal) {
            return Err(invalid("sampled partition already completed"));
        }
        self.fanout
            .as_ref()
            .ok_or_else(|| invalid("sampled fanout absent"))?
            .partitions
            .get(ordinal)
            .ok_or_else(|| invalid("sampled partition ordinal outside fanout"))
    }
    pub fn import_partition(
        &self,
        source: &impl CacheImport,
        ordinal: usize,
        response: ProgramExecutionResponse,
        policy: AnalysisOutputPolicy,
    ) -> AnalysisBridgeResult<ImportedObservationPartitionContribution> {
        import_observation_partition_contribution(
            source,
            response,
            self.partition(ordinal)?,
            policy.inventory_limits(),
        )
    }
    /// Observations the fan-out sampled, in plan order. An observation whose schedule excludes
    /// this sample (for example one frequency of a sweep) has no partitions and no reduction.
    fn sampled_observations(&self) -> impl Iterator<Item = &ResolvedObservation> + '_ {
        let fanout = self.fanout.as_ref();
        fanout
            .into_iter()
            .flat_map(|fanout| fanout.source_request.plan.observations.iter())
            .filter(move |observation| {
                fanout.is_some_and(|fanout| {
                    fanout.partitions.iter().any(|task| {
                        task.partition_plan.observation_id == observation.spec.observation_id
                    })
                })
            })
    }
    /// Reductions this fan-out needs before assembly: one per sampled observation.
    pub fn reduction_count(&self) -> usize {
        self.sampled_observations().count()
    }
    pub fn prepare_reduction(
        &self,
        revision: &ProgramRevision,
        policy: AnalysisOutputPolicy,
    ) -> AnalysisBridgeResult<PreparedObservationReduction> {
        let ordinal = self.reduced.len();
        let fanout = self.fanout.as_ref().expect("prepared fanout");
        let observation = self
            .sampled_observations()
            .nth(ordinal)
            .ok_or_else(|| invalid("sampled reduction ordinal outside fanout"))?;
        let selected = fanout
            .partitions
            .iter()
            .enumerate()
            .filter(|(_, task)| task.partition_plan.observation_id == observation.spec.observation_id)
            .collect::<Vec<_>>();
        let first = selected
            .first()
            .ok_or_else(|| invalid("sampled reduction source absent"))?
            .1;
        let contributions = selected
            .iter()
            .map(|(index, _)| {
                self.contributions
                    .get(index)
                    .cloned()
                    .ok_or_else(|| invalid("sampled reduction omits a completed partition"))
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        let task = prepare_observation_reduction(ObservationReductionPreparation {
            source_host: first.host.clone(),
            source_request: fanout.source_request.clone(),
            partition_plan: first.partition_plan.clone(),
            contributions,
            program_revision: revision.clone(),
            chunk_bytes: policy.chunk_bytes(),
            inventory_limits: policy.inventory_limits(),
        })?;

        Ok(task)
    }
    pub fn import_reduction(
        &self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        policy: AnalysisOutputPolicy,
    ) -> AnalysisBridgeResult<ImportedReducedObservation> {
        import_reduced_observation(
            source,
            response,
            self.reduction
                .as_ref()
                .ok_or_else(|| invalid("sampled reduction absent"))?,
            policy.inventory_limits(),
        )
    }
    pub fn prepare_assembly(
        &self,
        revision: &ProgramRevision,
        policy: AnalysisOutputPolicy,
    ) -> AnalysisBridgeResult<PreparedObservationAssembly> {
        let task = prepare_observation_assembly(ObservationAssemblyPreparation {
            source_host: self
                .reduction
                .as_ref()
                .expect("completed reduction")
                .host
                .clone(),
            source_request: self.fanout.as_ref().expect("fanout").source_request.clone(),
            results: self.reduced.clone(),
            program_revision: revision.clone(),
            chunk_bytes: policy.chunk_bytes(),
            inventory_limits: policy.inventory_limits(),
        })?;

        Ok(task)
    }
    pub fn import_assembly(
        &self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        policy: AnalysisOutputPolicy,
    ) -> AnalysisBridgeResult<ImportedObservations> {
        import_observation_assembly(
            source,
            response,
            self.assembly
                .as_ref()
                .ok_or_else(|| invalid("sampled assembly absent"))?,
            policy.inventory_limits(),
        )
    }
    pub fn publication_catalog(
        &self,
        artifact: &ImportedObservations,
        producer: StepIdentity,
        fields: ObservationFieldCatalog,
        step: &ProcedureStep,
        context: ObservationHistorySourceContext<'_>,
    ) -> AnalysisBridgeResult<ObservationPublicationCatalog> {
        let task = self
            .assembly
            .as_ref()
            .ok_or_else(|| invalid("sampled assembly absent"))?;
        Ok(ObservationPublicationCatalog::new(
            producer,
            task.source_request.plan.clone(),
            fields,
            step,
            vec![ObservationPublicationEntry::Sample(Box::new(
                ObservationSamplePublication {
                    request: task.source_request.clone(),
                    manifest: artifact.manifest.clone(),
                    partition_evidence: artifact
                        .manifest
                        .results
                        .iter()
                        .map(|entry| {
                            self.reduced
                                .iter()
                                .find(|result| result.result.observation_id == entry.observation_id)
                                .map(|result| result.result.partition_evidence.clone())
                                .ok_or_else(|| {
                                    invalid("sampled assembly lacks original reduction evidence")
                                })
                        })
                        .collect::<AnalysisBridgeResult<Vec<_>>>()?,
                },
            ))],
            context,
        )?)
    }
}
fn invalid(message: &str) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(message.into())
}
