//! Source-bound sampled fanout, ordered reductions and publication assembly.
use super::*;
use crate::{
    import_observation_assembly, import_observation_partition_contribution,
    import_reduced_observation, prepare_observation_assembly, prepare_observation_fanout,
    prepare_observation_reduction, ImportedObservationPartitionContribution, ImportedObservations,
    ImportedReducedObservation, ObservationAssemblyPreparation, ObservationPreparation,
    ObservationPublishedSource, ObservationReductionPreparation, ObservationWorkloadSource,
    PreparedObservationAssembly, PreparedObservationFanout, PreparedObservationReduction,
};
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind as Kind, ObservationHistorySourceContext},
    results::{
        field::RecoveredFieldSample,
        observation::{
            ObservationFieldCatalog, ObservationLimits, ObservationPublicationCatalog,
            ObservationPublicationEntry, ObservationSamplePublication,
        },
    },
};
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct MagneticObservations {
    pub fanout: Option<PreparedObservationFanout>,
    contributions: BTreeMap<usize, ImportedObservationPartitionContribution>,
    pub reduction: Option<PreparedObservationReduction>,
    reduced: Vec<ImportedReducedObservation>,
    pub assembly: Option<PreparedObservationAssembly>,
    pub completed: Option<ImportedObservations>,
    pub catalog: Option<ObservationPublicationCatalog>,
}

impl MagneticStudyCoordinator {
    pub fn prepare_observation_fanout(
        &mut self,
        source: &impl CacheImport,
    ) -> AnalysisBridgeResult<&PreparedObservationFanout> {
        self.require(MagneticStudyPhase::ReadyForObservationFanout)?;
        let recovery = self.recovery_task.as_ref().expect("completed recovery");
        let fields = self.fields.as_ref().expect("completed fields");
        let policy = self.preparation.output_policy;
        let task = prepare_observation_fanout(
            source,
            ObservationPreparation {
                source_host: recovery.host.clone(),
                source: ObservationPublishedSource::SampledRecovery {
                    recovery_request: recovery.request.clone(),
                    fields: fields.clone(),
                    sampling: self
                        .result
                        .as_ref()
                        .expect("completed solve")
                        .result
                        .sampling_topology()
                        .clone(),
                },
                source_workload: ObservationWorkloadSource::Magnetostatic(
                    self.input(Kind::MagnetostaticWorkload)?,
                ),
                mesh: Some(self.input(Kind::Mesh)?),
                accelerator_requests: Vec::new(),
                sequence: SequenceIdentity::from_solver_contract(
                    &self.preparation.workload().model.sequence,
                )?,
                step: self.selected_step().clone(),
                limits: ObservationLimits::default(),
                program_revision: self.preparation.program_revision.clone(),
                chunk_bytes: policy.chunk_bytes(),
                inventory_limits: policy.inventory_limits(),
            },
            RecoveredFieldSample::Final,
            None,
        )?;
        if task.partitions.is_empty() {
            return Err(invalid(
                "magnetic authored observations have no executable partitions",
            ));
        }
        self.observation_state.fanout = Some(task);
        self.phase = MagneticStudyPhase::AwaitingObservationPartitions;
        Ok(self
            .observation_state
            .fanout
            .as_ref()
            .expect("just prepared"))
    }

    /// Completion order is arbitrary; each immutable task ordinal may complete exactly once.
    pub fn complete_observation_partition(
        &mut self,
        source: &impl CacheImport,
        ordinal: usize,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        self.require(MagneticStudyPhase::AwaitingObservationPartitions)?;
        if self.observation_state.contributions.contains_key(&ordinal) {
            return Err(invalid("magnetic observation partition already completed"));
        }
        let fanout = self
            .observation_state
            .fanout
            .as_ref()
            .expect("prepared fanout");
        let task = fanout
            .partitions
            .get(ordinal)
            .ok_or_else(|| invalid("magnetic partition ordinal outside fanout"))?;
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = MagneticStudyResultClosure::new(
            MagneticStudyResultPhase::ObservationPartition { ordinal },
            &task.host,
            &response,
            &accounting,
        );
        let contribution = import_observation_partition_contribution(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        let count = fanout.partitions.len();
        self.observation_state
            .contributions
            .insert(ordinal, contribution);
        self.usage = Some(usage);
        self.closures.push(closure);
        if self.observation_state.contributions.len() == count {
            self.phase = MagneticStudyPhase::ReadyForObservationReduction { ordinal: 0 };
        }
        Ok(())
    }

    pub fn prepare_observation_reduction(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationReduction> {
        let ordinal = self.observation_state.reduced.len();
        self.require(MagneticStudyPhase::ReadyForObservationReduction { ordinal })?;
        let fanout = self
            .observation_state
            .fanout
            .as_ref()
            .expect("prepared fanout");
        let observation = &fanout.source_request.plan.observations[ordinal];
        let selected = fanout
            .partitions
            .iter()
            .enumerate()
            .filter(|(_, task)| {
                task.partition_plan.observation_id == observation.spec.observation_id
            })
            .collect::<Vec<_>>();
        let first = selected
            .first()
            .ok_or_else(|| invalid("magnetic reduction source absent"))?
            .1;
        let contributions = selected
            .iter()
            .map(|(index, _)| {
                self.observation_state
                    .contributions
                    .get(index)
                    .cloned()
                    .ok_or_else(|| invalid("magnetic reduction omits a completed partition"))
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        let policy = self.preparation.output_policy;
        let task = prepare_observation_reduction(ObservationReductionPreparation {
            source_host: first.host.clone(),
            source_request: fanout.source_request.clone(),
            partition_plan: first.partition_plan.clone(),
            contributions,
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: policy.chunk_bytes(),
            inventory_limits: policy.inventory_limits(),
        })?;
        self.observation_state.reduction = Some(task);
        self.phase = MagneticStudyPhase::AwaitingObservationReduction { ordinal };
        Ok(self
            .observation_state
            .reduction
            .as_ref()
            .expect("just prepared"))
    }

    pub fn complete_observation_reduction(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        let ordinal = self.observation_state.reduced.len();
        self.require(MagneticStudyPhase::AwaitingObservationReduction { ordinal })?;
        let task = self
            .observation_state
            .reduction
            .as_ref()
            .expect("awaited reduction");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = MagneticStudyResultClosure::new(
            MagneticStudyResultPhase::ObservationReduction { ordinal },
            &task.host,
            &response,
            &accounting,
        );
        let reduced = import_reduced_observation(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        self.observation_state.reduced.push(reduced);
        self.usage = Some(usage);
        self.closures.push(closure);
        let count = self
            .observation_state
            .fanout
            .as_ref()
            .expect("fanout")
            .source_request
            .plan
            .observations
            .len();
        self.phase = if self.observation_state.reduced.len() == count {
            MagneticStudyPhase::ReadyForObservationAssembly
        } else {
            MagneticStudyPhase::ReadyForObservationReduction {
                ordinal: ordinal + 1,
            }
        };
        Ok(())
    }

    pub fn prepare_observation_assembly(
        &mut self,
    ) -> AnalysisBridgeResult<&PreparedObservationAssembly> {
        self.require(MagneticStudyPhase::ReadyForObservationAssembly)?;
        let policy = self.preparation.output_policy;
        let task = prepare_observation_assembly(ObservationAssemblyPreparation {
            source_host: self
                .observation_state
                .reduction
                .as_ref()
                .expect("completed reduction")
                .host
                .clone(),
            source_request: self
                .observation_state
                .fanout
                .as_ref()
                .expect("fanout")
                .source_request
                .clone(),
            results: self.observation_state.reduced.clone(),
            program_revision: self.preparation.program_revision.clone(),
            chunk_bytes: policy.chunk_bytes(),
            inventory_limits: policy.inventory_limits(),
        })?;
        self.observation_state.assembly = Some(task);
        self.phase = MagneticStudyPhase::AwaitingObservationAssembly;
        Ok(self
            .observation_state
            .assembly
            .as_ref()
            .expect("just prepared"))
    }

    pub fn complete_observation_assembly(
        &mut self,
        source: &impl CacheImport,
        response: ProgramExecutionResponse,
        accounting: AttemptResourceAccounting,
    ) -> AnalysisBridgeResult<()> {
        self.require(MagneticStudyPhase::AwaitingObservationAssembly)?;
        let task = self
            .observation_state
            .assembly
            .as_ref()
            .expect("awaited assembly");
        validate_accounting(&task.host, &task.resource_plan, &response, &accounting)?;
        let closure = MagneticStudyResultClosure::new(
            MagneticStudyResultPhase::ObservationAssembly,
            &task.host,
            &response,
            &accounting,
        );
        let artifact = import_observation_assembly(
            source,
            response,
            task,
            self.preparation.output_policy.inventory_limits(),
        )?;
        let fields = self.fields.as_ref().expect("completed fields");
        let recovery = self.recovery_task.as_ref().expect("completed recovery");
        let catalog = ObservationPublicationCatalog::new(
            self.solve_task
                .as_ref()
                .expect("completed solve")
                .host
                .workload
                .step,
            task.source_request.plan.clone(),
            ObservationFieldCatalog::sampled_recovery(
                recovery.request.clone(),
                fields.manifest.clone(),
                self.result
                    .as_ref()
                    .expect("completed solve")
                    .result
                    .sampling_topology()
                    .clone(),
            )?,
            self.selected_step(),
            vec![ObservationPublicationEntry::Sample(Box::new(
                ObservationSamplePublication {
                    request: task.source_request.clone(),
                    manifest: artifact.manifest.clone(),
                    partition_evidence: artifact
                        .manifest
                        .results
                        .iter()
                        .map(|entry| {
                            self.observation_state
                                .reduced
                                .iter()
                                .find(|result| result.result.observation_id == entry.observation_id)
                                .map(|result| result.result.partition_evidence.clone())
                                .ok_or_else(|| {
                                    invalid("magnetic assembly lacks original reduction evidence")
                                })
                        })
                        .collect::<AnalysisBridgeResult<Vec<_>>>()?,
                },
            ))],
            ObservationHistorySourceContext::Magnetostatic {
                workload: self.preparation.workload(),
                mesh: &self.preparation.mesh,
            },
        )?;
        let usage = self.extend_usage(&task.resource_plan, accounting)?;
        let tail = self.prepare_tail(&usage, Some(catalog.clone()))?;
        self.observation_state.completed = Some(artifact);
        self.observation_state.catalog = Some(catalog);
        self.usage = Some(usage);
        self.closures.push(closure);
        self.tail = Some(tail);
        self.phase = MagneticStudyPhase::ReadyForAccuracy;
        Ok(())
    }
}
