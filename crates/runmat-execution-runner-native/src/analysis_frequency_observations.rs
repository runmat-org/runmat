use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{ObservationId, ObservationRequest, ProcedureStep};
use runmat_analysis_execution::{
    import_observation_history_checkpoint_task, import_observation_partition_contribution,
    import_reduced_observation, import_structural_observation_assembly,
    prepare_observation_history_checkpoint_task, prepare_structural_observation_assembly,
    prepare_structural_observation_fanout, prepare_structural_observation_reduction,
    ImportedObservationHistoryCheckpointTask, ImportedObservations, ImportedRecoveredFields,
    ImportedReducedObservation, ObservationHistoryCheckpointPreparation, PreparedAnalysisObjects,
    PreparedStructuralFrequencyFieldRecovery, StructuralObservationAssemblyPreparation,
    StructuralObservationPreparation, StructuralObservationReductionPreparation,
    StructuralObservationWorkloadSource,
};
use runmat_analysis_fea::execution::SequenceIdentity;
use runmat_analysis_fea::results::field::RecoveredFieldSample;
use runmat_analysis_fea::results::observation::{
    ObservationLimits, ObservationPartitionPlanIdentity,
};
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::object::ObjectInventoryLimits;

use crate::analysis_frequency::batch_policy;
use crate::analysis_sequence::StructuralSequenceExecutionPolicy;
use crate::analysis_task_pool::{
    execute_analysis_task_batch, planned_resource_attempt as resource_attempt,
    AnalysisTaskDependency, AnalysisTaskPool, PreparedAnalysisTask,
};
use crate::{NativeExecutionError, NativeExecutionResult};

pub struct CompletedStructuralFrequencyObservationSample {
    pub sample: RecoveredFieldSample,
    pub observations: Option<ImportedObservations>,
}

pub struct CompletedStructuralFrequencyHistory {
    pub observation_id: ObservationId,
    pub checkpoint: ImportedObservationHistoryCheckpointTask,
}

pub(crate) struct CompletedStructuralFrequencyObservations {
    pub samples: Vec<CompletedStructuralFrequencyObservationSample>,
    pub histories: Vec<CompletedStructuralFrequencyHistory>,
    pub resource_attempts: Vec<runmat_analysis_execution::PlannedResourceUsageAttempt>,
}

pub(crate) struct StructuralFrequencyObservationExecution<'a> {
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub step: ProcedureStep,
    pub sequence: SequenceIdentity,
    pub recoveries: &'a [PreparedStructuralFrequencyFieldRecovery],
    pub recovered_fields: &'a [ImportedRecoveredFields],
    pub recovery_dependencies: &'a [AnalysisTaskDependency],
    pub accelerator_requests: Vec<AcceleratorRequest>,
    pub observation_limits: ObservationLimits,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub(crate) async fn execute_structural_frequency_observations(
    pool: &mut impl AnalysisTaskPool,
    execution: StructuralFrequencyObservationExecution<'_>,
    policy: &StructuralSequenceExecutionPolicy,
) -> NativeExecutionResult<CompletedStructuralFrequencyObservations> {
    if execution.recoveries.len() != execution.recovered_fields.len()
        || execution.recoveries.len() != execution.recovery_dependencies.len()
    {
        return Err(NativeExecutionError::Protocol(
            "frequency observation sources have different inventories".into(),
        ));
    }
    if execution.step.observations.is_empty() {
        let samples = execution
            .recoveries
            .iter()
            .zip(execution.recovered_fields)
            .map(|(recovery, fields)| {
                Ok(CompletedStructuralFrequencyObservationSample {
                    sample: recovered_frequency_sample(recovery, fields)?,
                    observations: None,
                })
            })
            .collect::<NativeExecutionResult<Vec<_>>>()?;
        return Ok(CompletedStructuralFrequencyObservations {
            samples,
            histories: Vec::new(),
            resource_attempts: Vec::new(),
        });
    }
    let mut fanouts = Vec::with_capacity(execution.recoveries.len());
    for ((recovery, fields), dependency) in execution
        .recoveries
        .iter()
        .zip(execution.recovered_fields)
        .zip(execution.recovery_dependencies)
    {
        let sample = recovered_frequency_sample(recovery, fields)?;
        let fanout = prepare_structural_observation_fanout(
            pool,
            StructuralObservationPreparation {
                source_host: recovery.host.clone(),
                source_request: recovery.request.clone(),
                source_fields: fields.clone(),
                source_workload: StructuralObservationWorkloadSource::Frequency(
                    execution.workload.clone(),
                ),
                mesh: Some(execution.mesh.clone()),
                accelerator_requests: execution.accelerator_requests.clone(),
                sequence: execution.sequence,
                step: execution.step.clone(),
                limits: execution.observation_limits,
                program_revision: execution.program_revision.clone(),
                chunk_bytes: execution.chunk_bytes,
                inventory_limits: execution.inventory_limits,
            },
            sample,
            None,
        )
        .map_err(super::analysis_task_pool::protocol)?;
        fanouts.push((sample, *dependency, fanout));
    }

    let mut partition_origins = Vec::new();
    let mut prepared_partitions = Vec::new();
    for (sample_index, (_, _, fanout)) in fanouts.iter_mut().enumerate() {
        for partition in fanout.partitions.drain(..) {
            partition_origins.push((sample_index, partition.partition_plan.identity));
            prepared_partitions.push(partition);
        }
    }
    let executed_partitions = execute_analysis_task_batch(
        pool,
        prepared_partitions
            .iter()
            .zip(&partition_origins)
            .map(|(task, (sample_index, _))| PreparedAnalysisTask {
                host: &task.host,
                program: &task.program,
                resource_plan: Some(&task.resource_plan),
                fallback_resources: policy.resources.clone(),
                dependencies: vec![fanouts[*sample_index].1],
            })
            .collect(),
        batch_policy(policy),
    )
    .await?;
    let mut resource_attempts = Vec::new();
    let mut contribution_groups =
        BTreeMap::<(usize, ObservationPartitionPlanIdentity), ContributionGroup>::new();
    for ((task, origin), executed) in prepared_partitions
        .iter()
        .zip(&partition_origins)
        .zip(executed_partitions)
    {
        resource_attempts.push(resource_attempt(&task.resource_plan, &executed)?);
        let contribution = import_observation_partition_contribution(
            pool,
            executed.response,
            task,
            execution.inventory_limits,
        )
        .map_err(super::analysis_task_pool::protocol)?;
        let group = contribution_groups
            .entry(*origin)
            .or_insert_with(|| ContributionGroup {
                plan: task.partition_plan.clone(),
                contributions: Vec::new(),
                dependencies: Vec::new(),
            });
        if group.plan != task.partition_plan {
            return Err(NativeExecutionError::Protocol(
                "frequency observation partition identity aliases another plan".into(),
            ));
        }
        group.contributions.push(contribution);
        group.dependencies.push(AnalysisTaskDependency {
            identity: executed.identity,
            task_id: executed.task_id,
        });
    }

    let mut reduction_origins = Vec::new();
    let mut reductions = Vec::new();
    for ((sample_index, plan_identity), group) in contribution_groups {
        let source = &fanouts[sample_index].2;
        let reduction =
            prepare_structural_observation_reduction(StructuralObservationReductionPreparation {
                source_host: execution.recoveries[sample_index].host.clone(),
                source_request: source.source_request.clone(),
                partition_plan: group.plan,
                contributions: group.contributions,
                program_revision: execution.program_revision.clone(),
                chunk_bytes: execution.chunk_bytes,
                inventory_limits: execution.inventory_limits,
            })
            .map_err(super::analysis_task_pool::protocol)?;
        reduction_origins.push((sample_index, plan_identity, group.dependencies));
        reductions.push(reduction);
    }
    let executed_reductions = execute_analysis_task_batch(
        pool,
        reductions
            .iter()
            .zip(&reduction_origins)
            .map(|(task, (_, _, dependencies))| PreparedAnalysisTask {
                host: &task.host,
                program: &task.program,
                resource_plan: Some(&task.resource_plan),
                fallback_resources: policy.resources.clone(),
                dependencies: dependencies.clone(),
            })
            .collect(),
        batch_policy(policy),
    )
    .await?;
    let mut reduced = Vec::with_capacity(reductions.len());
    for ((task, (sample_index, _, _)), executed) in reductions
        .iter()
        .zip(&reduction_origins)
        .zip(executed_reductions)
    {
        resource_attempts.push(resource_attempt(&task.resource_plan, &executed)?);
        reduced.push(ReducedRecord {
            sample_index: *sample_index,
            source_host: task.host.clone(),
            dependency: AnalysisTaskDependency {
                identity: executed.identity,
                task_id: executed.task_id,
            },
            result: import_reduced_observation(
                pool,
                executed.response,
                task,
                execution.inventory_limits,
            )
            .map_err(super::analysis_task_pool::protocol)?,
        });
    }

    let samples = execute_sample_assemblies(
        pool,
        &execution,
        policy,
        &fanouts,
        &reduced,
        &mut resource_attempts,
    )
    .await?;
    let histories = execute_histories(
        pool,
        &execution,
        policy,
        &fanouts,
        &reduced,
        &mut resource_attempts,
    )
    .await?;
    Ok(CompletedStructuralFrequencyObservations {
        samples,
        histories,
        resource_attempts,
    })
}

fn recovered_frequency_sample(
    recovery: &PreparedStructuralFrequencyFieldRecovery,
    fields: &ImportedRecoveredFields,
) -> NativeExecutionResult<RecoveredFieldSample> {
    if !matches!(
        recovery.request.source,
        runmat_analysis_fea::execution::FieldRecoverySource::StructuralFrequencyEntry(_)
    ) {
        return Err(NativeExecutionError::Protocol(
            "frequency observation source is not a frequency recovery".into(),
        ));
    }
    let sample = fields
        .manifest
        .fields
        .first()
        .map(|entry| entry.sample)
        .ok_or_else(|| {
            NativeExecutionError::Protocol(
                "frequency recovery produced an empty field inventory".into(),
            )
        })?;
    if fields
        .manifest
        .fields
        .iter()
        .any(|entry| entry.sample != sample)
    {
        return Err(NativeExecutionError::Protocol(
            "one frequency recovery mixes sample coordinates".into(),
        ));
    }
    Ok(sample)
}

struct ContributionGroup {
    plan: runmat_analysis_fea::results::observation::ObservationPartitionPlan,
    contributions: Vec<runmat_analysis_execution::ImportedObservationPartitionContribution>,
    dependencies: Vec<AnalysisTaskDependency>,
}

struct ReducedRecord {
    sample_index: usize,
    source_host: runmat_analysis_execution::AnalysisHostWorkload,
    dependency: AnalysisTaskDependency,
    result: ImportedReducedObservation,
}

async fn execute_sample_assemblies(
    pool: &mut impl AnalysisTaskPool,
    execution: &StructuralFrequencyObservationExecution<'_>,
    policy: &StructuralSequenceExecutionPolicy,
    fanouts: &[(
        RecoveredFieldSample,
        AnalysisTaskDependency,
        runmat_analysis_execution::PreparedStructuralObservationFanout,
    )],
    reduced: &[ReducedRecord],
    resource_attempts: &mut Vec<runmat_analysis_execution::PlannedResourceUsageAttempt>,
) -> NativeExecutionResult<Vec<CompletedStructuralFrequencyObservationSample>> {
    let mut origins = Vec::new();
    let mut tasks = Vec::new();
    for (sample_index, (_, _, fanout)) in fanouts.iter().enumerate() {
        let results = reduced
            .iter()
            .filter(|record| record.sample_index == sample_index)
            .map(|record| record.result.clone())
            .collect::<Vec<_>>();
        if results.is_empty() {
            continue;
        }
        let dependencies: Vec<AnalysisTaskDependency> = reduced
            .iter()
            .filter(|record| record.sample_index == sample_index)
            .map(|record| record.dependency)
            .collect();
        let task =
            prepare_structural_observation_assembly(StructuralObservationAssemblyPreparation {
                source_host: execution.recoveries[sample_index].host.clone(),
                source_request: fanout.source_request.clone(),
                results,
                program_revision: execution.program_revision.clone(),
                chunk_bytes: execution.chunk_bytes,
                inventory_limits: execution.inventory_limits,
            })
            .map_err(super::analysis_task_pool::protocol)?;
        origins.push((sample_index, dependencies));
        tasks.push(task);
    }
    let executed = execute_analysis_task_batch(
        pool,
        tasks
            .iter()
            .zip(&origins)
            .map(|(task, (_, dependencies))| PreparedAnalysisTask {
                host: &task.host,
                program: &task.program,
                resource_plan: Some(&task.resource_plan),
                fallback_resources: policy.resources.clone(),
                dependencies: dependencies.clone(),
            })
            .collect(),
        batch_policy(policy),
    )
    .await?;
    let mut observations = BTreeMap::new();
    for ((task, (sample_index, _)), executed) in tasks.iter().zip(&origins).zip(executed) {
        resource_attempts.push(resource_attempt(&task.resource_plan, &executed)?);
        let imported = import_structural_observation_assembly(
            pool,
            executed.response,
            task,
            execution.inventory_limits,
        )
        .map_err(super::analysis_task_pool::protocol)?;
        if observations.insert(*sample_index, imported).is_some() {
            return Err(NativeExecutionError::Protocol(
                "frequency sample observations were assembled more than once".into(),
            ));
        }
    }
    Ok(fanouts
        .iter()
        .enumerate()
        .map(
            |(index, (sample, _, _))| CompletedStructuralFrequencyObservationSample {
                sample: *sample,
                observations: observations.remove(&index),
            },
        )
        .collect())
}

async fn execute_histories(
    pool: &mut impl AnalysisTaskPool,
    execution: &StructuralFrequencyObservationExecution<'_>,
    policy: &StructuralSequenceExecutionPolicy,
    fanouts: &[(
        RecoveredFieldSample,
        AnalysisTaskDependency,
        runmat_analysis_execution::PreparedStructuralObservationFanout,
    )],
    reduced: &[ReducedRecord],
    resource_attempts: &mut Vec<runmat_analysis_execution::PlannedResourceUsageAttempt>,
) -> NativeExecutionResult<Vec<CompletedStructuralFrequencyHistory>> {
    let history_ids = execution
        .step
        .observations
        .iter()
        .filter_map(|request| match request {
            ObservationRequest::History(spec) => Some(spec.observation_id.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let mut checkpoints = BTreeMap::<
        ObservationId,
        (
            ImportedObservationHistoryCheckpointTask,
            AnalysisTaskDependency,
        ),
    >::new();
    for (sample_index, (_, _, fanout)) in fanouts.iter().enumerate() {
        let mut origins = Vec::new();
        let mut tasks = Vec::new();
        for record in reduced {
            if record.sample_index != sample_index
                || !history_ids.contains(&record.result.result.observation_id)
            {
                continue;
            }
            let observation_id = record.result.result.observation_id.clone();
            let previous = checkpoints
                .get(&observation_id)
                .map(|(checkpoint, _)| checkpoint.objects.clone());
            let mut dependencies = vec![record.dependency];
            if let Some((_, dependency)) = checkpoints.get(&observation_id) {
                dependencies.push(*dependency);
            }
            let task = prepare_observation_history_checkpoint_task(
                ObservationHistoryCheckpointPreparation {
                    source_host: record.source_host.clone(),
                    plan: fanout.source_request.plan.clone(),
                    observation_id: observation_id.clone(),
                    sample: record.result.clone(),
                    previous,
                    program_revision: execution.program_revision.clone(),
                    chunk_bytes: execution.chunk_bytes,
                    inventory_limits: execution.inventory_limits,
                },
            )
            .map_err(super::analysis_task_pool::protocol)?;
            origins.push((observation_id, dependencies));
            tasks.push(task);
        }
        let executed = execute_analysis_task_batch(
            pool,
            tasks
                .iter()
                .zip(&origins)
                .map(|(task, (_, dependencies))| PreparedAnalysisTask {
                    host: &task.host,
                    program: &task.program,
                    resource_plan: Some(&task.resource_plan),
                    fallback_resources: policy.resources.clone(),
                    dependencies: dependencies.clone(),
                })
                .collect(),
            batch_policy(policy),
        )
        .await?;
        for ((task, (observation_id, _)), executed) in tasks.iter().zip(&origins).zip(executed) {
            resource_attempts.push(resource_attempt(&task.resource_plan, &executed)?);
            let dependency = AnalysisTaskDependency {
                identity: executed.identity,
                task_id: executed.task_id,
            };
            let imported = import_observation_history_checkpoint_task(
                pool,
                executed.response,
                task,
                execution.inventory_limits,
            )
            .map_err(super::analysis_task_pool::protocol)?;
            checkpoints.insert(observation_id.clone(), (imported, dependency));
        }
    }
    let mut histories = Vec::new();
    for request in &execution.step.observations {
        let ObservationRequest::History(spec) = request else {
            continue;
        };
        let (checkpoint, _) = checkpoints.remove(&spec.observation_id).ok_or_else(|| {
            NativeExecutionError::Protocol(format!(
                "frequency history `{}` produced no scheduled samples",
                spec.observation_id.0
            ))
        })?;
        histories.push(CompletedStructuralFrequencyHistory {
            observation_id: spec.observation_id.clone(),
            checkpoint,
        });
    }
    Ok(histories)
}
