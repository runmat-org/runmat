use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runmat_analysis_core::PhysicalSignal;
use runmat_analysis_execution::{
    build_signal_generator_task_submission, canonicalize_signal_generator_outputs,
    materialize_signal_for_solver, prepare_analysis_objects, PreparedAnalysisObjects,
    PreparedSignalMaterialization, SignalGeneratorChunkOutput, SignalGeneratorExecutionContext,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, SignalMaterializationIdentity, DEFAULT_ANALYSIS_CHUNK_BYTES,
};
use runmat_execution::resource::Capability;
use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValueLimits;
use runmat_execution::TaskId;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use super::RemotePoolDriver;
use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Clone, Debug)]
pub(crate) struct CompletedRemoteSignalMaterialization {
    pub signal: PhysicalSignal,
    pub objects: PreparedAnalysisObjects,
}

#[derive(Clone, Debug)]
pub(super) enum RemoteSignalMaterializationOutcome {
    Completed(Box<CompletedRemoteSignalMaterialization>),
    Failed(String),
    Cancelled,
    Indeterminate(String),
}

pub(super) async fn execute(
    pool: Arc<RemotePoolDriver>,
    materializations: impl IntoIterator<Item = PreparedSignalMaterialization>,
    artifact_access: &ArtifactAccess,
    resources: &AllocationResources,
) -> NativeExecutionResult<
    BTreeMap<SignalMaterializationIdentity, RemoteSignalMaterializationOutcome>,
> {
    let mut unique = BTreeMap::new();
    for prepared in materializations {
        prepared.validate().map_err(protocol)?;
        let identity = prepared.admitted.materialization_identity;
        if let Some(existing) = unique.get(&identity) {
            if existing != &prepared {
                return Err(protocol(
                    "remote signal materialization identity collides with different input",
                ));
            }
            continue;
        }
        unique.insert(identity, prepared);
    }
    if unique.is_empty() {
        return Ok(BTreeMap::new());
    }

    let mut task_resources = super::pool_resources::task_resources(resources)?;
    task_resources.max_artifact_bytes = ValueLimits::default().max_inline_bytes;
    task_resources.required_capabilities = BTreeSet::from([Capability::ProcessIsolation]);
    let deadline = super::pool_resources::task_deadline_unix_millis(resources)?;
    let mut tasks = Vec::new();
    let mut task_metadata = Vec::new();
    for (identity, prepared) in &unique {
        for (chunk_index, chunk) in prepared.program.chunks.iter().enumerate() {
            let submission = build_signal_generator_task_submission(
                &prepared.admitted,
                chunk,
                BTreeSet::new(),
                SignalGeneratorExecutionContext {
                    scope_id: pool.scope_id(),
                    pool_id: pool.pool_id(),
                    resources: task_resources.clone(),
                    retry: RetryPolicy::IdempotentInfrastructure,
                    deadline_unix_millis: Some(deadline),
                    priority: 0,
                },
            )
            .map_err(protocol)?;
            task_metadata.push((*identity, chunk_index, submission.request.id));
            tasks.push((chunk.request.clone(), submission));
        }
    }
    let completions = pool.submit_graph(tasks)?;
    if completions.len() != task_metadata.len() {
        return Err(protocol(
            "remote signal task handles differ from the admitted chunk inventory",
        ));
    }

    let mut chunk_results = unique
        .iter()
        .map(|(identity, prepared)| {
            (
                *identity,
                (0..prepared.program.chunks.len())
                    .map(|_| None)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for (completion, (identity, chunk_index, task_id)) in completions.into_iter().zip(task_metadata)
    {
        let outcome = completion.wait().await.and_then(|success| {
            if !success.result_objects.is_empty() || success.outputs.len() != 1 {
                Err("bounded signal task returned a non-inline or non-scalar output list".into())
            } else {
                Ok(success.outputs[0].clone())
            }
        });
        chunk_results
            .get_mut(&identity)
            .and_then(|chunks| chunks.get_mut(chunk_index))
            .ok_or_else(|| protocol("remote signal completion is outside its chunk inventory"))?
            .replace((task_id, outcome));
    }

    let limits = ObjectInventoryLimits::default();
    let mut outcomes = BTreeMap::new();
    for (identity, prepared) in unique {
        let results = chunk_results
            .remove(&identity)
            .ok_or_else(|| protocol("remote signal chunk results are absent"))?;
        let mut outputs = Vec::with_capacity(results.len());
        let mut terminal = None;
        for (chunk, result) in prepared.program.chunks.iter().zip(results) {
            let (task_id, result) = result
                .ok_or_else(|| protocol("remote signal task did not reach a terminal state"))?;
            match result {
                Ok(value) => outputs.push(SignalGeneratorChunkOutput {
                    chunk_identity: chunk.plan.identity,
                    value,
                }),
                Err(message) => {
                    terminal = Some(task_terminal(pool.as_ref(), task_id, message));
                    break;
                }
            }
        }
        let outcome = if let Some(terminal) = terminal {
            terminal
        } else {
            let realization = canonicalize_signal_generator_outputs(
                &prepared.authored,
                &prepared.admitted,
                None,
                &prepared
                    .program
                    .chunks
                    .iter()
                    .map(|chunk| chunk.plan.clone())
                    .collect::<Vec<_>>(),
                &outputs,
            )
            .map_err(protocol)?;
            let objects = prepare_analysis_objects(
                AnalysisNumericalObjectKind::Signal,
                realization.identity.digest(),
                &realization.canonical_bytes().map_err(protocol)?,
                DEFAULT_ANALYSIS_CHUNK_BYTES,
                ObjectNamespace::ResultValue,
                limits,
            )
            .map_err(protocol)?;
            for (object, reference) in objects.objects.iter().zip(
                objects
                    .references(
                        artifact_access,
                        runmat_execution::value::ValueRefKind::ResultObject,
                    )
                    .map_err(protocol)?,
            ) {
                pool.register_execution_object(reference, object.bytes.clone())?;
            }
            RemoteSignalMaterializationOutcome::Completed(Box::new(
                CompletedRemoteSignalMaterialization {
                    signal: materialize_signal_for_solver(
                        &prepared.authored,
                        &prepared.admitted,
                        &realization,
                    )
                    .map_err(protocol)?,
                    objects,
                },
            ))
        };
        outcomes.insert(identity, outcome);
    }
    Ok(outcomes)
}

pub(super) fn apply_to_workload(
    workload: &mut runmat_analysis_fea::execution::StructuralLinearWorkload,
    required: &[PreparedSignalMaterialization],
    outcomes: &BTreeMap<SignalMaterializationIdentity, RemoteSignalMaterializationOutcome>,
) -> Result<Vec<CompletedRemoteSignalMaterialization>, RemoteSignalMaterializationOutcome> {
    let mut replacements = BTreeMap::new();
    let mut completed = Vec::with_capacity(required.len());
    for prepared in required {
        let outcome = outcomes
            .get(&prepared.admitted.materialization_identity)
            .ok_or_else(|| {
                RemoteSignalMaterializationOutcome::Failed(
                    "remote signal materialization result is absent".into(),
                )
            })?;
        let result = match outcome {
            RemoteSignalMaterializationOutcome::Completed(result) => result.as_ref().clone(),
            terminal => return Err(terminal.clone()),
        };
        replacements.insert(prepared.authored.signal_id().clone(), result.signal.clone());
        completed.push(result);
    }
    workload.model.signals = workload
        .model
        .signals
        .iter()
        .map(|signal| {
            if signal.generator().is_none() {
                signal.clone()
            } else {
                replacements
                    .get(signal.signal_id())
                    .cloned()
                    .unwrap_or_else(|| signal.clone())
            }
        })
        .collect();
    if workload
        .model
        .signals
        .iter()
        .any(|signal| signal.generator().is_some())
    {
        return Err(RemoteSignalMaterializationOutcome::Failed(
            "remote study retained an unmaterialized signal generator".into(),
        ));
    }
    Ok(completed)
}

fn task_terminal(
    pool: &RemotePoolDriver,
    task_id: TaskId,
    message: String,
) -> RemoteSignalMaterializationOutcome {
    match pool.snapshot().tasks.get(&task_id).map(|task| task.state) {
        Some(TaskState::Indeterminate) => {
            RemoteSignalMaterializationOutcome::Indeterminate(message)
        }
        Some(TaskState::Cancelled) => RemoteSignalMaterializationOutcome::Cancelled,
        _ => RemoteSignalMaterializationOutcome::Failed(message),
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
