use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runmat_analysis_execution::{
    build_generator_task_submission, prepare_analysis_objects, GeneratorChunkOutput,
    GeneratorExecutionContext, PreparedInputMaterialization,
};
use runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES;
use runmat_execution::resource::Capability;
use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValueLimits;
use runmat_execution::TaskId;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use super::RemotePoolDriver;
use crate::input_materialization::{materialize_input, CompletedInputMaterialization};
use crate::{NativeExecutionError, NativeExecutionResult};

#[derive(Clone, Debug)]
pub(super) enum RemoteInputMaterializationOutcome {
    Completed(Box<CompletedInputMaterialization>),
    Failed(String),
    Cancelled,
    Indeterminate(String),
}

pub(super) async fn execute(
    pool: Arc<RemotePoolDriver>,
    materializations: impl IntoIterator<Item = PreparedInputMaterialization>,
    artifact_access: &ArtifactAccess,
    resources: &AllocationResources,
) -> NativeExecutionResult<BTreeMap<runmat_execution::Digest, RemoteInputMaterializationOutcome>> {
    let mut unique = BTreeMap::new();
    for prepared in materializations {
        prepared.validate().map_err(protocol)?;
        for object in prepared.input_objects() {
            pool.register_value_object(object.reference.clone(), object.bytes.clone())?;
        }
        validate_registered_inputs(pool.as_ref(), &prepared)?;
        let identity = prepared.materialization_digest();
        if let Some(existing) = unique.get(&identity) {
            if existing != &prepared {
                return Err(protocol(
                    "remote input materialization identity collides with different input",
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
        for (chunk_index, chunk) in prepared.program().chunks.iter().enumerate() {
            let context = GeneratorExecutionContext {
                scope_id: pool.scope_id(),
                pool_id: pool.pool_id(),
                resources: task_resources.clone(),
                retry: RetryPolicy::IdempotentInfrastructure,
                deadline_unix_millis: Some(deadline),
                priority: 0,
            };
            let submission = match prepared {
                PreparedInputMaterialization::Signal(prepared) => build_generator_task_submission(
                    &prepared.admitted,
                    chunk,
                    BTreeSet::new(),
                    context,
                ),
                PreparedInputMaterialization::Numerical(prepared) => {
                    build_generator_task_submission(
                        &prepared.admitted,
                        chunk,
                        BTreeSet::new(),
                        context,
                    )
                }
            }
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
                (0..prepared.program().chunks.len())
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
        for (chunk, result) in prepared.program().chunks.iter().zip(results) {
            let (task_id, result) = result
                .ok_or_else(|| protocol("remote signal task did not reach a terminal state"))?;
            match result {
                Ok(value) => outputs.push(GeneratorChunkOutput {
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
            let chunks = prepared
                .program()
                .chunks
                .iter()
                .map(|chunk| chunk.plan.clone())
                .collect::<Vec<_>>();
            let (input, kind, identity, bytes) = materialize_input(&prepared, &chunks, &outputs)?;
            let objects = prepare_analysis_objects(
                kind,
                identity,
                &bytes,
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
            RemoteInputMaterializationOutcome::Completed(Box::new(CompletedInputMaterialization {
                input,
                objects,
            }))
        };
        outcomes.insert(identity, outcome);
    }
    Ok(outcomes)
}

fn validate_registered_inputs(
    pool: &RemotePoolDriver,
    prepared: &PreparedInputMaterialization,
) -> NativeExecutionResult<()> {
    let (admissions, program) = match prepared {
        PreparedInputMaterialization::Signal(value) => (&value.input_admissions, &value.program),
        PreparedInputMaterialization::Numerical(value) => (&value.input_admissions, &value.program),
    };
    for admission in admissions {
        let reference = program
            .input_references
            .iter()
            .find(|reference| reference.logical_digest == admission.logical_digest)
            .ok_or_else(|| protocol("signal input admission has no registered value reference"))?;
        let value = pool.resolve_value_object(reference)?;
        runmat_analysis_execution::validate_signal_input_payload(admission, &value)
            .map_err(protocol)?;
    }
    Ok(())
}

pub(super) fn apply_to_model(
    model: &mut runmat_analysis_core::AnalysisModel,
    required: &[PreparedInputMaterialization],
    outcomes: &BTreeMap<runmat_execution::Digest, RemoteInputMaterializationOutcome>,
) -> Result<Vec<CompletedInputMaterialization>, RemoteInputMaterializationOutcome> {
    let mut completed = Vec::with_capacity(required.len());
    for prepared in required {
        let outcome = outcomes
            .get(&prepared.materialization_digest())
            .ok_or_else(|| {
                RemoteInputMaterializationOutcome::Failed(
                    "remote input materialization result is absent".into(),
                )
            })?;
        let result = match outcome {
            RemoteInputMaterializationOutcome::Completed(result) => result.as_ref().clone(),
            terminal => return Err(terminal.clone()),
        };
        completed.push(result);
    }
    crate::input_materialization::apply_to_model(model, completed)
        .map_err(|error| RemoteInputMaterializationOutcome::Failed(error.to_string()))
}

fn task_terminal(
    pool: &RemotePoolDriver,
    task_id: TaskId,
    message: String,
) -> RemoteInputMaterializationOutcome {
    match pool.snapshot().tasks.get(&task_id).map(|task| task.state) {
        Some(TaskState::Indeterminate) => RemoteInputMaterializationOutcome::Indeterminate(message),
        Some(TaskState::Cancelled) => RemoteInputMaterializationOutcome::Cancelled,
        _ => RemoteInputMaterializationOutcome::Failed(message),
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
