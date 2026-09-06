use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use runmat_analysis_core::{AnalysisModel, PhysicalSignal};
use runmat_analysis_execution::{
    build_signal_generator_task_submission, canonicalize_signal_generator_outputs,
    materialize_signal_for_solver, plan_signal_generator_chunks, prepare_analysis_objects,
    prepare_signal_materialization, PreparedSignalMaterialization, SignalGeneratorChunkOutput,
    SignalGeneratorExecutionContext,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, SignalMaterializationIdentity, DEFAULT_ANALYSIS_CHUNK_BYTES,
};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValueLimits;
use runmat_execution::CancellationReason;
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ObjectNamespace;
use runmat_execution_runner_native::{
    FilesystemObjectStore, NativeExecutionConfig, NativeProgramSession,
};

use crate::cli::Cli;

use super::callables::{
    compile_signal_generators, compile_signal_generators_in_frozen, CompiledSignalGenerator,
};

const POLL_INTERVAL: Duration = Duration::from_millis(10);
const SIGNAL_TASK_MEMORY_BYTES: u64 = 512 * 1024 * 1024;
const SIGNAL_TASK_SCRATCH_BYTES: u64 = 128 * 1024 * 1024;

pub(super) async fn materialize_study(
    document_path: &std::path::Path,
    study: &mut runmat_runtime::analysis::AnalysisStudySpec,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    let Some(model) = study.model.as_mut() else {
        return Ok(());
    };
    let wall_millis = materialization_wall_millis(model);
    model.signals =
        materialize_signals(document_path, &model.signals, wall_millis, cli, config).await?;
    Ok(())
}

pub(super) async fn materialize_sweep(
    document_path: &std::path::Path,
    sweep: &mut runmat_runtime::analysis::AnalysisStudySweepSpec,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    let mut signals = Vec::new();
    let mut study_signal_counts = Vec::with_capacity(sweep.studies.len());
    let mut wall_millis = 1_u64;
    for study in &sweep.studies {
        let Some(model) = &study.model else {
            study_signal_counts.push(0);
            continue;
        };
        wall_millis = wall_millis.max(materialization_wall_millis(model));
        study_signal_counts.push(model.signals.len());
        signals.extend(model.signals.iter().cloned());
    }
    let mut materialized = materialize_signals(document_path, &signals, wall_millis, cli, config)
        .await?
        .into_iter();
    for (study, signal_count) in sweep.studies.iter_mut().zip(study_signal_counts) {
        if signal_count == 0 {
            continue;
        }
        let model = study
            .model
            .as_mut()
            .context("sweep study model disappeared during signal materialization")?;
        model.signals = materialized.by_ref().take(signal_count).collect();
        if model.signals.len() != signal_count {
            bail!("signal materialization returned an incomplete sweep inventory");
        }
    }
    if materialized.next().is_some() {
        bail!("signal materialization returned an oversized sweep inventory");
    }
    Ok(())
}

async fn materialize_signals(
    document_path: &std::path::Path,
    signals: &[PhysicalSignal],
    maximum_wall_millis: u64,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<Vec<PhysicalSignal>> {
    let generator_signals = signals
        .iter()
        .filter(|signal| signal.generator().is_some())
        .cloned()
        .collect::<Vec<_>>();
    if generator_signals.is_empty() {
        return Ok(signals.to_vec());
    }
    let compiled = compile_signal_generators(
        document_path,
        &generator_signals,
        cli,
        config,
        &BTreeSet::new(),
    )
    .await?;
    if compiled.len() != generator_signals.len() {
        bail!("compiled signal generator inventory is incomplete");
    }
    let prepared = prepare_compiled_materializations(&generator_signals, &compiled)?;
    let mut unique =
        BTreeMap::<SignalMaterializationIdentity, PreparedSignalMaterialization>::new();
    for prepared in prepared {
        let identity = prepared.admitted.materialization_identity;
        if let Some(existing) = unique.get(&identity) {
            if existing != &prepared {
                bail!("signal materialization identity collides with different frozen input");
            }
            continue;
        }
        unique.insert(identity, prepared);
    }

    let mut native_config = NativeExecutionConfig::for_current_executable()?;
    native_config
        .worker_capabilities
        .insert(Capability::ProcessIsolation);
    let maximum_parallel_tasks = native_config.max_workers as usize;
    let session = NativeProgramSession::new(native_config)?;
    let resources = ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: SIGNAL_TASK_MEMORY_BYTES,
        scratch_bytes: SIGNAL_TASK_SCRATCH_BYTES,
        max_wall_millis: maximum_wall_millis.max(1),
        max_artifact_bytes: ValueLimits::default().max_inline_bytes,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
    };
    let mut submissions = Vec::new();
    let mut chunk_identities = Vec::new();
    for materialization in unique.values() {
        for chunk in &materialization.program.chunks {
            chunk_identities.push(chunk.plan.identity);
            submissions.push((
                chunk.request.clone(),
                build_signal_generator_task_submission(
                    &materialization.admitted,
                    chunk,
                    BTreeSet::new(),
                    SignalGeneratorExecutionContext {
                        scope_id: session.scope_id(),
                        pool_id: session.pool_id(),
                        resources: resources.clone(),
                        retry: RetryPolicy::IdempotentInfrastructure,
                        deadline_unix_millis: None,
                        priority: 0,
                    },
                )?,
            ));
        }
    }
    let tasks = session.submit_graph(submissions)?;
    let outputs = wait_for_outputs(
        &session,
        tasks,
        &chunk_identities,
        materialization_wait_millis(
            maximum_wall_millis.max(1),
            chunk_identities.len(),
            maximum_parallel_tasks,
        )?,
    )?;

    let limits = ObjectInventoryLimits::default();
    let mut persistent_store =
        FilesystemObjectStore::open(signal_store_root()?, limits.max_object_bytes)?;
    let mut replacements = BTreeMap::new();
    let mut cursor = 0_usize;
    for (identity, materialization) in &unique {
        let chunk_count = materialization.program.chunks.len();
        let end = cursor
            .checked_add(chunk_count)
            .context("signal chunk output inventory overflows")?;
        let chunk_outputs = outputs
            .get(cursor..end)
            .context("signal task results do not cover the prepared chunk inventory")?;
        let realization = canonicalize_signal_generator_outputs(
            &materialization.authored,
            &materialization.admitted,
            None,
            &materialization
                .program
                .chunks
                .iter()
                .map(|chunk| chunk.plan.clone())
                .collect::<Vec<_>>(),
            chunk_outputs,
        )?;
        let canonical_bytes = realization.canonical_bytes()?;
        let objects = prepare_analysis_objects(
            AnalysisNumericalObjectKind::Signal,
            realization.identity.digest(),
            &canonical_bytes,
            DEFAULT_ANALYSIS_CHUNK_BYTES,
            ObjectNamespace::ResultValue,
            limits,
        )?;
        for object in &objects.objects {
            persistent_store.write_verified(object)?;
        }
        replacements.insert(
            *identity,
            materialize_signal_for_solver(
                &materialization.authored,
                &materialization.admitted,
                &realization,
            )?,
        );
        cursor = end;
    }
    if cursor != outputs.len() {
        bail!("signal task results exceed the prepared chunk inventory");
    }

    let mut compiled = compiled.iter();
    signals
        .iter()
        .map(|signal| {
            if signal.generator().is_none() {
                return Ok(signal.clone());
            }
            let compiled = compiled
                .next()
                .context("compiled signal generator order is incomplete")?;
            replacements
                .get(&compiled.admission.materialization_identity)
                .cloned()
                .context("materialized signal replacement is absent")
        })
        .collect()
}

pub(super) async fn prepare_signal_materializations_in_frozen(
    document_path: &std::path::Path,
    signals: &[PhysicalSignal],
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
) -> Result<Vec<PreparedSignalMaterialization>> {
    let generator_signals = signals
        .iter()
        .filter(|signal| signal.generator().is_some())
        .cloned()
        .collect::<Vec<_>>();
    let compiled = compile_signal_generators_in_frozen(
        document_path,
        &generator_signals,
        frozen,
        config,
        &BTreeSet::new(),
    )
    .await?;
    prepare_compiled_materializations(&generator_signals, &compiled)
}

fn prepare_compiled_materializations(
    generator_signals: &[PhysicalSignal],
    compiled: &[CompiledSignalGenerator],
) -> Result<Vec<PreparedSignalMaterialization>> {
    if compiled.len() != generator_signals.len() {
        bail!("compiled signal generator inventory is incomplete");
    }
    let maximum_inline_result_bytes = usize::try_from(ValueLimits::default().max_inline_bytes)
        .context("inline result bound exceeds this host")?;
    generator_signals
        .iter()
        .cloned()
        .zip(compiled)
        .map(|(authored, compiled)| {
            compiled.validate()?;
            let chunks = plan_signal_generator_chunks(
                &authored,
                &compiled.admission,
                None,
                maximum_inline_result_bytes,
                |_| {
                    Err(runmat_analysis_execution::AnalysisBridgeError::Invalid(
                        "generator artifact arguments require the immutable artifact resolver"
                            .into(),
                    ))
                },
            )?;
            Ok(prepare_signal_materialization(
                authored,
                compiled.admission.clone(),
                compiled.executable.clone(),
                chunks,
                Vec::new(),
            )?)
        })
        .collect()
}

fn wait_for_outputs(
    session: &NativeProgramSession,
    tasks: Vec<runmat_execution_runner_native::NativeProgramTask>,
    chunk_identities: &[runmat_execution::Digest],
    maximum_wall_millis: u64,
) -> Result<Vec<SignalGeneratorChunkOutput>> {
    if tasks.len() != chunk_identities.len() {
        bail!("signal task handles differ from the prepared chunk inventory");
    }
    let started = Instant::now();
    let mut outputs = (0..tasks.len()).map(|_| None).collect::<Vec<_>>();
    while outputs.iter().any(Option::is_none) {
        for (index, task) in tasks.iter().enumerate() {
            if outputs[index].is_some() {
                continue;
            }
            if let Some(result) = task.try_result() {
                let success = match result {
                    Ok(success) => success,
                    Err(message) => {
                        session.cancel(CancellationReason::ParentScope);
                        bail!("signal materialization task failed: {message}");
                    }
                };
                if !success.result_objects.is_empty() || success.outputs.len() != 1 {
                    session.cancel(CancellationReason::ParentScope);
                    bail!("bounded signal task returned a non-inline or non-scalar output list");
                }
                outputs[index] = Some(success.outputs[0].clone());
            }
        }
        if outputs.iter().all(Option::is_some) {
            break;
        }
        if started.elapsed() >= Duration::from_millis(maximum_wall_millis) {
            session.cancel(CancellationReason::Deadline);
            bail!("signal materialization exceeded its authored wall-time limit");
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    outputs
        .into_iter()
        .enumerate()
        .map(|(index, output)| {
            Ok(SignalGeneratorChunkOutput {
                chunk_identity: chunk_identities[index],
                value: output.context("signal task completed without a result")?,
            })
        })
        .collect()
}

fn materialization_wall_millis(model: &AnalysisModel) -> u64 {
    model
        .sequence
        .steps
        .iter()
        .map(|step| step.accuracy.resources.maximum_wall_time_milliseconds)
        .max()
        .unwrap_or(60_000)
}

fn materialization_wait_millis(
    task_wall_millis: u64,
    task_count: usize,
    maximum_parallel_tasks: usize,
) -> Result<u64> {
    if task_count == 0 || maximum_parallel_tasks == 0 {
        bail!("signal materialization requires tasks and available workers");
    }
    let waves = task_count.div_ceil(maximum_parallel_tasks);
    let intervals = u64::try_from(waves)
        .context("signal task wave count exceeds the host")?
        .checked_add(1)
        .context("signal materialization wait interval overflows")?;
    task_wall_millis
        .checked_mul(intervals)
        .context("signal materialization wait duration overflows")
}

fn signal_store_root() -> Result<std::path::PathBuf> {
    let configured = std::env::var_os("RUNMAT_FEA_ARTIFACT_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(runmat_runtime::analysis::default_fea_artifact_root);
    let root = if configured.is_absolute() {
        configured
    } else {
        std::env::current_dir()?.join(configured)
    };
    Ok(root.join("generated-signals").join("objects"))
}
