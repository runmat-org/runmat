use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use runmat_analysis_core::{PhysicalSignal, SignalId};
use runmat_analysis_execution::{
    admit_callable_signal_generator, build_generator_task_submission,
    canonicalize_signal_generator_outputs, materialize_signal_for_solver,
    plan_signal_generator_chunks, prepare_analysis_objects, prepare_signal_materialization,
    AdmittedSignalGenerator, GeneratorChunkOutput, GeneratorExecutionContext,
    PreparedSignalMaterialization,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, SignalMaterializationIdentity, DEFAULT_ANALYSIS_CHUNK_BYTES,
};
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValueLimits;
use runmat_execution::{CancellationReason, Digest, ExecutableUnitEnvelope};
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{BundleCallable, ObjectNamespace};
use runmat_execution_runner_native::{
    FilesystemObjectStore, NativeExecutionConfig, NativeProgramSession,
};

const POLL_INTERVAL: Duration = Duration::from_millis(10);
const INPUT_TASK_MEMORY_BYTES: u64 = 512 * 1024 * 1024;
const INPUT_TASK_SCRATCH_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Clone, Debug)]
struct CompiledSignalGenerator {
    signal_id: SignalId,
    admission: AdmittedSignalGenerator,
    executable: ExecutableUnitEnvelope,
}

impl CompiledSignalGenerator {
    fn validate(&self) -> Result<()> {
        self.executable.validate().with_context(|| {
            format!(
                "validate portable executable for FEA signal `{}`",
                self.signal_id.0
            )
        })?;
        let identity = &self.executable.manifest.identity;
        if identity.program != self.admission.program_revision
            || identity.root_package != self.admission.callable.owner_identity
            || identity.entrypoint_function != Some(self.admission.function)
            || identity.source_digest != self.admission.executable_source_digest
        {
            bail!(
                "portable executable for FEA signal `{}` differs from its admitted callable",
                self.signal_id.0
            );
        }
        Ok(())
    }
}

pub(crate) fn materialize_runtime_signal(
    request: runmat_runtime::context::RuntimeSignalMaterializationRequest,
) -> Result<PhysicalSignal> {
    let program = request.program;
    program.executable.validate()?;
    let identity = &program.executable.manifest.identity;
    // The runtime carries only the frozen product; its compiler facts are
    // re-derived here from the same envelope, never from live session state.
    let unit = runmat_core::ExecutableUnit::from_portable_envelope(&program.executable)
        .map_err(anyhow::Error::msg)?;
    let function = identity
        .entrypoint_function
        .and_then(|entrypoint| unit.function_analysis(entrypoint))
        .ok_or_else(|| anyhow::anyhow!("frozen signal executable has no entrypoint analysis"))?
        .clone();
    if identity.root_package != program.owner_identity
        || identity.entrypoint != program.display_name
        || identity.source_digest != program.executable_source_digest
    {
        bail!("runtime signal callable differs from its frozen executable product");
    }
    let compiled_binding =
        runmat_analysis_execution::named_callable::CompiledCallableBinding::from_selected_identity(
            identity,
            &program.display_name,
        )?;
    let admission = admit_callable_signal_generator(
        &request.signal,
        identity.program.clone(),
        BundleCallable {
            owner_identity: program.owner_identity,
            qualified_name: program.display_name,
            source_digest: program.source_digest,
        },
        compiled_binding,
        &function,
        request.callable,
        &BTreeSet::new(),
    )?;
    let compiled = CompiledSignalGenerator {
        signal_id: request.signal.signal_id().clone(),
        admission,
        executable: program.executable,
    };
    let materialized = execute_compiled_materializations(
        std::slice::from_ref(&request.signal),
        std::slice::from_ref(&compiled),
        request.maximum_wall_millis,
    )?;
    materialized
        .into_iter()
        .next()
        .context("runtime signal materialization returned no signal")
}

fn execute_compiled_materializations(
    generator_signals: &[PhysicalSignal],
    compiled: &[CompiledSignalGenerator],
    maximum_wall_millis: u64,
) -> Result<Vec<PhysicalSignal>> {
    let mut native_config = NativeExecutionConfig::for_current_executable()?;
    native_config
        .worker_capabilities
        .insert(Capability::ProcessIsolation);
    let maximum_parallel_tasks = native_config.max_workers as usize;
    let session = NativeProgramSession::new(native_config)?;
    let prepared = prepare_compiled_materializations(generator_signals, compiled)?;
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
    for materialization in unique.values() {
        materialization.validate_closed()?;
        for object in &materialization.input_objects {
            session.register_value_object(object.clone())?;
        }
    }
    let resources = ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: INPUT_TASK_MEMORY_BYTES,
        scratch_bytes: INPUT_TASK_SCRATCH_BYTES,
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
            let mut chunk_resources = resources.clone();
            chunk_resources
                .required_capabilities
                .extend(chunk.graph_capability()?);
            chunk_identities.push(chunk.plan.identity);
            submissions.push((
                chunk.request.clone(),
                build_generator_task_submission(
                    &materialization.admitted,
                    chunk,
                    BTreeSet::new(),
                    GeneratorExecutionContext {
                        scope_id: session.scope_id(),
                        pool_id: session.pool_id(),
                        host: chunk.execution_host_requirement(BTreeSet::from([
                            runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
                        ]))?,
                        resources: chunk_resources,
                        retry: RetryPolicy::IdempotentInfrastructure,
                        deadline_unix_millis: None,
                        priority: 0,
                    },
                )?,
            ));
        }
    }
    let tasks = session.submit_workers(submissions)?;
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
        FilesystemObjectStore::open(input_store_root()?, limits.max_object_bytes)?;
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
            &materialization.input_admissions,
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
    generator_signals
        .iter()
        .zip(compiled)
        .map(|(_, compiled)| {
            replacements
                .get(&compiled.admission.materialization_identity)
                .cloned()
                .context("materialized signal replacement is absent")
        })
        .collect()
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
            let plan = plan_signal_generator_chunks(
                &authored,
                &compiled.admission,
                None,
                maximum_inline_result_bytes,
                |artifact| {
                    Err(runmat_analysis_execution::AnalysisBridgeError::Invalid(
                        format!(
                            "runtime signal artifact `{}` has no immutable object binding",
                            artifact.artifact_id
                        ),
                    ))
                },
            )?;
            Ok(prepare_signal_materialization(
                authored,
                compiled.admission.clone(),
                compiled.executable.clone(),
                plan.chunks,
                Vec::new(),
                plan.input_admissions,
                None,
            )?)
        })
        .collect()
}

fn wait_for_outputs(
    session: &NativeProgramSession,
    tasks: Vec<runmat_execution_runner_native::NativeProgramTask>,
    chunk_identities: &[Digest],
    maximum_wall_millis: u64,
) -> Result<Vec<GeneratorChunkOutput>> {
    if tasks.len() != chunk_identities.len() {
        bail!("input task handles differ from the prepared chunk inventory");
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
                        bail!("input materialization task failed: {message}");
                    }
                };
                let output = runmat_analysis_execution::generator_chunk_output_from_success(
                    chunk_identities[index],
                    success,
                )
                .inspect_err(|_| {
                    session.cancel(CancellationReason::ParentScope);
                })?;
                outputs[index] = Some(output.value);
            }
        }
        if outputs.iter().all(Option::is_some) {
            break;
        }
        if started.elapsed() >= Duration::from_millis(maximum_wall_millis) {
            session.cancel(CancellationReason::Deadline);
            bail!("input materialization exceeded its authored wall-time limit");
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    outputs
        .into_iter()
        .enumerate()
        .map(|(index, output)| {
            Ok(GeneratorChunkOutput {
                chunk_identity: chunk_identities[index],
                value: output.context("input task completed without a result")?,
            })
        })
        .collect()
}

fn materialization_wait_millis(
    task_wall_millis: u64,
    task_count: usize,
    maximum_parallel_tasks: usize,
) -> Result<u64> {
    if task_count == 0 || maximum_parallel_tasks == 0 {
        bail!("input materialization requires tasks and available workers");
    }
    let waves = task_count.div_ceil(maximum_parallel_tasks);
    let intervals = u64::try_from(waves)
        .context("input task wave count exceeds the host")?
        .checked_add(1)
        .context("input materialization wait interval overflows")?;
    task_wall_millis
        .checked_mul(intervals)
        .context("input materialization wait duration overflows")
}

fn input_store_root() -> Result<std::path::PathBuf> {
    let configured = std::env::var_os("RUNMAT_FEA_ARTIFACT_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(runmat_runtime::analysis::default_fea_artifact_root);
    let root = if configured.is_absolute() {
        configured
    } else {
        std::env::current_dir()?.join(configured)
    };
    Ok(root.join("generated-inputs").join("objects"))
}
