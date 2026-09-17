use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use runmat_analysis_core::{
    AnalysisModel, PhysicalSignal, RandomExcitationSpectrum, ResponseSpectrum,
};
use runmat_analysis_execution::{
    admit_callable_signal_generator, build_generator_task_submission,
    canonical_materialized_random_spectrum_bytes, canonical_materialized_response_spectrum_bytes,
    canonicalize_signal_generator_outputs, materialize_random_spectrum,
    materialize_response_spectrum, materialize_signal_for_solver,
    plan_random_spectrum_generator_chunks, plan_response_spectrum_generator_chunks,
    plan_signal_generator_chunks, prepare_analysis_objects, prepare_numerical_materialization,
    prepare_signal_materialization, GeneratorChunkOutput, GeneratorExecutionContext,
    NumericalGeneratorSource, PreparedNumericalMaterialization, PreparedSignalMaterialization,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, SignalMaterializationIdentity, DEFAULT_ANALYSIS_CHUNK_BYTES,
};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{ValueLimits, ValuePayload, ValueRefKind};
use runmat_execution::{CancellationReason, Digest};
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, BundleCallable, ObjectNamespace, ProgramExecutionObject,
    EXTERNALIZED_VALUE_SCHEMA, MAX_PROGRAM_EXECUTION_OBJECT_BYTES,
};
use runmat_execution_runner_native::{
    FilesystemObjectStore, NativeExecutionConfig, NativeProgramSession,
};

use crate::cli::Cli;

use super::callables::{
    compile_numerical_generators, compile_numerical_generators_in_frozen,
    compile_signal_generators, compile_signal_generators_in_frozen, CompiledNumericalGenerator,
    CompiledSignalGenerator, NumericalGeneratorTarget,
};

const POLL_INTERVAL: Duration = Duration::from_millis(10);
const INPUT_TASK_MEMORY_BYTES: u64 = 512 * 1024 * 1024;
const INPUT_TASK_SCRATCH_BYTES: u64 = 128 * 1024 * 1024;

pub(super) async fn materialize_study(
    document_path: &std::path::Path,
    study: &mut runmat_runtime::analysis::AnalysisStudySpec,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    let immutable_values = study.immutable_values.clone();
    let Some(model) = study.model.as_mut() else {
        return Ok(());
    };
    let wall_millis = materialization_wall_millis(model);
    model.signals = materialize_signals(
        document_path,
        &model.signals,
        &immutable_values,
        wall_millis,
        cli,
        config,
    )
    .await?;
    materialize_numerical_sources(
        document_path,
        &mut model.response_spectra,
        &mut model.random_excitations,
        &immutable_values,
        wall_millis,
        cli,
        config,
    )
    .await?;
    Ok(())
}

pub(super) async fn materialize_sweep(
    document_path: &std::path::Path,
    sweep: &mut runmat_runtime::analysis::AnalysisStudySweepSpec,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    let mut signals = Vec::new();
    let mut immutable_values = Vec::new();
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
        immutable_values.extend(study.immutable_values.iter().cloned());
    }
    let mut materialized = materialize_signals(
        document_path,
        &signals,
        &immutable_values,
        wall_millis,
        cli,
        config,
    )
    .await?
    .into_iter();
    for (study, signal_count) in sweep.studies.iter_mut().zip(study_signal_counts) {
        if signal_count == 0 {
            continue;
        }
        let model = study
            .model
            .as_mut()
            .context("sweep study model disappeared during input materialization")?;
        model.signals = materialized.by_ref().take(signal_count).collect();
        if model.signals.len() != signal_count {
            bail!("input materialization returned an incomplete sweep inventory");
        }
    }
    if materialized.next().is_some() {
        bail!("input materialization returned an oversized sweep inventory");
    }
    for study in &mut sweep.studies {
        let immutable_values = study.immutable_values.clone();
        let Some(model) = study.model.as_mut() else {
            continue;
        };
        materialize_numerical_sources(
            document_path,
            &mut model.response_spectra,
            &mut model.random_excitations,
            &immutable_values,
            wall_millis,
            cli,
            config,
        )
        .await?;
    }
    Ok(())
}

struct PreparedNumericalSource {
    target: NumericalGeneratorTarget,
    prepared: PreparedNumericalMaterialization,
}

async fn materialize_numerical_sources(
    document_path: &std::path::Path,
    response_spectra: &mut Vec<ResponseSpectrum>,
    random_spectra: &mut Vec<RandomExcitationSpectrum>,
    immutable_values: &[runmat_runtime::analysis::AnalysisImmutableValueSpec],
    maximum_wall_millis: u64,
    cli: &Cli,
    config: &RunMatRuntimeConfig,
) -> Result<()> {
    let compiled = compile_numerical_generators(
        document_path,
        response_spectra,
        random_spectra,
        cli,
        config,
        &BTreeSet::new(),
    )
    .await?;
    if compiled.is_empty() {
        return Ok(());
    }
    let mut native_config = NativeExecutionConfig::for_current_executable()?;
    native_config
        .worker_capabilities
        .insert(Capability::ProcessIsolation);
    let maximum_parallel_tasks = native_config.max_workers as usize;
    let session = NativeProgramSession::new(native_config)?;
    let prepared = prepare_compiled_numerical_materializations(
        document_path,
        response_spectra,
        random_spectra,
        compiled,
        immutable_values,
        session.artifact_access(),
    )?;
    for source in &prepared {
        for object in &source.prepared.input_objects {
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
    for source in &prepared {
        for chunk in &source.prepared.program.chunks {
            chunk_identities.push(chunk.plan.identity);
            submissions.push((
                chunk.request.clone(),
                build_generator_task_submission(
                    &source.prepared.admitted,
                    chunk,
                    BTreeSet::new(),
                    GeneratorExecutionContext {
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
    let mut store = FilesystemObjectStore::open(input_store_root()?, limits.max_object_bytes)?;
    let mut cursor = 0_usize;
    for source in prepared {
        let end = cursor
            .checked_add(source.prepared.program.chunks.len())
            .context("numerical generator result inventory overflows")?;
        let source_outputs = outputs
            .get(cursor..end)
            .context("numerical generator results are incomplete")?;
        let chunks = source
            .prepared
            .program
            .chunks
            .iter()
            .map(|chunk| chunk.plan.clone())
            .collect::<Vec<_>>();
        let (kind, digest, bytes) = match source.prepared.authored {
            NumericalGeneratorSource::ResponseSpectrum(authored) => {
                let materialized = materialize_response_spectrum(
                    &authored,
                    &source.prepared.admitted,
                    &source.prepared.input_admissions,
                    &chunks,
                    source_outputs,
                )?;
                let digest = materialized
                    .ordinates
                    .numerical_digest()
                    .map_err(anyhow::Error::msg)?
                    .parse::<Digest>()?;
                let bytes = canonical_materialized_response_spectrum_bytes(&materialized)?;
                let NumericalGeneratorTarget::ResponseSpectrum(index) = source.target else {
                    bail!("response-spectrum materialization target changed kind");
                };
                response_spectra[index] = materialized;
                (AnalysisNumericalObjectKind::ResponseSpectrum, digest, bytes)
            }
            NumericalGeneratorSource::RandomSpectrum(authored) => {
                let materialized = materialize_random_spectrum(
                    &authored,
                    &source.prepared.admitted,
                    &source.prepared.input_admissions,
                    &chunks,
                    source_outputs,
                )?;
                let digest = materialized
                    .spectral_density
                    .numerical_digest()
                    .map_err(anyhow::Error::msg)?
                    .parse::<Digest>()?;
                let bytes = canonical_materialized_random_spectrum_bytes(&materialized)?;
                let NumericalGeneratorTarget::RandomSpectrum(index) = source.target else {
                    bail!("random-spectrum materialization target changed kind");
                };
                random_spectra[index] = materialized;
                (AnalysisNumericalObjectKind::RandomSpectrum, digest, bytes)
            }
        };
        let objects = prepare_analysis_objects(
            kind,
            digest,
            &bytes,
            DEFAULT_ANALYSIS_CHUNK_BYTES,
            ObjectNamespace::ResultValue,
            limits,
        )?;
        for object in &objects.objects {
            store.write_verified(object)?;
        }
        cursor = end;
    }
    if cursor != outputs.len() {
        bail!("numerical generator results exceed their prepared inventory");
    }
    Ok(())
}

async fn materialize_signals(
    document_path: &std::path::Path,
    signals: &[PhysicalSignal],
    immutable_values: &[runmat_runtime::analysis::AnalysisImmutableValueSpec],
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
    execute_compiled_materializations(
        document_path,
        signals,
        &generator_signals,
        &compiled,
        immutable_values,
        maximum_wall_millis,
    )
}

fn execute_compiled_materializations(
    document_path: &std::path::Path,
    signals: &[PhysicalSignal],
    generator_signals: &[PhysicalSignal],
    compiled: &[CompiledSignalGenerator],
    immutable_values: &[runmat_runtime::analysis::AnalysisImmutableValueSpec],
    maximum_wall_millis: u64,
) -> Result<Vec<PhysicalSignal>> {
    let mut native_config = NativeExecutionConfig::for_current_executable()?;
    native_config
        .worker_capabilities
        .insert(Capability::ProcessIsolation);
    let maximum_parallel_tasks = native_config.max_workers as usize;
    let session = NativeProgramSession::new(native_config)?;
    let prepared = prepare_compiled_materializations(
        document_path,
        generator_signals,
        compiled,
        immutable_values,
        session.artifact_access(),
    )?;
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

pub(crate) fn materialize_runtime_signal(
    request: runmat_runtime::context::RuntimeSignalMaterializationRequest,
) -> Result<PhysicalSignal> {
    let program = request.program;
    program.executable.validate()?;
    let identity = &program.executable.manifest.identity;
    if identity.root_package != program.owner_identity
        || identity.entrypoint != program.display_name
        || identity.entrypoint_function != program.function.function
        || identity.source_digest != program.executable_source_digest
    {
        bail!("runtime signal callable differs from its frozen executable product");
    }
    let admission = admit_callable_signal_generator(
        &request.signal,
        identity.program.clone(),
        BundleCallable {
            owner_identity: program.owner_identity,
            qualified_name: program.display_name,
            source_digest: program.source_digest,
        },
        program.executable_source_digest,
        &program.function,
        request.callable,
        &BTreeSet::new(),
    )?;
    let compiled = CompiledSignalGenerator {
        signal_id: request.signal.signal_id().clone(),
        admission,
        executable: program.executable,
    };
    let materialized = execute_compiled_materializations(
        std::path::Path::new("."),
        std::slice::from_ref(&request.signal),
        std::slice::from_ref(&request.signal),
        std::slice::from_ref(&compiled),
        &[],
        request.maximum_wall_millis,
    )?;
    materialized
        .into_iter()
        .next()
        .context("runtime signal materialization returned no signal")
}

pub(super) async fn prepare_signal_materializations_in_frozen(
    document_path: &std::path::Path,
    signals: &[PhysicalSignal],
    immutable_values: &[runmat_runtime::analysis::AnalysisImmutableValueSpec],
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
    prepare_compiled_materializations(
        document_path,
        &generator_signals,
        &compiled,
        immutable_values,
        &preparation_access(frozen),
    )
}

pub(super) async fn prepare_numerical_materializations_in_frozen(
    document_path: &std::path::Path,
    response_spectra: &[ResponseSpectrum],
    random_spectra: &[RandomExcitationSpectrum],
    immutable_values: &[runmat_runtime::analysis::AnalysisImmutableValueSpec],
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
) -> Result<Vec<PreparedNumericalMaterialization>> {
    let compiled = compile_numerical_generators_in_frozen(
        document_path,
        response_spectra,
        random_spectra,
        frozen,
        config,
        &BTreeSet::new(),
    )
    .await?;
    Ok(prepare_compiled_numerical_materializations(
        document_path,
        response_spectra,
        random_spectra,
        compiled,
        immutable_values,
        &preparation_access(frozen),
    )?
    .into_iter()
    .map(|source| source.prepared)
    .collect())
}

fn prepare_compiled_numerical_materializations(
    document_path: &std::path::Path,
    response_spectra: &[ResponseSpectrum],
    random_spectra: &[RandomExcitationSpectrum],
    compiled: Vec<CompiledNumericalGenerator>,
    immutable_values: &[runmat_runtime::analysis::AnalysisImmutableValueSpec],
    artifact_access: &ArtifactAccess,
) -> Result<Vec<PreparedNumericalSource>> {
    let immutable_values = load_immutable_values(document_path, immutable_values, artifact_access)?;
    let inline_limit = usize::try_from(ValueLimits::default().max_inline_bytes)
        .context("inline result bound exceeds this host")?;
    compiled
        .into_iter()
        .map(|compiled| {
            compiled.validate()?;
            let authored = match compiled.target {
                NumericalGeneratorTarget::ResponseSpectrum(index) => {
                    NumericalGeneratorSource::ResponseSpectrum(
                        response_spectra
                            .get(index)
                            .context("response-spectrum generator index is out of range")?
                            .clone(),
                    )
                }
                NumericalGeneratorTarget::RandomSpectrum(index) => {
                    NumericalGeneratorSource::RandomSpectrum(
                        random_spectra
                            .get(index)
                            .context("random-spectrum generator index is out of range")?
                            .clone(),
                    )
                }
            };
            let mut selected_objects = BTreeMap::new();
            let mut resolve = |artifact: &runmat_analysis_core::SignalArtifactRef| {
                if artifact.schema != EXTERNALIZED_VALUE_SCHEMA {
                    return Err(runmat_analysis_execution::AnalysisBridgeError::Invalid(
                        "generator artifact uses an unsupported immutable value schema".into(),
                    ));
                }
                let digest = artifact
                    .content_digest
                    .parse::<Digest>()
                    .map_err(runmat_analysis_execution::AnalysisBridgeError::Platform)?;
                let (object, admission) = immutable_values
                    .get(&(artifact.artifact_id.clone(), digest))
                    .ok_or_else(|| {
                        runmat_analysis_execution::AnalysisBridgeError::Invalid(format!(
                            "generator artifact `{}` has no resolved immutable value",
                            artifact.artifact_id
                        ))
                    })?;
                selected_objects.insert(digest, object.clone());
                Ok(runmat_analysis_execution::ResolvedSignalArtifact {
                    value: ValuePayload::Object(Box::new(object.reference.clone())),
                    admission: admission.clone(),
                })
            };
            let plan = match &authored {
                NumericalGeneratorSource::ResponseSpectrum(spectrum) => {
                    plan_response_spectrum_generator_chunks(
                        spectrum,
                        &compiled.admission,
                        inline_limit,
                        &mut resolve,
                    )?
                }
                NumericalGeneratorSource::RandomSpectrum(spectrum) => {
                    plan_random_spectrum_generator_chunks(
                        spectrum,
                        &compiled.admission,
                        inline_limit,
                        &mut resolve,
                    )?
                }
            };
            let prepared = prepare_numerical_materialization(
                authored,
                compiled.admission.clone(),
                compiled.executable.clone(),
                plan.chunks,
                selected_objects.into_values().collect(),
                plan.input_admissions,
            )?;
            Ok(PreparedNumericalSource {
                target: compiled.target,
                prepared,
            })
        })
        .collect()
}

fn prepare_compiled_materializations(
    document_path: &std::path::Path,
    generator_signals: &[PhysicalSignal],
    compiled: &[CompiledSignalGenerator],
    immutable_values: &[runmat_runtime::analysis::AnalysisImmutableValueSpec],
    artifact_access: &ArtifactAccess,
) -> Result<Vec<PreparedSignalMaterialization>> {
    if compiled.len() != generator_signals.len() {
        bail!("compiled signal generator inventory is incomplete");
    }
    let maximum_inline_result_bytes = usize::try_from(ValueLimits::default().max_inline_bytes)
        .context("inline result bound exceeds this host")?;
    let immutable_values = load_immutable_values(document_path, immutable_values, artifact_access)?;
    generator_signals
        .iter()
        .cloned()
        .zip(compiled)
        .map(|(authored, compiled)| {
            compiled.validate()?;
            let mut selected_objects = BTreeMap::new();
            let plan = plan_signal_generator_chunks(
                &authored,
                &compiled.admission,
                None,
                maximum_inline_result_bytes,
                |artifact| {
                    if artifact.schema != EXTERNALIZED_VALUE_SCHEMA {
                        return Err(runmat_analysis_execution::AnalysisBridgeError::Invalid(
                            "generator artifact uses an unsupported immutable value schema".into(),
                        ));
                    }
                    let digest = artifact
                        .content_digest
                        .parse::<Digest>()
                        .map_err(runmat_analysis_execution::AnalysisBridgeError::Platform)?;
                    let (object, admission) = immutable_values
                        .get(&(artifact.artifact_id.clone(), digest))
                        .ok_or_else(|| {
                            runmat_analysis_execution::AnalysisBridgeError::Invalid(format!(
                                "generator artifact `{}` has no resolved immutable value",
                                artifact.artifact_id
                            ))
                        })?;
                    selected_objects.insert(digest, object.clone());
                    Ok(runmat_analysis_execution::ResolvedSignalArtifact {
                        value: ValuePayload::Object(Box::new(object.reference.clone())),
                        admission: admission.clone(),
                    })
                },
            )?;
            Ok(prepare_signal_materialization(
                authored,
                compiled.admission.clone(),
                compiled.executable.clone(),
                plan.chunks,
                selected_objects.into_values().collect(),
                plan.input_admissions,
            )?)
        })
        .collect()
}

type ImmutableValueInventory = BTreeMap<
    (String, Digest),
    (
        ProgramExecutionObject,
        runmat_execution::value::ValueAdmissionReceipt,
    ),
>;

fn load_immutable_values(
    document_path: &std::path::Path,
    declarations: &[runmat_runtime::analysis::AnalysisImmutableValueSpec],
    artifact_access: &ArtifactAccess,
) -> Result<ImmutableValueInventory> {
    if declarations.is_empty() {
        return Ok(BTreeMap::new());
    }
    let document_root = document_path
        .parent()
        .context("FEA document has no containing directory")?
        .canonicalize()
        .context("resolve FEA document directory")?;
    let mut inventory = BTreeMap::new();
    for declaration in declarations {
        let declared_digest = declaration.validate().map_err(anyhow::Error::msg)?;
        let relative = std::path::Path::new(&declaration.relative_path);
        let path = document_root
            .join(relative)
            .canonicalize()
            .with_context(|| format!("resolve immutable value `{}`", declaration.artifact_id))?;
        if !path.starts_with(&document_root) {
            bail!(
                "immutable value `{}` resolves outside the FEA document directory",
                declaration.artifact_id
            );
        }
        let metadata = std::fs::metadata(&path)
            .with_context(|| format!("inspect immutable value `{}`", declaration.artifact_id))?;
        if !metadata.is_file()
            || metadata.len() == 0
            || metadata.len() > MAX_PROGRAM_EXECUTION_OBJECT_BYTES
        {
            bail!(
                "immutable value `{}` is empty, not a file, or exceeds the portable object bound",
                declaration.artifact_id
            );
        }
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read immutable value `{}`", declaration.artifact_id))?;
        let payload: ValuePayload = serde_json::from_slice(&bytes)
            .with_context(|| format!("decode immutable value `{}`", declaration.artifact_id))?;
        if serde_json::to_vec(&payload)? != bytes {
            bail!(
                "immutable value `{}` is not canonical compact JSON",
                declaration.artifact_id
            );
        }
        let admitted = runmat_runtime::analysis::value_admission::admit_portable_real64(
            payload,
            runmat_runtime::analysis::value_admission::AnalysisValueAdmissionPolicy::exact_host(),
        )?;
        let (payload, receipt) = admitted.into_value_payload()?;
        receipt.validate()?;
        let object = ProgramExecutionObject::externalized_value(
            artifact_access,
            ValueRefKind::DriverObject,
            payload,
        )?;
        if object.reference.logical_digest != declared_digest {
            bail!(
                "immutable value `{}` differs from its declared logical digest",
                declaration.artifact_id
            );
        }
        let key = (declaration.artifact_id.clone(), declared_digest);
        let candidate = (object, receipt);
        match inventory.get(&key) {
            Some(existing) if existing == &candidate => {}
            Some(_) => {
                bail!(
                    "immutable value `{}` resolves inconsistently across the sweep",
                    declaration.artifact_id
                );
            }
            None => {
                inventory.insert(key, candidate);
            }
        }
    }
    Ok(inventory)
}

fn preparation_access(frozen: &runmat_package::FrozenProject) -> ArtifactAccess {
    let revision = frozen.revision();
    let authorization_scope = format!("input-preparation-{}", revision.graph_digest);
    ArtifactAccess {
        encryption_context: Digest::sha256(
            [
                b"runmat-input-preparation/v1\0".as_slice(),
                revision.graph_digest.bytes(),
                revision.source_revision.bytes(),
            ]
            .concat(),
        ),
        authorization_scope,
    }
}

fn wait_for_outputs(
    session: &NativeProgramSession,
    tasks: Vec<runmat_execution_runner_native::NativeProgramTask>,
    chunk_identities: &[runmat_execution::Digest],
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
                if !success.result_objects.is_empty() || success.outputs.len() != 1 {
                    session.cancel(CancellationReason::ParentScope);
                    bail!("bounded input task returned a non-inline or non-scalar output list");
                }
                outputs[index] = Some(success.outputs[0].clone());
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

#[cfg(test)]
mod tests {
    use runmat_execution::value::{DenseValue, ElementType, InlineValue};

    use super::*;

    #[test]
    fn immutable_value_loader_is_path_bounded_canonical_and_content_bound() {
        let directory = tempfile::tempdir().unwrap();
        let document = directory.path().join("study.fea");
        std::fs::write(&document, b"study").unwrap();
        let payload = ValuePayload::Inline(Box::new(InlineValue::Dense(DenseValue {
            element_type: ElementType::F64,
            shape: vec![2, 1],
            little_endian_data: [4.0_f64, 8.0]
                .into_iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
        })));
        let digest = payload.logical_digest().unwrap();
        std::fs::write(
            directory.path().join("magnitude.value.json"),
            serde_json::to_vec(&payload).unwrap(),
        )
        .unwrap();
        let declaration = runmat_runtime::analysis::AnalysisImmutableValueSpec {
            artifact_id: "magnitude".into(),
            relative_path: "magnitude.value.json".into(),
            logical_digest: digest.to_string(),
        };
        let access = ArtifactAccess {
            authorization_scope: "loader-test".into(),
            encryption_context: Digest::sha256(b"loader-test-key"),
        };

        let loaded =
            load_immutable_values(&document, std::slice::from_ref(&declaration), &access).unwrap();
        let (object, receipt) = loaded
            .get(&(declaration.artifact_id.clone(), digest))
            .unwrap();
        assert_eq!(object.decode_externalized_value().unwrap(), Some(payload));
        receipt.validate().unwrap();

        let mut outside = declaration.clone();
        outside.relative_path = "../magnitude.value.json".into();
        assert!(load_immutable_values(&document, &[outside], &access).is_err());

        let mut substituted = declaration;
        substituted.logical_digest = Digest::sha256(b"substituted").to_string();
        assert!(load_immutable_values(&document, &[substituted], &access).is_err());
    }
}
