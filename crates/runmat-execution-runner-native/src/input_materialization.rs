//! Placement-neutral completion and application of generated solver inputs.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use runmat_analysis_core::{
    AnalysisModel, PhysicalSignal, RandomExcitationSpectrum, ResponseSpectrum,
};
use runmat_analysis_execution::{
    canonical_materialized_random_spectrum_bytes, canonical_materialized_response_spectrum_bytes,
    canonicalize_signal_generator_outputs, materialize_random_spectrum,
    materialize_response_spectrum, materialize_signal_for_solver, GeneratorChunk,
    GeneratorChunkOutput, GeneratorExecutionContext, NumericalGeneratorSource,
    PreparedAnalysisObjects, PreparedInputMaterialization,
};
use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, DEFAULT_ANALYSIS_CHUNK_BYTES};
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValueLimits;
use runmat_execution::CancellationReason;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace, ProgramExecutionObject};

use crate::{NativeExecutionError, NativeExecutionResult};

const LOCAL_GENERATOR_TIMEOUT: Duration = Duration::from_secs(5 * 60);

#[derive(Clone, Debug)]
pub(crate) enum CompletedInput {
    Signal(PhysicalSignal),
    ResponseSpectrum(ResponseSpectrum),
    RandomSpectrum(RandomExcitationSpectrum),
}

#[derive(Clone, Debug)]
pub(crate) struct CompletedInputMaterialization {
    pub input: CompletedInput,
    pub objects: PreparedAnalysisObjects,
}

pub(crate) async fn execute_locally(
    mut config: crate::NativeExecutionConfig,
    materializations: &[PreparedInputMaterialization],
    source: &impl CacheImport,
    artifact_access: &ArtifactAccess,
    workers: u32,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> NativeExecutionResult<Vec<CompletedInputMaterialization>> {
    if materializations.is_empty() {
        return Ok(Vec::new());
    }
    for prepared in materializations {
        validate_materialization_mesh(prepared, Some(mesh))?;
    }
    config.max_workers = workers.max(1);
    config
        .worker_capabilities
        .insert(Capability::ProcessIsolation);
    let session = crate::NativeProgramSession::new(config)?;
    for reference in materializations
        .iter()
        .flat_map(|prepared| prepared.program().input_references.iter())
    {
        let bytes = source
            .read_verified(reference.logical_digest)
            .map_err(protocol)?
            .ok_or_else(|| protocol("generator input object is unavailable"))?;
        session.register_value_object(ProgramExecutionObject {
            reference: reference.clone(),
            bytes,
        })?;
    }
    let resources = ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 512 * 1024 * 1024,
        scratch_bytes: 128 * 1024 * 1024,
        max_wall_millis: LOCAL_GENERATOR_TIMEOUT.as_millis() as u64,
        max_artifact_bytes: ValueLimits::default().max_inline_bytes,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
    };
    let mut tasks = Vec::new();
    let mut task_locations = Vec::new();
    for (materialization_index, prepared) in materializations.iter().enumerate() {
        prepared.validate().map_err(protocol)?;
        for (chunk_index, chunk) in prepared.program().chunks.iter().enumerate() {
            let mut chunk_resources = resources.clone();
            chunk_resources
                .required_capabilities
                .extend(chunk.graph_capability().map_err(protocol)?);
            let context = GeneratorExecutionContext {
                scope_id: session.scope_id(),
                pool_id: session.pool_id(),
                host: chunk
                    .execution_host_requirement(crate::program_host::local_trust())
                    .map_err(protocol)?,
                resources: chunk_resources,
                retry: RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
                deadline_unix_millis: None,
                priority: 0,
            };
            let submission = match prepared {
                PreparedInputMaterialization::Signal(value) => {
                    runmat_analysis_execution::build_generator_task_submission(
                        &value.admitted,
                        chunk,
                        BTreeSet::new(),
                        context,
                    )
                }
                PreparedInputMaterialization::Numerical(value) => {
                    runmat_analysis_execution::build_generator_task_submission(
                        &value.admitted,
                        chunk,
                        BTreeSet::new(),
                        context,
                    )
                }
            }
            .map_err(protocol)?;
            tasks.push((chunk.request.clone(), submission));
            task_locations.push((materialization_index, chunk_index, chunk.plan.identity));
        }
    }
    let task_handles = session.submit_workers(tasks)?;
    if task_handles.len() != task_locations.len() {
        return Err(protocol(
            "generated solver input task inventory differs from its chunk plans",
        ));
    }
    let mut outputs = materializations
        .iter()
        .map(|prepared| vec![None; prepared.program().chunks.len()])
        .collect::<Vec<_>>();
    let started = Instant::now();
    let mut remaining = task_handles.len();
    while remaining > 0 {
        for (task, (materialization_index, chunk_index, chunk_identity)) in
            task_handles.iter().zip(&task_locations)
        {
            if outputs[*materialization_index][*chunk_index].is_some() {
                continue;
            }
            let Some(result) = task.try_result() else {
                continue;
            };
            let success = match result {
                Ok(success) => success,
                Err(error) => {
                    session.cancel(CancellationReason::ParentScope);
                    return Err(protocol(error));
                }
            };
            let output = runmat_analysis_execution::generator_chunk_output_from_success(
                *chunk_identity,
                success,
            )
            .map_err(|error| {
                session.cancel(CancellationReason::ParentScope);
                protocol(error)
            })?;
            outputs[*materialization_index][*chunk_index] = Some(output);
            remaining -= 1;
        }
        if remaining == 0 {
            break;
        }
        if started.elapsed() >= LOCAL_GENERATOR_TIMEOUT {
            session.cancel(CancellationReason::Deadline);
            return Err(protocol(
                "generated solver input exceeded its execution limit",
            ));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let mut completed = Vec::with_capacity(materializations.len());
    for (prepared, outputs) in materializations.iter().zip(outputs) {
        let outputs = outputs
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| protocol("generated solver input omitted a chunk result"))?;
        let chunks = prepared
            .program()
            .chunks
            .iter()
            .map(|chunk| chunk.plan.clone())
            .collect::<Vec<_>>();
        let (input, kind, identity, bytes) =
            materialize_input(prepared, &chunks, &outputs, Some(mesh))?;
        let objects = runmat_analysis_execution::prepare_analysis_objects(
            kind,
            identity,
            &bytes,
            DEFAULT_ANALYSIS_CHUNK_BYTES,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
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
            object.validate().map_err(protocol)?;
            runmat_execution::value::ValuePayload::Object(Box::new(reference))
                .validate(runmat_execution::value::ValueLimits::default())
                .map_err(protocol)?;
        }
        completed.push(CompletedInputMaterialization { input, objects });
    }
    Ok(completed)
}

pub(crate) fn validate_materialization_mesh(
    prepared: &PreparedInputMaterialization,
    mesh: Option<&runmat_meshing_core::SolverMeshArtifact>,
) -> NativeExecutionResult<()> {
    prepared.validate().map_err(protocol)?;
    if let PreparedInputMaterialization::Signal(signal) = prepared {
        if signal.topology_binding.is_some() {
            let mesh = mesh.ok_or_else(|| {
                protocol("topology signal preprocessing requires the authoritative solver mesh")
            })?;
            signal.validate_for_mesh(mesh).map_err(protocol)?;
        }
    }
    Ok(())
}

pub(crate) fn materialize_input(
    prepared: &PreparedInputMaterialization,
    chunks: &[GeneratorChunk],
    outputs: &[GeneratorChunkOutput],
    mesh: Option<&runmat_meshing_core::SolverMeshArtifact>,
) -> NativeExecutionResult<(
    CompletedInput,
    AnalysisNumericalObjectKind,
    runmat_execution::Digest,
    Vec<u8>,
)> {
    validate_materialization_mesh(prepared, mesh)?;
    match prepared {
        PreparedInputMaterialization::Signal(prepared) => {
            let realization = canonicalize_signal_generator_outputs(
                &prepared.authored,
                &prepared.admitted,
                prepared
                    .topology_binding
                    .as_ref()
                    .map(|binding| binding.entity_count()),
                &prepared.input_admissions,
                chunks,
                outputs,
            )
            .map_err(protocol)?;
            prepared
                .expectation()
                .map_err(protocol)?
                .validate_realization(&realization)
                .map_err(protocol)?;
            let signal =
                materialize_signal_for_solver(&prepared.authored, &prepared.admitted, &realization)
                    .map_err(protocol)?;
            Ok((
                CompletedInput::Signal(signal),
                AnalysisNumericalObjectKind::Signal,
                realization.identity.digest(),
                realization.canonical_bytes().map_err(protocol)?,
            ))
        }
        PreparedInputMaterialization::Numerical(prepared) => match &prepared.authored {
            NumericalGeneratorSource::ResponseSpectrum(authored) => {
                let spectrum = materialize_response_spectrum(
                    authored,
                    &prepared.admitted,
                    &prepared.input_admissions,
                    chunks,
                    outputs,
                )
                .map_err(protocol)?;
                let identity = spectrum
                    .ordinates
                    .numerical_digest()
                    .map_err(protocol)?
                    .parse()
                    .map_err(protocol)?;
                let bytes =
                    canonical_materialized_response_spectrum_bytes(&spectrum).map_err(protocol)?;
                Ok((
                    CompletedInput::ResponseSpectrum(spectrum),
                    AnalysisNumericalObjectKind::ResponseSpectrum,
                    identity,
                    bytes,
                ))
            }
            NumericalGeneratorSource::RandomSpectrum(authored) => {
                let spectrum = materialize_random_spectrum(
                    authored,
                    &prepared.admitted,
                    &prepared.input_admissions,
                    chunks,
                    outputs,
                )
                .map_err(protocol)?;
                let identity = spectrum
                    .spectral_density
                    .numerical_digest()
                    .map_err(protocol)?
                    .parse()
                    .map_err(protocol)?;
                let bytes =
                    canonical_materialized_random_spectrum_bytes(&spectrum).map_err(protocol)?;
                Ok((
                    CompletedInput::RandomSpectrum(spectrum),
                    AnalysisNumericalObjectKind::RandomSpectrum,
                    identity,
                    bytes,
                ))
            }
        },
    }
}

pub(crate) fn apply_to_model(
    model: &mut AnalysisModel,
    completed: impl IntoIterator<Item = CompletedInputMaterialization>,
) -> NativeExecutionResult<Vec<CompletedInputMaterialization>> {
    let completed = completed.into_iter().collect::<Vec<_>>();
    let mut signal_replacements = BTreeMap::new();
    let mut response_replacements = BTreeMap::new();
    let mut random_replacements = BTreeMap::new();
    for result in &completed {
        match &result.input {
            CompletedInput::Signal(signal) => {
                signal_replacements.insert(signal.signal_id().clone(), signal.clone());
            }
            CompletedInput::ResponseSpectrum(spectrum) => {
                response_replacements.insert(spectrum.spectrum_id.clone(), spectrum.clone());
            }
            CompletedInput::RandomSpectrum(spectrum) => {
                random_replacements.insert(spectrum.excitation_id.clone(), spectrum.clone());
            }
        }
    }
    model.signals = model
        .signals
        .iter()
        .map(|signal| {
            if signal.generator().is_none() {
                signal.clone()
            } else {
                signal_replacements
                    .get(signal.signal_id())
                    .cloned()
                    .unwrap_or_else(|| signal.clone())
            }
        })
        .collect();
    for spectrum in &mut model.response_spectra {
        if let Some(replacement) = response_replacements.get(&spectrum.spectrum_id) {
            *spectrum = replacement.clone();
        }
    }
    for spectrum in &mut model.random_excitations {
        if let Some(replacement) = random_replacements.get(&spectrum.excitation_id) {
            *spectrum = replacement.clone();
        }
    }
    if model
        .signals
        .iter()
        .any(|signal| signal.generator().is_some())
        || model
            .response_spectra
            .iter()
            .any(|spectrum| spectrum.ordinates.generator().is_some())
        || model
            .random_excitations
            .iter()
            .any(|spectrum| spectrum.spectral_density.generator().is_some())
    {
        return Err(protocol(
            "study retained an unmaterialized generated solver input",
        ));
    }
    Ok(completed)
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}
