use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use runmat_analysis_execution::{
    build_generator_task_submission, import_geometry_materialization_output,
    prepare_generated_geometry_product, prepare_geometry_materialization_program,
    prepare_geometry_materialization_request, GeneratorExecutionContext,
    GeometryMaterializationPreparation, PreparedGeometrySource,
};
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{ValueLimits, ValuePayload};
use runmat_execution::{CancellationReason, Digest, ProgramRevision};

use runmat_execution_artifact::{ArtifactAccess, ProgramExecutionResponse};
use runmat_geometry_core::{
    GeometryAuthoringValue, GeometryDefinition, GeometryDefinitionRef, GeometryDefinitionSource,
};

const POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedFeaGeometry {
    pub output: runmat_analysis_execution::GeometryMaterializationOutput,
    pub topology: runmat_geometry_core::ExactBRepTopology,
    pub kernel_representation: Vec<u8>,
}

pub(super) async fn materialize_study_geometry(
    source: &runmat_core::CheckedFeaSessionSource,
    resolved: &runmat_analysis_language::ResolvedAnalysisStudy,
    inputs: &[runmat_analysis_core::AuthoredStudyInputBinding],
    authority: &str,
) -> Result<MaterializedFeaGeometry, String> {
    let realized = resolved
        .study
        .realize_with_models(&resolved.models, &resolved.geometry, inputs)
        .map_err(|error| error.to_string())?;
    let revision = super::preparation::program_revision(source)?;
    let access = super::preparation::checked_fea_artifact_access(authority);
    let geometry_source = prepare_geometry_source(
        source,
        &resolved.geometry,
        &resolved.geometry_dependencies,
        &realized.geometry.bindings,
        &revision,
        &access,
    )
    .await?;
    let provider = crate::native_geometry_provider_profile().ok_or_else(|| {
        "typed FEA geometry preview requires a RunMat build with the OCCT feature enabled"
            .to_string()
    })?;
    let request = prepare_geometry_materialization_request(
        resolved.geometry.clone(),
        realized.geometry.bindings,
        &geometry_source,
        provider,
        GeometryMaterializationPreparation::default(),
    )
    .map_err(|error| error.to_string())?;
    let prepared = prepare_geometry_materialization_program(
        request.clone(),
        revision,
        &access,
        &geometry_source.roots,
    )
    .map_err(|error| error.to_string())?;
    let mut store = CapturedObjectStore::default();
    geometry_source
        .write_to(&mut store)
        .map_err(|error| error.to_string())?;
    prepared
        .write_to(&mut store)
        .map_err(|error| error.to_string())?;
    let limits = crate::NativeGeometryMaterializationLimits::default();
    let response = crate::execute_geometry_materialization_program_request(
        &prepared.program,
        &mut store,
        limits,
    );
    let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = response else {
        return Err(format!(
            "typed FEA geometry preview failed to materialize: {response:?}"
        ));
    };
    let [ValuePayload::Object(output_root)] = outputs.as_slice() else {
        return Err("typed FEA geometry preview returned an invalid result root".into());
    };
    let output = import_geometry_materialization_output(&store, output_root, &access)
        .map_err(|error| error.to_string())?;
    output
        .validate_against(&request)
        .map_err(|error| error.to_string())?;
    let exact = runmat_meshing_execution::import_exact_geometry_input(
        &store,
        output.document.clone(),
        &output.geometry_root,
        access,
        limits.inventory,
    )
    .map_err(|error| error.to_string())?;
    let components = exact
        .into_geometry_objects()
        .into_components()
        .map_err(|error| error.to_string())?;
    let topology = components.topology;
    let mapped_faces = output
        .source_face_ids
        .values()
        .cloned()
        .collect::<BTreeSet<_>>();
    let topology_faces = topology
        .faces
        .iter()
        .map(|face| face.id.clone())
        .collect::<BTreeSet<_>>();
    if mapped_faces != topology_faces {
        return Err(
            "materialized FEA geometry face ordinals do not match its exact topology".into(),
        );
    }
    let kernel_representation = components.kernel_representation.ok_or_else(|| {
        "materialized FEA geometry has no exact kernel representation".to_string()
    })?;
    Ok(MaterializedFeaGeometry {
        output,
        topology,
        kernel_representation,
    })
}

pub(super) async fn prepare_geometry_source(
    source: &runmat_core::CheckedFeaSessionSource,
    definition: &GeometryDefinition,
    dependencies: &[GeometryDefinition],
    bindings: &BTreeMap<String, GeometryAuthoringValue>,
    revision: &ProgramRevision,
    access: &ArtifactAccess,
) -> Result<PreparedGeometrySource, String> {
    match &definition.source {
        GeometryDefinitionSource::Constructed(_) => Ok(PreparedGeometrySource::constructed()),
        GeometryDefinitionSource::Imported(imported) => {
            let path = physical_source_path(&source.source_root, &imported.logical_path)?;
            let bytes = source.read_source(&path).await.map_err(|error| {
                format!("cannot read imported geometry {}: {error}", path.display())
            })?;
            if Digest::sha256(&bytes).bytes() != imported.content_digest.bytes() {
                return Err(format!(
                    "imported geometry {} changed after typed source checking",
                    path.display()
                ));
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| "imported geometry file name is not valid UTF-8".to_string())?;
            PreparedGeometrySource::imported(name, &bytes, 4 * 1024 * 1024, access)
                .map_err(|error| error.to_string())
        }
        GeometryDefinitionSource::Generated(_) => {
            prepare_generated_source(source, definition, bindings, access).await
        }
        GeometryDefinitionSource::CrossSection(_) => Err(
            "a cross-section is meshed and solved by the transmission-line solver; no geometry kernel materializes it".into(),
        ),
        GeometryDefinitionSource::Derived(derived) => {
            let (dependency_index, dependency) = dependencies
                .iter()
                .enumerate()
                .find(|(_, candidate)| {
                    candidate.module_path == derived.source.module_path
                        && candidate.geometry_id == derived.source.geometry_id
                })
                .ok_or_else(|| {
                    format!(
                        "derived geometry source `{}::{}` is absent from the resolved closure",
                        derived.source.module_path, derived.source.geometry_id
                    )
                })?;
            let dependency_source = Box::pin(prepare_geometry_source(
                source,
                dependency,
                &dependencies[..dependency_index],
                &BTreeMap::new(),
                revision,
                access,
            ))
            .await?;
            materialize_dependency(dependency, dependency_source, revision, access)
        }
    }
}

async fn prepare_generated_source(
    source: &runmat_core::CheckedFeaSessionSource,
    definition: &GeometryDefinition,
    bindings: &BTreeMap<String, GeometryAuthoringValue>,
    access: &ArtifactAccess,
) -> Result<PreparedGeometrySource, String> {
    let prepared = compile_geometry_generator(source, definition, bindings).await?;
    let response = execute_generator(&prepared).await?;
    let product = prepare_generated_geometry_product(&prepared.admitted, response, access)
        .map_err(|error| error.to_string())?;
    Ok(PreparedGeometrySource {
        source: runmat_analysis_execution::GeometryMaterializationSource::Generated {
            receipt: Box::new(product.receipt),
        },
        objects: vec![product.object],
        roots: vec![product.root],
    })
}

async fn compile_geometry_generator(
    source: &runmat_core::CheckedFeaSessionSource,
    definition: &GeometryDefinition,
    bindings: &BTreeMap<String, GeometryAuthoringValue>,
) -> Result<runmat_analysis_execution::PreparedGeometryGenerator, String> {
    let GeometryDefinitionSource::Generated(generated) = &definition.source else {
        return Err("geometry generator compilation requires generated geometry".into());
    };
    let compiled = super::callable::compile_imported_function(
        source,
        &generated.module_path,
        &generated.function,
        None,
    )
    .await?;
    let request =
        runmat_analysis_execution::resolve_geometry_generator_request(definition, bindings)
            .map_err(|error| error.to_string())?;
    let admitted = runmat_analysis_execution::admit_named_geometry_generator(
        request,
        compiled.revision,
        compiled.callable,
        compiled.compiled_binding,
        &compiled.function,
        &BTreeSet::new(),
    )
    .map_err(|error| error.to_string())?;
    runmat_analysis_execution::prepare_geometry_generator(admitted, compiled.executable)
        .map_err(|error| error.to_string())
}

async fn execute_generator(
    prepared: &runmat_analysis_execution::PreparedGeometryGenerator,
) -> Result<ProgramExecutionResponse, String> {
    let [chunk] = prepared.program.chunks.as_slice() else {
        return Err("geometry generator must contain exactly one frozen invocation".into());
    };
    let mut native = crate::NativeExecutionConfig::for_current_executable()
        .map_err(|error| error.to_string())?;
    native
        .worker_capabilities
        .insert(Capability::ProcessIsolation);
    let session = crate::NativeProgramSession::new(native).map_err(|error| error.to_string())?;
    let resources = ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 512 * 1024 * 1024,
        scratch_bytes: 128 * 1024 * 1024,
        max_wall_millis: 5 * 60 * 1_000,
        max_artifact_bytes: ValueLimits::default().max_inline_bytes,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
    };
    let submission = build_generator_task_submission(
        &prepared.admitted,
        chunk,
        BTreeSet::new(),
        GeneratorExecutionContext {
            scope_id: session.scope_id(),
            pool_id: session.pool_id(),
            host: crate::program_host::program_host(
                chunk.direct_request().map_err(|error| error.to_string())?,
                crate::program_host::local_trust(),
            )
            .map_err(|error| error.to_string())?,
            resources,
            retry: RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
            deadline_unix_millis: None,
            priority: 0,
        },
    )
    .map_err(|error| error.to_string())?;
    let task = session
        .submit(
            chunk
                .direct_request()
                .map_err(|error| error.to_string())?
                .clone(),
            submission,
        )
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    loop {
        if let Some(result) = task.try_result() {
            let success = result.map_err(|failure| failure.to_string())?;
            if !success.result_objects().is_empty() {
                return Err("geometry generator returned an external object closure".into());
            }
            let Some(([value], _)) = success.values() else {
                return Err("geometry generator returned an invalid output count".into());
            };
            return Ok(ProgramExecutionResponse::Success {
                outputs: vec![value.clone()],
            });
        }
        if started.elapsed() >= Duration::from_millis(5 * 60 * 1_000) {
            session.cancel(CancellationReason::Deadline);
            return Err("geometry generator exceeded its bounded execution time".into());
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

fn materialize_dependency(
    definition: &GeometryDefinition,
    source: PreparedGeometrySource,
    revision: &ProgramRevision,
    access: &ArtifactAccess,
) -> Result<PreparedGeometrySource, String> {
    let provider = crate::native_geometry_provider_profile()
        .ok_or_else(|| "derived geometry requires OCCT support".to_string())?;
    let request = prepare_geometry_materialization_request(
        definition.clone(),
        BTreeMap::new(),
        &source,
        provider,
        GeometryMaterializationPreparation::default(),
    )
    .map_err(|error| error.to_string())?;
    let prepared = prepare_geometry_materialization_program(
        request.clone(),
        revision.clone(),
        access,
        &source.roots,
    )
    .map_err(|error| error.to_string())?;
    let mut store = CapturedObjectStore::default();
    source
        .write_to(&mut store)
        .map_err(|error| error.to_string())?;
    prepared
        .write_to(&mut store)
        .map_err(|error| error.to_string())?;
    let response = crate::execute_geometry_materialization_program_request(
        &prepared.program,
        &mut store,
        crate::NativeGeometryMaterializationLimits::default(),
    );
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        return Err(format!(
            "derived geometry prerequisite failed to materialize: {response:?}"
        ));
    };
    let [ValuePayload::Object(output_root)] = outputs.as_slice() else {
        return Err("derived geometry prerequisite returned an invalid result root".into());
    };
    let output = import_geometry_materialization_output(&store, output_root, access)
        .map_err(|error| error.to_string())?;
    output
        .validate_against(&request)
        .map_err(|error| error.to_string())?;
    let roots = result_objects
        .into_iter()
        .filter(|reference| reference.logical_digest != output_root.logical_digest)
        .collect::<Vec<_>>();
    let objects = roots
        .iter()
        .map(|reference| {
            store
                .logical_object(reference.logical_digest)
                .cloned()
                .ok_or_else(|| {
                    "derived geometry closure contains an unavailable object".to_string()
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PreparedGeometrySource {
        source: runmat_analysis_execution::GeometryMaterializationSource::Derived {
            definition: GeometryDefinitionRef {
                module_path: definition.module_path.clone(),
                geometry_id: definition.geometry_id.clone(),
            },
            document: Box::new(output.document.clone()),
            root: output.document.primary_artifact().clone(),
        },
        objects,
        roots,
    })
}

fn physical_source_path(
    source_root: &std::path::Path,
    logical_path: &str,
) -> Result<std::path::PathBuf, String> {
    let path = runmat_runtime::analysis::LogicalSourcePath::new(logical_path.to_owned())
        .map_err(|error| error.to_string())?;
    runmat_runtime::analysis::physical_source_path(source_root, &path)
        .map_err(|error| error.to_string())
}

// The in-process and sidecar hosts share the same bounded artifact store.
use runmat_execution_artifact::cache::MemoryObjectStore as CapturedObjectStore;
