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
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactAccess, ArtifactResult, BundleCallable, LogicalObject, ProgramExecutionResponse,
};
use runmat_geometry_core::{
    GeometryAuthoringValue, GeometryDefinition, GeometryDefinitionRef, GeometryDefinitionSource,
};

const POLL_INTERVAL: Duration = Duration::from_millis(10);

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
            let bytes = runmat_filesystem::read_async(&path)
                .await
                .map_err(|error| {
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
    let frozen = source.frozen_project().ok_or_else(|| {
        "generated geometry requires an enclosing frozen RunMat project".to_string()
    })?;
    let module = runmat_analysis_language::LogicalSourcePath::new(generated.module_path.clone())
        .map_err(|error| error.to_string())?;
    let symbol = runmat_analysis_language::SourceSymbol::new(generated.function.clone())
        .map_err(|error| error.to_string())?;
    let expected_path =
        runmat_runtime::analysis::physical_source_path(&source.source_root, &module)
            .map_err(|error| error.to_string())?;
    let definition_record = runmat_package::resolve_visible_function_definition(
        frozen,
        source.physical_path(),
        &expected_path,
        symbol.as_str(),
    )?;
    let source_id = definition_record.source_id.as_ref().ok_or_else(|| {
        format!(
            "FEA function `{}` has no frozen source identity",
            definition_record.qualified_name
        )
    })?;
    let source_bytes = runmat_filesystem::read_async(&definition_record.source_path)
        .await
        .map_err(|error| {
            format!(
                "cannot read frozen FEA function source {}: {error}",
                definition_record.source_path.display()
            )
        })?;
    let source_digest = Digest::sha256(&source_bytes);
    if source_digest.bytes() != source_id.content_digest.bytes() {
        return Err(format!(
            "FEA function `{}` changed after its project graph was frozen",
            definition_record.qualified_name
        ));
    }
    let source_text = String::from_utf8(source_bytes).map_err(|_| {
        format!(
            "FEA function source {} is not UTF-8 MATLAB text",
            definition_record.source_path.display()
        )
    })?;
    let revision = ProgramRevision::new(
        Digest::from_bytes(*frozen.graph_digest().bytes()),
        Digest::from_bytes(*frozen.source_revision().bytes()),
        source.program_environment().clone(),
    )
    .map_err(|error| error.to_string())?;
    let mut session = runmat_core::RunMatSession::with_options(false, false)
        .map_err(|error| error.to_string())?;
    session.set_compat_mode(source.compatibility_mode());
    session
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(frozen.clone()))
        .map_err(|error| error.to_string())?;
    let entrypoint = definition_record
        .qualified_name
        .rsplit('.')
        .next()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "FEA function has an empty function name".to_string())?;
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::frozen(
                source_id.package_instance.to_string(),
                source_id.relative_path.as_str(),
                definition_record.source_path.to_string_lossy(),
                source_text,
            ),
            Some(revision.clone()),
        )
        .await
        .map_err(|error| format!("cannot compile FEA function: {error}"))?;
    let executable = unit
        .portable_envelope_for(Some(entrypoint))
        .map_err(|error| error.to_string())?;
    let function = unit
        .function_analysis(executable.manifest.identity.entrypoint_function)
        .ok_or_else(|| "FEA function has no static function facts".to_string())?;
    let request =
        runmat_analysis_execution::resolve_geometry_generator_request(definition, bindings)
            .map_err(|error| error.to_string())?;
    let admitted = runmat_analysis_execution::admit_named_geometry_generator(
        request,
        revision,
        BundleCallable {
            owner_identity: source_id.package_instance.to_string(),
            qualified_name: definition_record.qualified_name,
            source_digest,
        },
        executable.manifest.identity.source_digest,
        function,
        &BTreeSet::new(),
    )
    .map_err(|error| error.to_string())?;
    runmat_analysis_execution::prepare_geometry_generator(admitted, executable)
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
            resources,
            retry: RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
            deadline_unix_millis: None,
            priority: 0,
        },
    )
    .map_err(|error| error.to_string())?;
    let task = session
        .submit(chunk.request.clone(), submission)
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    loop {
        if let Some(result) = task.try_result() {
            let success = result?;
            if !success.result_objects.is_empty() {
                return Err("geometry generator returned an external object closure".into());
            }
            let [value] = success.outputs.as_slice() else {
                return Err("geometry generator returned an invalid output count".into());
            };
            return Ok(ProgramExecutionResponse::Success {
                value: value.clone(),
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
                .objects
                .get(&reference.logical_digest)
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
    let relative = logical_path
        .strip_prefix("project/")
        .ok_or_else(|| "geometry source is outside the checked project root".to_string())?;
    if relative.is_empty()
        || relative
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err("geometry source path is not normalized".into());
    }
    Ok(relative
        .split('/')
        .fold(source_root.to_path_buf(), |path, part| path.join(part)))
}

#[derive(Default)]
struct CapturedObjectStore {
    objects: BTreeMap<Digest, LogicalObject>,
}

impl CacheImport for CapturedObjectStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.objects.get(&digest).map(|object| object.bytes.clone()))
    }
}

impl CacheExport for CapturedObjectStore {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        object.validate()?;
        match self.objects.get(&object.descriptor.digest) {
            Some(existing) if existing != object => {
                Err(runmat_execution_artifact::ArtifactError::Identity(
                    "geometry object digest was reused for different content".into(),
                ))
            }
            Some(_) => Ok(()),
            None => {
                self.objects
                    .insert(object.descriptor.digest, object.clone());
                Ok(())
            }
        }
    }
}
