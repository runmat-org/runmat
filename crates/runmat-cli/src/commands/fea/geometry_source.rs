use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use runmat_analysis_execution::{
    build_generator_task_submission, import_geometry_materialization_output,
    prepare_generated_geometry_product, prepare_geometry_materialization_program,
    prepare_geometry_materialization_request, GeneratorExecutionContext,
    GeometryMaterializationPreparation, PreparedGeometrySource,
};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{ValueLimits, ValuePayload};
use runmat_execution::{CancellationReason, Digest, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactAccess, ArtifactResult, LogicalObject, ProgramExecutionResponse,
};
use runmat_execution_runner_native::{
    execute_geometry_materialization_program_request, NativeExecutionConfig,
    NativeGeometryMaterializationLimits, NativeProgramSession,
};
use runmat_geometry_core::{
    GeometryAuthoringValue, GeometryDefinition, GeometryDefinitionRef, GeometryDefinitionSource,
};

use super::source::NativeCheckedFeaSource;

const POLL_INTERVAL: Duration = Duration::from_millis(10);

pub(super) async fn prepare_geometry_source(
    source: &NativeCheckedFeaSource,
    definition: &GeometryDefinition,
    dependencies: &[GeometryDefinition],
    bindings: &BTreeMap<String, GeometryAuthoringValue>,
    revision: &ProgramRevision,
    access: &ArtifactAccess,
    config: &RunMatRuntimeConfig,
) -> Result<PreparedGeometrySource> {
    match &definition.source {
        GeometryDefinitionSource::Constructed(_) => Ok(PreparedGeometrySource::constructed()),
        GeometryDefinitionSource::Imported(imported) => {
            let path = physical_source_path(&source.source_root, &imported.logical_path)?;
            let bytes = std::fs::read(&path)
                .with_context(|| format!("read imported geometry {}", path.display()))?;
            if Digest::sha256(&bytes).bytes() != imported.content_digest.bytes() {
                bail!(
                    "imported geometry {} changed after typed source checking",
                    path.display()
                );
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("imported geometry file name is not valid UTF-8")?;
            PreparedGeometrySource::imported(name, &bytes, 4 * 1024 * 1024, access)
                .map_err(anyhow::Error::new)
        }
        GeometryDefinitionSource::Generated(_) => {
            prepare_generated_source(source, definition, bindings, access, config).await
        }
        GeometryDefinitionSource::Derived(derived) => {
            let dependency = dependencies
                .iter()
                .find(|candidate| {
                    candidate.module_path == derived.source.module_path
                        && candidate.geometry_id == derived.source.geometry_id
                })
                .with_context(|| {
                    format!(
                        "derived geometry source `{}::{}` is absent from the resolved closure",
                        derived.source.module_path, derived.source.geometry_id
                    )
                })?;
            let nested = dependencies
                .iter()
                .take_while(|candidate| *candidate != dependency)
                .cloned()
                .collect::<Vec<_>>();
            let dependency_source = Box::pin(prepare_geometry_source(
                source,
                dependency,
                &nested,
                &BTreeMap::new(),
                revision,
                access,
                config,
            ))
            .await?;
            materialize_dependency(dependency, dependency_source, revision, access)
        }
    }
}

async fn prepare_generated_source(
    source: &NativeCheckedFeaSource,
    definition: &GeometryDefinition,
    bindings: &BTreeMap<String, GeometryAuthoringValue>,
    access: &ArtifactAccess,
    config: &RunMatRuntimeConfig,
) -> Result<PreparedGeometrySource> {
    let frozen = source
        .frozen_project
        .as_ref()
        .context("generated geometry requires an enclosing frozen RunMat project")?;
    let document_path = physical_source_path(&source.source_root, source.root_module.as_str())?;
    let prepared = super::callables::compile_geometry_generator(
        &document_path,
        &source.source_root,
        definition,
        bindings,
        frozen,
        config,
    )
    .await?;
    let response = execute_generator(&prepared).await?;
    let product = prepare_generated_geometry_product(&prepared.admitted, response, access)?;
    let receipt = product.receipt.clone();
    let root = product.root.clone();
    Ok(PreparedGeometrySource {
        source: runmat_analysis_execution::GeometryMaterializationSource::Generated {
            receipt: Box::new(receipt),
        },
        objects: vec![product.object],
        roots: vec![root],
    })
}

async fn execute_generator(
    prepared: &runmat_analysis_execution::PreparedGeometryGenerator,
) -> Result<ProgramExecutionResponse> {
    let [chunk] = prepared.program.chunks.as_slice() else {
        bail!("geometry generator must contain exactly one frozen invocation");
    };
    let mut native = NativeExecutionConfig::for_current_executable()?;
    native
        .worker_capabilities
        .insert(Capability::ProcessIsolation);
    let session = NativeProgramSession::new(native)?;
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
    )?;
    let task = session.submit(chunk.request.clone(), submission)?;
    let started = Instant::now();
    loop {
        if let Some(result) = task.try_result() {
            let success = result.map_err(anyhow::Error::msg)?;
            if !success.result_objects.is_empty() {
                bail!("geometry generator returned an external object closure");
            }
            let [value] = success.outputs.as_slice() else {
                bail!("geometry generator returned an invalid output count");
            };
            return Ok(ProgramExecutionResponse::Success {
                value: value.clone(),
            });
        }
        if started.elapsed() >= Duration::from_millis(5 * 60 * 1_000) {
            session.cancel(CancellationReason::Deadline);
            bail!("geometry generator exceeded its bounded execution time");
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

fn materialize_dependency(
    definition: &GeometryDefinition,
    source: PreparedGeometrySource,
    revision: &ProgramRevision,
    access: &ArtifactAccess,
) -> Result<PreparedGeometrySource> {
    let provider = runmat_execution_runner_native::native_geometry_provider_profile()
        .context("derived geometry requires a RunMat build with the OCCT feature enabled")?;
    let request = prepare_geometry_materialization_request(
        definition.clone(),
        BTreeMap::new(),
        &source,
        provider,
        GeometryMaterializationPreparation::default(),
    )?;
    let prepared = prepare_geometry_materialization_program(
        request.clone(),
        revision.clone(),
        access,
        &source.roots,
    )?;
    let mut store = CapturedObjectStore::default();
    source.write_to(&mut store)?;
    prepared.write_to(&mut store)?;
    let response = execute_geometry_materialization_program_request(
        &prepared.program,
        &mut store,
        NativeGeometryMaterializationLimits::default(),
    );
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        bail!("derived geometry prerequisite failed to materialize: {response:?}");
    };
    let [ValuePayload::Object(output_root)] = outputs.as_slice() else {
        bail!("derived geometry prerequisite returned an invalid result root");
    };
    let output = import_geometry_materialization_output(&store, output_root, access)?;
    output.validate_against(&request)?;
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
                .context("derived geometry closure contains an unavailable object")
        })
        .collect::<Result<Vec<_>>>()?;
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
) -> Result<std::path::PathBuf> {
    let relative = logical_path
        .strip_prefix("project/")
        .context("geometry source is outside the checked project root")?;
    if relative.is_empty()
        || relative
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        bail!("geometry source path is not normalized");
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
