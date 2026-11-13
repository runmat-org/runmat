//! Normal browser cluster producer: resolve project-only settings *before*
//! compiling code. Never relabel a locally compiled user-policy executable.
use runmat_config::runtime::ClusterRuntimePolicy;
use runmat_execution::{Digest, OutputContract, ProgramRevision};
use runmat_execution_artifact::{ExecutableForm, ExecutionBundleBuilder, ProgramBuildRecipe};
use wasm_bindgen::prelude::*;

pub(super) async fn project_policy(
    project: &runmat_package::FrozenProject,
) -> Result<ClusterRuntimePolicy, JsValue> {
    let bytes = runmat_filesystem::read_async(&project.manifest_path)
        .await
        .map_err(error)?;
    let source = String::from_utf8(bytes).map_err(error)?;
    let format = runmat_config::document::RunmatConfigFormat::from_path(&project.manifest_path)
        .map_err(error)?;
    ClusterRuntimePolicy::from_project_document(&source, format).map_err(error)
}

#[wasm_bindgen(js_name = prepareClusterProgram)]
pub async fn prepare_cluster_program(
    source_path: String,
    function: Option<String>,
) -> Result<JsValue, JsValue> {
    let project = runmat_package::discover_frozen_project_from_async(
        std::path::Path::new(&source_path),
        Default::default(),
    )
    .await
    .map_err(error)?
    .ok_or_else(|| {
        JsValue::from_str("cluster program preparation requires a frozen RunMat project")
    })?;
    let handoff = runmat_package::FrozenProjectHandoff::new(project);
    handoff.validate().map_err(error)?;
    let policy = project_policy(&handoff.project).await?;
    prepare_project_program(handoff, source_path, function, policy).await
}

/// Both direct-path and named selection consume this one project-only producer.
pub(super) async fn compile_project_program(
    handoff: &runmat_package::FrozenProjectHandoff,
    source_path: &str,
    function: Option<&str>,
    policy: &ClusterRuntimePolicy,
) -> Result<(runmat_execution::ExecutableUnitEnvelope, ProgramRevision), JsValue> {
    handoff.validate().map_err(error)?;
    let config = policy.runtime_config().map_err(error)?;
    let compatibility = match config.language.compat {
        runmat_config::runtime::LanguageCompatMode::RunMat => runmat_core::CompatMode::RunMat,
        runmat_config::runtime::LanguageCompatMode::Matlab => runmat_core::CompatMode::Matlab,
        runmat_config::runtime::LanguageCompatMode::Strict => runmat_core::CompatMode::Strict,
    };
    let project_revision = handoff.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_revision.graph_digest.bytes()),
        Digest::from_bytes(*project_revision.source_revision.bytes()),
        runmat_core::program_environment(compatibility),
    )
    .map_err(error)?;
    let mut session = runmat_core::RunMatSession::with_options(false, false).map_err(error)?;
    session
        .apply_cluster_runtime_config(&config)
        .map_err(error)?;
    session
        .install_project_handoff(handoff.clone())
        .map_err(error)?;
    let source = runmat_core::ExecutableSource::capture_project_primary(
        &handoff.project,
        std::path::Path::new(source_path),
    )
    .await
    .map_err(error)?;
    let unit = session
        .compile_executable_unit(source, Some(revision.clone()))
        .await
        .map_err(error)?;
    let envelope = unit.portable_envelope_for(function).map_err(error)?;
    if envelope.manifest.identity.entrypoint_function.is_none() {
        return Err(JsValue::from_str(
            "cluster programs require an executable script or function",
        ));
    }
    session.close_gracefully().await.map_err(error)?;
    Ok((envelope, revision))
}

pub(super) async fn prepare_project_program(
    handoff: runmat_package::FrozenProjectHandoff,
    source_path: String,
    function: Option<String>,
    policy: ClusterRuntimePolicy,
) -> Result<JsValue, JsValue> {
    let (envelope, revision) =
        compile_project_program(&handoff, &source_path, function.as_deref(), &policy).await?;
    crate::runtime::settings::validate_browser_interop(&envelope.manifest.interop)
        .map_err(error)?;
    let identity = &envelope.manifest.identity;
    let outputs = match identity.entrypoint_kind {
        runmat_execution::ExecutableEntrypointKind::Script => 0,
        runmat_execution::ExecutableEntrypointKind::Function => 1,
        runmat_execution::ExecutableEntrypointKind::DeclarationOnly => {
            return Err(JsValue::from_str(
                "cluster programs require an executable script or function",
            ))
        }
    };
    let entrypoint = identity
        .entrypoint_function
        .ok_or_else(|| JsValue::from_str("cluster program has no semantic entrypoint"))?;
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision.clone(),
        entrypoint: identity.entrypoint.clone(),
        outputs: OutputContract {
            requested_outputs: outputs,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("portable-executable-unit-v3"),
        interop: envelope.manifest.interop.clone(),
        foreign_policy: None,
        accelerators: runmat_execution::resource::accelerator_requirements_for_capabilities(
            &envelope.manifest.capabilities,
        )
        .map_err(error)?,
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let runtime_policy =
        runmat_core::cluster_settings::capture_project_policy(&policy).map_err(error)?;
    let reader = |_: &std::path::Path| {
        Err(runmat_execution_artifact::ArtifactError::Invalid(
            "compiled cluster bundle must not reread source".into(),
        ))
    };
    let bundle = ExecutionBundleBuilder::new(&handoff.project, revision, reader)
        .map_err(error)?
        .with_compiled_package_closure()
        .with_materialized_program(
            recipe,
            ExecutableForm::ExecutableUnitV4,
            envelope.canonical_bytes().map_err(error)?,
        )
        .with_project_runtime_policy_object(runtime_policy)
        .map_err(error)?
        .build()
        .map_err(error)?;
    let descriptor = runmat_execution_artifact::ProgramExecutionDescriptor {
        console: None,
        environment: Default::default(),
        schema_version: runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe: bundle.manifest.recipes[0].clone(),
        artifact: bundle.manifest.artifacts[0].clone(),
        callable: runmat_execution::ProgramCallable::semantic(
            runmat_execution::ProgramFunctionId(entrypoint.0),
            None,
        ),
        requested_outputs: outputs,
    };
    let descriptor = serde_json::to_vec(&descriptor).map_err(error)?;
    let mut archive = Vec::new();
    runmat_execution_artifact::archive::write_bundle(&bundle, &mut archive, Default::default())
        .map_err(error)?;
    let result = js_sys::Object::new();
    js_sys::Reflect::set(
        &result,
        &"bundleArchive".into(),
        &js_sys::Uint8Array::from(archive.as_slice()),
    )?;
    js_sys::Reflect::set(
        &result,
        &"descriptor".into(),
        &js_sys::Uint8Array::from(descriptor.as_slice()),
    )?;
    Ok(result.into())
}

fn error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

/// Attach real project-only settings before serializing/encrypting a selected
/// Graph Program. Different compiled language/foreign semantics are rejected.
#[wasm_bindgen(js_name = prepareClusterGraph)]
pub fn prepare_cluster_graph(program_json: Vec<u8>, project: JsValue) -> Result<Vec<u8>, JsValue> {
    if program_json.len() as u64 > runmat_execution::resource::MAX_GRAPH_PROGRAM_WIRE_BYTES {
        return Err(JsValue::from_str(
            "Graph Program exceeds its wire-size limit",
        ));
    }
    let project: runmat_config::runtime::HostConfigDocument =
        serde_wasm_bindgen::from_value(project).map_err(error)?;
    let policy = ClusterRuntimePolicy::from_project_document(&project.source, project.format)
        .map_err(error)?;
    let graph: runmat_execution_artifact::GraphExecutionRequest =
        serde_json::from_slice(&program_json).map_err(error)?;
    let graph =
        runmat_core::cluster_settings::prepare_graph_for_cluster(graph, &policy).map_err(error)?;
    serde_json::to_vec(&graph).map_err(error)
}
