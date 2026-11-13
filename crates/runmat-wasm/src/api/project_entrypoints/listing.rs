use super::super::session::RunMatWasm;
use super::*;
use runmat_runtime::analysis::AnalysisExecutionReadiness;

#[wasm_bindgen]
impl RunMatWasm {
    #[wasm_bindgen(js_name = listProjectEntrypoints)]
    pub async fn list_project_entrypoints_js(
        &self,
        start: String,
        context: JsValue,
    ) -> Result<JsValue, JsValue> {
        let context: Context = decode(context)?;
        let (original, paths) = runmat_filesystem::CapturedReadScope::capture(
            &[Path::new(&start), Path::new(".")],
            None,
        )
        .map_err(|error| failure("project.entrypoint.filesystem", error))?;
        let pinned = context
            .input_manifest
            .map(|inputs| {
                original
                    .pin_inputs(inputs)
                    .and_then(|provider| original.pinned(provider))
            })
            .transpose()
            .map_err(|error| failure("project.entrypoint.filesystem", error))?;
        let filesystem =
            crate::runtime::filesystem::captured::CapturedFilesystem::new(original, pinned)?;
        let publication_supported = filesystem.publication_supported().await?;
        filesystem.read.run(async {
            let project = capture(&paths[0]).await?;
            let local_config = self.config_for_named_project(project.project_document.clone(), context.user_settings_json);
            let policy = cluster_policy(&project)?;
            let mut descriptions = Vec::new();
            for declared in &project.manifest.entrypoints {
                let selected = project.resolve(&declared.name).await.map_err(|error| failure("project.entrypoint.resolve", error))?;
                let source_path = path(&selected)?;
                let (kind, local, cluster) = if is_fea(&selected) {
                    let config = local_config.as_ref().map_err(|error| failure("project.entrypoint.policy", error))?;
                    let mut session = runmat_core::RunMatSession::with_options(false, false).map_err(|error| failure("project.entrypoint.session", error))?;
                    session.apply_runtime_config(&config.runtime).map_err(|error| failure("project.entrypoint.policy", error))?;
                    session.install_project_handoff(project.handoff.clone()).map_err(|error| failure("project.entrypoint.install", error))?;
                    let checked = session.load_and_check_fea_source_path(&selected.source_file).await.map_err(|error| failure("project.entrypoint.check", error))?;
                    let choices = super::super::fea_execution::browser_fea_plan(&session, &checked, publication_supported)?;
                    let local = match self.admit_named_config(config) {
                        Err(error) => Support::unavailable("project.entrypoint.hostReinitializationRequired", error),
                        Ok(()) if choices.candidates.iter().any(|candidate| candidate.readiness == AnalysisExecutionReadiness::Ready) => Support::available(),
                        Ok(()) => Support::unavailable("project.entrypoint.browserFeaUnavailable", choices.candidates.iter().find_map(|candidate| candidate.reason.clone()).unwrap_or_else(|| "Checked study is unavailable in this browser host".into())),
                    };
                    (Kind::FeaStudy, local, Support::unavailable("project.entrypoint.browserFeaClusterUnavailable", FEA_CLUSTER_LIMITATION))
                } else {
                    let local_unit = match &local_config {
                        Ok(config) => compile_local(&project, &selected, config).await,
                        Err(error) => Err(failure("project.entrypoint.policy", error)),
                    };
                    // Independent project-only compilation; laptop compatibility
                    // and availability never stand in for the cluster producer.
                    let cluster_unit = super::super::cluster_program::compile_project_program(&project.handoff, &source_path, selected.function.as_deref(), &policy).await;
                    let kind = match (&local_unit, &cluster_unit) {
                        (Ok(unit), _) => code_kind(unit.entrypoint_identity(selected.function.as_deref()).map_err(|error| failure("project.entrypoint.compile", error))?.0)?,
                        (_, Ok((envelope, _))) => code_kind(envelope.manifest.identity.entrypoint_kind)?,
                        (Err(error), Err(_)) => return Err(error.clone()),
                    };
                    let local = match (local_unit, local_config.as_ref()) {
                        (Ok(unit), Ok(config)) => match self.admit_named_config(config).and_then(|_| crate::runtime::settings::validate_browser_interop(&unit.inferred_interop_manifest())) {
                            Ok(()) => Support::available(),
                            Err(error) => Support::unavailable("project.entrypoint.localUnavailable", error),
                        },
                        (Err(error), _) => Support::unavailable("project.entrypoint.localUnavailable", crate::wire::errors::js_value_to_string(error)),
                        (_, Err(error)) => Support::unavailable("project.entrypoint.localUnavailable", error.clone()),
                    };
                    let cluster = match cluster_unit {
                        Ok((envelope, _)) => match crate::runtime::settings::validate_browser_interop(&envelope.manifest.interop).and_then(|_| runmat_execution::resource::accelerator_requirements_for_capabilities(&envelope.manifest.capabilities).map(|_| ()).map_err(|error| error.to_string())) {
                            Ok(()) => Support::available(),
                            Err(error) => Support::unavailable("project.entrypoint.clusterUnavailable", error),
                        },
                        Err(error) => Support::unavailable("project.entrypoint.clusterUnavailable", crate::wire::errors::js_value_to_string(error)),
                    };
                    (kind, local, cluster)
                };
                let target = match (&declared.path, &declared.module, &declared.function) {
                    (Some(path), _, _) => Target::Path { path: path.to_str().ok_or_else(|| failure("project.entrypoint.path", "declared path is not UTF-8"))?.into() },
                    (_, Some(module), Some(function)) => Target::ModuleFunction { module: module.clone(), function: function.clone() },
                    _ => return Err(failure("project.entrypoint.target", "Named target has no path or module/function")),
                };
                descriptions.push(Description { name: declared.name.clone(), kind, target, local, cluster });
            }
            crate::wire::json_contract_to_js(&descriptions).map_err(|error| failure("project.entrypoint.encoding", error))
        }).await.map_err(|error| failure("project.entrypoint.filesystem", error))?
    }
}

async fn compile_local(
    project: &CapturedProjectEntrypoints,
    selected: &runmat_config::project::ResolvedProjectEntrypoint,
    config: &crate::runtime::config::SessionConfig,
) -> Result<runmat_core::ExecutableUnit, JsValue> {
    let source = project
        .capture_source(selected)
        .await
        .map_err(|error| failure("project.entrypoint.source", error))?;
    let mut session = runmat_core::RunMatSession::with_options(false, false)
        .map_err(|error| failure("project.entrypoint.session", error))?;
    session
        .apply_runtime_config(&config.runtime)
        .map_err(|error| failure("project.entrypoint.policy", error))?;
    session
        .install_project_handoff(project.handoff.clone())
        .map_err(|error| failure("project.entrypoint.install", error))?;
    session
        .compile_executable_unit(source, None)
        .await
        .map_err(|error| {
            crate::wire::errors::run_error_to_js(&error, selected.source_file.to_str(), None)
        })
}
