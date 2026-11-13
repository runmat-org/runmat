use super::super::session::RunMatWasm;
use super::*;
use runmat_execution::ExecutableEntrypointKind;
use runmat_runtime::analysis::AnalysisExecutionChoices;

#[wasm_bindgen]
impl RunMatWasm {
    #[wasm_bindgen(js_name = runProjectEntrypoint)]
    pub async fn run_project_entrypoint_js(
        &self,
        start: String,
        name: String,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        let options: RunOptions = decode(options)?;
        let (original, paths) = runmat_filesystem::CapturedReadScope::capture(
            &[Path::new(&start), Path::new(".")],
            None,
        )
        .map_err(|error| failure("project.entrypoint.filesystem", error))?;
        let mut session = self.take_session()?;
        // This request owns preparation before the Core invocation is admitted.
        // A fresh generation avoids poisoning the next command when discovery
        // is cancelled before any Core execution admission exists to retire it.
        let interrupt = runmat_execution::cancellation::CancellationSource::new();
        let interrupt_guard = super::super::session::install_wasm_interrupt(
            &self.active_interrupt,
            interrupt.clone(),
        );
        let pinned = options
            .input_manifest
            .clone()
            .map(|inputs| session.capture_input_filesystem(&original, inputs))
            .transpose()
            .map_err(|error| crate::wire::errors::run_error_to_js(&error, Some(&start), None))?;
        let filesystem =
            crate::runtime::filesystem::captured::CapturedFilesystem::new(original, pinned)?;
        let read_scope = &filesystem.read;
        // Resolve settings, target and source under the immutable owner BEFORE
        // admitting/mutating the live execution session.
        let (project, target, config, source) =
            super::super::session::await_wasm_preparation(&interrupt, async {
                read_scope
                    .run(async {
                        let project = capture(&paths[0]).await?;
                        let target = project
                            .resolve(&name)
                            .await
                            .map_err(|error| failure("project.entrypoint.resolve", error))?;
                        let config = self
                            .config_for_named_project(
                                project.project_document.clone(),
                                options.user_settings_json.clone(),
                            )
                            .map_err(|error| failure("project.entrypoint.policy", error))?;
                        self.admit_named_config(&config).map_err(|error| {
                            failure("project.entrypoint.hostReinitializationRequired", error)
                        })?;
                        let source = if is_fea(&target) {
                            None
                        } else {
                            Some(
                                project
                                    .capture_source(&target)
                                    .await
                                    .map_err(|error| failure("project.entrypoint.source", error))?,
                            )
                        };
                        Ok::<_, JsValue>((project, target, config, source))
                    })
                    .await
                    .map_err(|error| failure("project.entrypoint.filesystem", error))?
            })
            .await?;
        let source_path = path(&target)?;
        let arguments: Vec<runmat_execution::value::ValuePayload> = options
            .arguments_json
            .as_deref()
            .map(serde_json::from_slice)
            .transpose()
            .map_err(|error| failure("project.entrypoint.arguments", error))?
            .unwrap_or_default();
        if is_fea(&target) {
            if !arguments.is_empty() || options.requested_outputs.is_some() {
                return Err(failure(
                    "project.entrypoint.feaOptions",
                    "FEA targets accept an input manifest, not function arguments or output arity",
                ));
            }
            {
                session
                    .apply_runtime_config(&config.runtime)
                    .map_err(|error| failure("project.entrypoint.policy", error))?;
                session
                    .install_project_handoff(project.handoff)
                    .map_err(|error| failure("project.entrypoint.install", error))?;
                self.publish_named_config(config);
            }
            drop(session);
            drop(interrupt_guard);
            let plan = self
                .plan_fea_source_in_scope(source_path.clone(), &filesystem)
                .await?;
            let plan: AnalysisExecutionChoices = serde_wasm_bindgen::from_value(plan)
                .map_err(|error| failure("project.entrypoint.plan", error))?;
            return result(
                Kind::FeaStudy,
                self.run_fea_source_in_scope(source_path, plan.plan_id, &filesystem)
                    .await?,
            );
        }
        session
            .apply_runtime_config(&config.runtime)
            .map_err(|error| failure("project.entrypoint.policy", error))?;
        session
            .install_project_handoff(project.handoff)
            .map_err(|error| failure("project.entrypoint.install", error))?;
        self.publish_named_config(config);
        let source = source.ok_or_else(|| {
            failure(
                "project.entrypoint.source",
                "Code target has no captured source",
            )
        })?;
        let unit = super::super::session::await_wasm_preparation(&interrupt, async {
            read_scope
                .run(session.compile_executable_unit(source, None))
                .await
                .map_err(|error| failure("project.entrypoint.filesystem", error))?
                .map_err(|error| {
                    crate::wire::errors::run_error_to_js(&error, Some(&source_path), None)
                })
        })
        .await?;
        let (kind, semantic) = unit
            .entrypoint_identity(target.function.as_deref())
            .map_err(|error| failure("project.entrypoint.compile", error))?;
        crate::runtime::settings::validate_browser_interop(&unit.inferred_interop_manifest())
            .map_err(|error| failure("project.entrypoint.nativeRuntimeRequired", error))?;
        match kind {
            ExecutableEntrypointKind::Script => {
                if !arguments.is_empty()
                    || options.requested_outputs.is_some_and(|count| count != 0)
                {
                    return Err(failure(
                        "project.entrypoint.scriptArguments",
                        "Scripts do not accept positional arguments or ordinal outputs",
                    ));
                }
                drop(session);
                drop(interrupt_guard);
                let request = serde_json::json!({"source":{"kind":"path","path":source_path},"inputManifest":options.input_manifest});
                return result(
                    Kind::Script,
                    filesystem
                        .run_original(
                            self.execute_request_js(
                                crate::wire::json_contract_to_js(&request).map_err(|error| {
                                    failure("project.entrypoint.encoding", error)
                                })?,
                            ),
                        )
                        .await??,
                );
            }
            ExecutableEntrypointKind::DeclarationOnly => {
                return Err(failure(
                    "project.entrypoint.notExecutable",
                    "Named target is declaration-only",
                ))
            }
            ExecutableEntrypointKind::Function => {}
        }
        let runtime = session
            .execution_runtime_context(unit.revision().program_revision.clone())
            .with_cancellation_token(interrupt.token());
        let arguments =
            runmat_runtime::execution::value_codec::decode_inline_values_cooperatively_in_context(
                &arguments,
                Default::default(),
                None,
                None,
                &runtime,
            )
            .await
            .map_err(|error| failure("project.entrypoint.arguments", error))?;
        let requested = options.requested_outputs.unwrap_or(1);
        let invocation = runmat_core::ProcedureInvocation {
            target: runmat_core::ProcedureTarget::SemanticFunction(semantic.ok_or_else(|| {
                failure(
                    "project.entrypoint.notExecutable",
                    "Target has no compiler function identity",
                )
            })?),
            arguments: runmat_core::ProcedureArguments::Retained(arguments.into()),
            requested_outputs: requested,
        };
        let control =
            runmat_core::InvocationControl::default().with_cancellation(interrupt.token());
        let values = match &filesystem.pinned {
            Some(pinned) => {
                session
                    .invoke_executable_with_inputs(&unit, invocation, &control, pinned)
                    .await
            }
            None => {
                filesystem
                    .run_original(session.invoke_executable(&unit, invocation, &control))
                    .await?
            }
        }
        .map_err(|error| crate::wire::errors::run_error_to_js(&error, Some(&source_path), None))?;
        let outputs = runmat_runtime::execution::value_codec::encode_inline_sequence(
            values.value(),
            requested,
        )
        .map_err(|error| failure("project.entrypoint.outputs", error))?;
        let outputs = serde_json::to_vec(&outputs)
            .map_err(|error| failure("project.entrypoint.encoding", error))?;
        result(
            Kind::TypedFunction,
            js_sys::Uint8Array::from(outputs.as_slice()).into(),
        )
    }
}
