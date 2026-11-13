use super::*;
use std::rc::Rc;

#[path = "run/command.rs"]
mod command;
#[path = "run/contextual.rs"]
mod contextual;
#[path = "run/phases.rs"]
mod phases;
mod preparation;
mod request_scope;
#[path = "run/selected_class_product.rs"]
mod selected_class_product;
#[path = "run/selected_source_owner.rs"]
mod selected_source_owner;

#[cfg(not(target_arch = "wasm32"))]
fn entrypoint_target_function(
    assembly: &runmat_hir::HirAssembly,
) -> Option<runmat_hir::FunctionId> {
    assembly
        .entrypoints
        .first()
        .map(|entrypoint| entrypoint.target)
}

#[cfg(not(target_arch = "wasm32"))]
fn mir_local_fact_count_for_entrypoint(
    analysis: &runmat_mir::analysis::AnalysisStore,
    assembly: &runmat_hir::HirAssembly,
) -> usize {
    let function = entrypoint_target_function(assembly)
        .and_then(|function| analysis.function_for_local(function))
        .map(|facts| facts.function);
    analysis.local_value_count(function)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn discover_source_catalog(
    source_name: Option<&str>,
) -> Option<runmat_package::DiscoveredSourceSymbols> {
    use runmat_package::discover_source_symbols_from_source_name;
    use std::path::{Path, PathBuf};

    let Ok(cwd) = runmat_filesystem::current_dir() else {
        let source_name = source_name?;
        let source_path = PathBuf::from(source_name);
        if crate::diagnostic_path::host_path_is_absolute(&source_path) {
            let source_cwd = source_path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("/"));
            return discover_source_symbols_from_source_name(source_name, &source_cwd)
                .ok()
                .flatten();
        }
        return None;
    };
    let source_name = source_name?;
    discover_source_symbols_from_source_name(source_name, &cwd)
        .ok()
        .flatten()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn discover_known_project_symbols(source_name: Option<&str>) -> HashSet<String> {
    discover_source_catalog(source_name)
        .map(|catalog| catalog.symbols)
        .unwrap_or_default()
}

impl RunMatSession {
    pub(super) fn configure_dynamic_function_services(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.runtime_context
            .set_native_name_observer(Some(self.mex_runtime.name_observer()));
        let dynamic_environment = Rc::new(super::dynamic::DynamicFunctionEnvironment {
            compat: self.compat_mode,
            top_level_await_enabled: self.top_level_await_enabled,
            cache: Rc::clone(&self.dynamic_function_cache),
            project_handoff: self.project_handoff.clone(),
            source_evaluator: Rc::clone(&self.source_evaluator),
            #[cfg(not(target_arch = "wasm32"))]
            mex_runtime: std::rc::Rc::clone(&self.mex_runtime),
        });
        let class_environment = dynamic_environment.clone();
        let class_resolver: std::rc::Rc<runmat_runtime::context::CurrentClassProductResolver> =
            std::rc::Rc::new(move |runtime, class| {
                let environment = Rc::clone(&class_environment);
                runmat_runtime::context::admit_current_class_product_with(
                    &runtime.clone(),
                    move || {
                        super::dynamic::resolve_current_class_product(runtime, class, environment)
                    },
                )
            });
        self.runtime_context
            .set_current_class_product_resolver(Some(class_resolver));
        let loader: std::rc::Rc<runmat_runtime::user_functions::DynamicFunctionLoader> =
            std::rc::Rc::new(
                move |runtime, name, args, requested_outputs, phase, input_callsite| {
                    let environment = Rc::clone(&dynamic_environment);
                    runmat_runtime::execution::admit_dynamic_function_load_with(
                        &runtime.clone(),
                        move || {
                            super::dynamic::load_dynamic_function(
                                runtime,
                                name,
                                args,
                                requested_outputs,
                                phase,
                                input_callsite,
                                environment,
                            )
                        },
                    )
                },
            );
        self.runtime_context
            .set_dynamic_function_loader(Some(loader));

        let dynamic_function_cache = Rc::clone(&self.dynamic_function_cache);
        #[cfg(not(target_arch = "wasm32"))]
        let mex_runtime = std::rc::Rc::clone(&self.mex_runtime);
        let clearer: std::rc::Rc<runmat_runtime::user_functions::DynamicFunctionClearer> =
            std::rc::Rc::new(move |runtime, request| {
                let dynamic_function_cache = Rc::clone(&dynamic_function_cache);
                #[cfg(not(target_arch = "wasm32"))]
                let mex_runtime = std::rc::Rc::clone(&mex_runtime);
                runmat_runtime::execution::admit_domain_continuation_with(
                    &runtime.clone(),
                    move || async move {
                        if !matches!(
                        request,
                        runmat_runtime::user_functions::DynamicFunctionClearRequest::NativeExtensions
                    ) {
                        match &request {
                            runmat_runtime::user_functions::DynamicFunctionClearRequest::All => {
                                dynamic_function_cache.clear();
                            }
                            runmat_runtime::user_functions::DynamicFunctionClearRequest::Named(
                                name,
                            ) => {
                                dynamic_function_cache.clear_named(name);
                            }
                            runmat_runtime::user_functions::DynamicFunctionClearRequest::NativeExtensions => {}
                        }
                    }
                        #[cfg(not(target_arch = "wasm32"))]
                        mex_runtime
                            .clear_gracefully(&request, runtime.clone())
                            .await?;
                        // A successful typed function clear invalidates its
                        // selected session's persistent bindings. Do this after
                        // fallible module retirement so a failed clear cannot
                        // erase live function state.
                        let authority = runtime.session_variable_authority();
                        match &request {
                        runmat_runtime::user_functions::DynamicFunctionClearRequest::All => {
                            authority.clear_all_persistents()?;
                        }
                        runmat_runtime::user_functions::DynamicFunctionClearRequest::Named(name) => {
                            authority.clear_persistent_function(name)?;
                        }
                        runmat_runtime::user_functions::DynamicFunctionClearRequest::NativeExtensions => {}
                    }
                        Ok(())
                    },
                )
            });
        self.runtime_context
            .set_dynamic_function_clearer(Some(clearer));

        let refresh_cache = Rc::clone(&self.dynamic_function_cache);
        let refresher: std::rc::Rc<runmat_runtime::user_functions::DynamicFunctionRefresher> =
            std::rc::Rc::new(move |runtime, _request| {
                let cache = Rc::clone(&refresh_cache);
                runmat_runtime::execution::admit_domain_continuation_with(
                    &runtime.clone(),
                    move || async move {
                        // This refresh is one invocation-owned cache mutation.
                        // Yield/cancel before changing either revision or entries;
                        // the two mutations then form one short synchronous step.
                        runtime
                            .cpu_cooperation()
                            .checkpoint(1)
                            .await
                            .map_err(|source| source.into_runtime_error())?;
                        // RunMat has one live source cache, not a separate persisted
                        // matlabroot/toolbox cache. All four public requests refresh
                        // its lookup revision and compiled source entries; none
                        // unload a MEX module or clear workspace values.
                        if let Some(path) = runtime.search_path() {
                            path.refresh_generation().map_err(|source| {
                                build_runtime_error(format!("rehash: {source}"))
                                    .with_builtin("rehash")
                                    .with_source(source)
                                    .build()
                            })?;
                        }
                        cache.clear();
                        Ok(())
                    },
                )
            });
        self.runtime_context
            .set_dynamic_function_refresher(Some(refresher));
    }

    async fn run(
        &mut self,
        input: &str,
    ) -> std::result::Result<
        runmat_runtime::execution::RootedResult<crate::abi::ExecutionOutcome>,
        RunError,
    > {
        self.runtime_context = self.runtime_context.clone().with_fresh_plot_receipts();
        self.configure_runtime_context();
        let runtime = self
            .runtime_context
            .clone()
            .with_cancellation_token(self.runtime_context.cancellation());
        runtime.scope(self.run_in_context(input)).await
    }

    fn run_in_context<'a>(
        &'a mut self,
        input: &'a str,
    ) -> impl std::future::Future<
        Output = std::result::Result<
            runmat_runtime::execution::RootedResult<crate::abi::ExecutionOutcome>,
            RunError,
        >,
    > + 'a {
        runmat_runtime::execution::after_result(
            contextual::prepare(self, input),
            contextual::execute,
        )
    }
    /// Execute a structured runtime/workspace ABI request.
    pub async fn execute_request(
        &mut self,
        request: crate::abi::ExecutionRequest,
    ) -> crate::abi::ExecutionResponse {
        let _admission = self.admit_execution(None);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let local_tasks = self.local_tasks.clone();
            // The driver rejects admission before polling its child. Keep the
            // request outside that child until admission so an error can
            // report its source without copying source text on the hot path.
            let mut request = Some(request);
            return match local_tasks
                .participate_foreground(async {
                    self.execute_request_with_cleanup(
                        request.take().expect("execution request polled only once"),
                    )
                    .await
                })
                .await
            {
                Ok(response) => response,
                Err(error) => crate::abi::ExecutionResponse {
                    dependency_lock: None,
                    source_context: unresolved_source_context(
                        &request
                            .as_ref()
                            .expect("rejected foreground request was not polled")
                            .source,
                    ),
                    result: Err(RunError::Runtime(error)),
                },
            };
        }
        #[cfg(target_arch = "wasm32")]
        self.execute_request_with_cleanup(request).await
    }

    async fn execute_request_with_cleanup(
        &mut self,
        request: crate::abi::ExecutionRequest,
    ) -> crate::abi::ExecutionResponse {
        let command_start = Instant::now();
        let profiling_root = self.admit_command_profiling();
        let receipt = self
            .runtime_context
            .command_profiling()
            .cloned()
            .expect("admitted command");
        let mut response = runmat_execution::profiling::observe_command_future(
            receipt,
            self.execute_request_and_cleanup(request),
        )
        .await;
        // The owned future and its final selected poll/drop scope have retired
        // before the sole root publishes an immutable CPU snapshot.
        if let Some(admission) = profiling_root {
            let canonical_device_route = admission.canonical_device_route
                && self
                    .runtime_context
                    .command_device_access()
                    .is_some_and(|access| !access.has_raw_export())
                && self
                    .runtime_context
                    .service_ports()
                    .same_device_route_owners(&self.canonical_device_route_ports);
            self.observe_command_device_time_at_seal(&admission.root);
            let snapshot = admission.root.seal();
            let command_wall_ms = command_start.elapsed().as_millis() as u64;
            response.result = response.result.and_then(|outcome| {
                outcome.try_map_owned(
                    |mut outcome| {
                        outcome.profiling = Some(execution_profiling(
                            command_wall_ms,
                            &snapshot,
                            canonical_device_route,
                        ));
                        Ok::<_, RunError>(outcome)
                    },
                    |error| RunError::Runtime(error.into_runtime_error()),
                )
            });
        }
        response
    }

    async fn execute_request_and_cleanup(
        &mut self,
        request: crate::abi::ExecutionRequest,
    ) -> crate::abi::ExecutionResponse {
        let _input = self.install_lifecycle_input();
        let mut response = self.execute_request_inner(request).await;
        if let Err(error) = self.lifecycle_host.service_safepoint(&self.runtime_context) {
            if response
                .result
                .as_ref()
                .is_ok_and(|outcome| !outcome.has_errors())
            {
                response.result = Err(RunError::Runtime(error));
            } else {
                tracing::warn!(%error, "cleanup safepoint failed after an execution error");
            }
        }
        if let Err(error) = self
            .lifecycle_host
            .drain_gracefully(&self.runtime_context)
            .await
        {
            if response
                .result
                .as_ref()
                .is_ok_and(|outcome| !outcome.has_errors())
            {
                response.result = Err(RunError::Runtime(error));
            } else {
                tracing::warn!(%error, "cleanup callback failed after an execution error");
            }
        }
        response
    }

    async fn execute_request_inner(
        &mut self,
        request: crate::abi::ExecutionRequest,
    ) -> crate::abi::ExecutionResponse {
        let requested_outputs = request.requested_outputs;
        let source_input = request.source.clone();
        if let Err(error) = self.ensure_user_path_hydrated().await {
            return crate::abi::ExecutionResponse {
                dependency_lock: None,
                source_context: unresolved_source_context(&source_input),
                result: Err(error),
            };
        }
        let source_resolution = match source_input_text(request.source).await {
            Ok(resolved) => resolved,
            Err(err) => {
                return crate::abi::ExecutionResponse {
                    dependency_lock: None,
                    source_context: unresolved_source_context(&source_input),
                    result: Err(err),
                };
            }
        };
        let source_name = source_resolution.display_name;
        let source_fullpath_name = source_resolution.fullpath_name;
        let source_text = source_resolution.text;
        let (primary_source, admitted_fullpath) = match (
            &source_input,
            source_fullpath_name.as_deref(),
            self.project_handoff.as_ref(),
        ) {
            (
                crate::abi::SourceInput::Path(_) | crate::abi::SourceInput::CapturedPath(_),
                Some(path),
                Some(handoff),
            ) => {
                match crate::executable::AdmittedProjectPrimary::issue(
                    &handoff.project,
                    std::path::Path::new(path),
                    &source_text,
                )
                .await
                {
                    Ok(source) => {
                        let (captured, access_path) = source.into_request_parts();
                        (captured, Some(access_path))
                    }
                    Err(error) => {
                        return crate::abi::ExecutionResponse {
                            dependency_lock: None,
                            source_context: crate::abi::ExecutionSourceContext {
                                name: source_name,
                                text: Some(source_text),
                                identity: None,
                            },
                            result: Err(RunError::Runtime(
                                build_runtime_error(error)
                                    .with_identifier("RunMat:ExecutableSourceIdentity")
                                    .build(),
                            )),
                        };
                    }
                }
            }
            _ => (None, source_fullpath_name.clone()),
        };
        let source_identity = resolve_source_identity(&source_input, &source_text);
        let mut scope = request_scope::RequestExecutionScope::enter(self);
        let session = scope.session();
        session.compat_mode = request.compatibility;
        session.top_level_await_enabled = request.host_policy.top_level_await;
        session.dynamic_eval_enabled = request.host_policy.dynamic_eval;
        session.active_source_name = source_name.clone();
        session.active_source_fullpath_name = admitted_fullpath;
        session.abi_workspace_handle = request.workspace;
        session.active_source_identity = source_identity.clone();
        session.active_primary_source = primary_source;

        let result = session.run(&source_text).await;
        drop(scope);

        crate::abi::ExecutionResponse {
            dependency_lock: None,
            source_context: crate::abi::ExecutionSourceContext {
                name: source_name,
                text: Some(source_text),
                identity: source_identity,
            },
            result: result.and_then(|outcome| {
                outcome.try_map_owned(
                    |outcome| apply_requested_output_policy(outcome, &requested_outputs),
                    |error| RunError::Runtime(error.into_runtime_error()),
                )
            }),
        }
    }

    #[tracing::instrument(
        name = "runtime.execute",
        skip_all,
        fields(input_len = input.len(), verbose = self.verbose)
    )]
    fn execute_internal<'a>(
        &'a mut self,
        input: &'a str,
        preserve_layout_var_names: bool,
    ) -> impl std::future::Future<Output = std::result::Result<SessionExecution, RunError>> + 'a
    {
        // Attach the span when constructing the request-owned future. An
        // async instrument wrapper otherwise reserves its full construction
        // frame during every nested interpreter poll.
        tracing::Instrument::instrument(
            self.execute_internal_inner(input, preserve_layout_var_names),
            tracing::Span::current(),
        )
    }

    fn execute_internal_inner<'a>(
        &'a mut self,
        input: &'a str,
        preserve_layout_var_names: bool,
    ) -> impl std::future::Future<Output = std::result::Result<SessionExecution, RunError>> + 'a
    {
        runmat_runtime::execution::after_result(
            self.prepare_command(input, preserve_layout_var_names),
            command::run,
        )
    }

    /// Interpret bytecode with persistent variable context
    async fn interpret_with_context(
        &mut self,
        bytecode: &runmat_vm::Bytecode,
        variable_array: runmat_vm::InterpreterVars,
        runtime: runmat_runtime::context::RuntimeContext,
        admission: Option<runmat_vm::PendingWorkspaceAdmission>,
    ) -> runmat_vm::WorkspaceInterpreterOutcome {
        let source_name = self.current_source_name().to_string();
        runmat_vm::interpret_workspace(
            bytecode,
            variable_array,
            Some(source_name.as_str()),
            runtime,
            admission,
            Some(self.source_evaluator.dynamic_admission_port()),
        )
        .await
    }

    fn abi_workspace_upserts(
        &self,
        workspace_names: Vec<String>,
    ) -> std::result::Result<Vec<crate::abi::WorkspaceBindingValue>, RunError> {
        let workspace = self.logical_workspace_values().map_err(RunError::Runtime)?;
        let mut workspace_names = workspace_names;
        workspace_names.sort();
        workspace_names.dedup();
        let mut upserts = Vec::new();
        for name in workspace_names {
            let Some(value) = workspace.get(&name) else {
                continue;
            };
            let key = self
                .admitted_workspace_binding_key(&name)
                .map_err(RunError::Runtime)?;
            upserts.push(crate::abi::WorkspaceBindingValue {
                key,
                name: runmat_hir::BindingName(name),
                value: value.clone(),
            });
        }
        Ok(upserts)
    }

    fn abi_workspace_removals(
        &self,
        previous_workspace_bindings: HashMap<String, crate::abi::WorkspaceBindingKey>,
        include_missing_names: bool,
    ) -> std::result::Result<(Vec<crate::abi::WorkspaceBindingRemoval>, bool), RunError> {
        let workspace = self.logical_workspace_names().map_err(RunError::Runtime)?;
        let mut removed_names = false;
        let mut removed = previous_workspace_bindings
            .into_iter()
            .filter(|(name, previous_key)| {
                if !workspace.contains(name) {
                    removed_names |= include_missing_names;
                    return include_missing_names;
                }
                self.workspace_bindings
                    .borrow()
                    .get(name.as_str())
                    .map(|binding| &binding.key)
                    .is_some_and(|current_key| current_key != previous_key)
            })
            .collect::<Vec<_>>();
        removed.sort_by(|left, right| left.0.cmp(&right.0));
        Ok((
            removed
                .into_iter()
                .map(|(name, key)| crate::abi::WorkspaceBindingRemoval {
                    key,
                    name: runmat_hir::BindingName(name),
                })
                .collect(),
            removed_names,
        ))
    }
}

fn apply_requested_output_policy(
    mut outcome: crate::abi::ExecutionOutcome,
    requested_outputs: &runmat_hir::RequestedOutputCount,
) -> Result<crate::abi::ExecutionOutcome, RunError> {
    use crate::abi::RuntimeFlow;
    use runmat_hir::RequestedOutputCount;
    use runmat_value::{ValueSequence, ValueSequenceKind};

    let flow = match outcome.flow {
        RuntimeFlow::DynamicList(handle) => {
            outcome.flow = match requested_outputs {
                RequestedOutputCount::Zero | RequestedOutputCount::Exactly(0) => {
                    RuntimeFlow::NoValue
                }
                _ => RuntimeFlow::DynamicList(handle),
            };
            return Ok(outcome);
        }
        RuntimeFlow::NoValue => Ok(ValueSequence::empty()),
        RuntimeFlow::Single(value) => ValueSequence::single(value),
        RuntimeFlow::OutputList(values) => ValueSequence::from_output_storage(values),
    }
    .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
    .map_err(RunError::Runtime)?;

    outcome.flow = match requested_outputs {
        RequestedOutputCount::Zero => RuntimeFlow::NoValue,
        RequestedOutputCount::One => first_runtime_flow_value(flow)?,
        RequestedOutputCount::Exactly(count) => {
            if *count == 0 {
                RuntimeFlow::NoValue
            } else if *count == 1 {
                first_runtime_flow_value(flow)?
            } else {
                let mut values = flow.into_output_storage();
                values.truncate(*count);
                RuntimeFlow::OutputList(values)
            }
        }
        RequestedOutputCount::CurrentFunctionNargout
        | RequestedOutputCount::DestinationSequenceCardinality => match flow.kind() {
            ValueSequenceKind::Single => first_runtime_flow_value(flow)?,
            ValueSequenceKind::CommaSeparated => {
                RuntimeFlow::OutputList(flow.into_output_storage())
            }
        },
    };
    Ok(outcome)
}

fn first_runtime_flow_value(
    sequence: runmat_value::ValueSequence,
) -> Result<crate::abi::RuntimeFlow, RunError> {
    sequence
        .into_output_storage()
        .try_into_first()
        .map_err(|_| {
            RunError::Runtime(runmat_runtime::sequence::sequence_error_to_runtime(
                runmat_value::ValueSequenceError::SharedOutputStorage,
            ))
        })
        .map(|value| {
            value
                .map(crate::abi::RuntimeFlow::Single)
                .unwrap_or(crate::abi::RuntimeFlow::NoValue)
        })
}

fn resolve_source_identity(
    source: &crate::abi::SourceInput,
    source_text: &str,
) -> Option<crate::abi::SourceIdentity> {
    match source {
        crate::abi::SourceInput::Path(path) => {
            Some(crate::abi::SourceIdentity::PathAndContentHash {
                path: path.clone(),
                hash: source_text_hash(source_text),
            })
        }
        crate::abi::SourceInput::CapturedPath(source) => {
            Some(crate::abi::SourceIdentity::PathAndContentHash {
                path: source.access_path().to_owned(),
                hash: source_text_hash(source_text),
            })
        }
        crate::abi::SourceInput::Text { name, .. } => {
            if name.starts_with('<') {
                None
            } else {
                Some(crate::abi::SourceIdentity::PathAndContentHash {
                    path: name.clone(),
                    hash: source_text_hash(source_text),
                })
            }
        }
    }
}

fn source_text_hash(source_text: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source_text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

pub(super) fn unresolved_source_context(
    source: &crate::abi::SourceInput,
) -> crate::abi::ExecutionSourceContext {
    let name = match source {
        crate::abi::SourceInput::Path(path) => path.clone(),
        crate::abi::SourceInput::Text { name, .. } => name.clone(),
        crate::abi::SourceInput::CapturedPath(source) => source.display_name().to_owned(),
    };
    crate::abi::ExecutionSourceContext {
        name,
        text: match source {
            crate::abi::SourceInput::Text { text, .. } => Some(text.clone()),
            crate::abi::SourceInput::CapturedPath(source) => Some(source.text().to_owned()),
            crate::abi::SourceInput::Path(_) => None,
        },
        identity: None,
    }
}

fn runtime_error_span(error: &runmat_runtime::RuntimeError) -> Option<runmat_hir::Span> {
    error.span.as_ref().map(|span| {
        let start = span.offset();
        runmat_hir::Span {
            start,
            end: start + span.len().max(1),
        }
    })
}

fn execution_workspace_mapping(bytecode: &runmat_vm::Bytecode) -> HashMap<String, usize> {
    let Some(layout) = &bytecode.layout else {
        return HashMap::new();
    };
    let mut mapping = HashMap::new();
    for entrypoint in layout.entrypoints.values() {
        for export in &entrypoint.exports {
            mapping.insert(export.name.clone(), export.slot.0);
        }
    }
    // The VM compiler reserves later reads of names that a current-workspace
    // eval may create. They are not entrypoint exports yet, but the pending
    // workspace admission must know their exact slots before eval runs; the
    // separate assigned snapshot keeps them unassigned until a real write.
    for slot in &bytecode.initially_unassigned_slots {
        if let Some(name) = bytecode.var_names.get(slot) {
            mapping.insert(name.clone(), *slot);
        }
    }
    mapping
}

fn entry_function(assembly: &runmat_hir::HirAssembly) -> Option<&runmat_hir::HirFunction> {
    let entrypoint = assembly.entrypoints.first()?;
    assembly
        .functions
        .iter()
        .find(|function| function.id == entrypoint.target)
}

fn entry_statement_count(assembly: &runmat_hir::HirAssembly) -> usize {
    entry_function(assembly)
        .map(|function| function.body.statements.len())
        .unwrap_or(0)
}

fn first_entry_statement(assembly: &runmat_hir::HirAssembly) -> Option<&runmat_hir::HirStmt> {
    entry_function(assembly)?.body.statements.first()
}

struct SessionExecution {
    outcome: runmat_runtime::execution::RootedResult<crate::abi::ExecutionOutcome>,
    workspace_snapshot: WorkspaceSnapshot,
}

#[derive(Debug)]
struct ResolvedSourceInput {
    display_name: String,
    fullpath_name: Option<String>,
    text: String,
}

async fn source_input_text(
    source: crate::abi::SourceInput,
) -> std::result::Result<ResolvedSourceInput, RunError> {
    match source {
        crate::abi::SourceInput::CapturedPath(source) => {
            let (display_name, access_path, text) = source.into_parts();
            Ok(ResolvedSourceInput {
                display_name,
                fullpath_name: Some(access_path),
                text,
            })
        }
        crate::abi::SourceInput::Text { name, text } => Ok(ResolvedSourceInput {
            display_name: name,
            fullpath_name: None,
            text,
        }),
        crate::abi::SourceInput::Path(path) => {
            let source_path = resolve_path_source_input(&path).await?;
            let source_name = crate::diagnostic_path::display_path_for_current_cwd(&source_path);
            let source_fullpath_name = source_path.to_string_lossy().to_string();

            let text = match runmat_runtime::builtins::io::repl_fs::pcode::read_source_text_async(
                &source_path,
            )
            .await
            {
                Ok(text) => text,
                Err(
                    runmat_runtime::builtins::io::repl_fs::pcode::PcodeSourceReadError::InvalidPcode(
                        err,
                    ),
                ) => {
                    return Err(RunError::Runtime(
                        runmat_runtime::builtins::io::repl_fs::pcode::invalid_pcode_runtime_error(
                            format!("{} ({err})", source_path.display()),
                        ),
                    ));
                }
                Err(runmat_runtime::builtins::io::repl_fs::pcode::PcodeSourceReadError::Io(err)) => {
                    return Err(RunError::Runtime(
                        build_runtime_error(format!(
                            "failed to read source path '{}': {err}",
                            source_path.display()
                        ))
                        .with_identifier("RunMat:SourceReadFailed")
                        .with_source(err)
                        .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                        .build(),
                    ));
                }
            };
            Ok(ResolvedSourceInput {
                display_name: source_name,
                fullpath_name: Some(source_fullpath_name),
                text,
            })
        }
    }
}

impl crate::abi::SourceInput {
    /// Read once through the same resolver as Path and retain those exact bytes
    /// for repeated invocations such as benchmark warmup and measured runs.
    pub async fn capture_path(path: impl Into<String>) -> Result<Self, RunError> {
        let resolved = source_input_text(Self::Path(path.into())).await?;
        let access_path = resolved.fullpath_name.ok_or_else(|| {
            RunError::Runtime(
                build_runtime_error("captured path source has no issued access path")
                    .with_identifier("RunMat:ExecutableSourceIdentity")
                    .build(),
            )
        })?;
        Ok(Self::CapturedPath(crate::abi::CapturedPathSource::new(
            resolved.display_name,
            access_path,
            resolved.text,
        )))
    }
}

async fn resolve_path_source_input(
    path: &str,
) -> std::result::Result<std::path::PathBuf, RunError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use runmat_config::project::resolve_project_source_input_from;
        use std::path::Path;

        let cwd = runmat_filesystem::current_dir().map_err(|err| {
            RunError::Runtime(
                build_runtime_error(format!(
                    "failed to resolve current working directory while resolving source path '{path}': {err}"
                ))
                .with_identifier("RunMat:SourceResolveFailed")
                .with_source(err)
                .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                .build(),
            )
        })?;

        let source_path = std::path::PathBuf::from(path);
        let candidate = crate::diagnostic_path::resolve_against_base(path, &cwd);

        if runmat_runtime::builtins::common::path_search::path_is_file(&candidate)
            .await
            .map_err(|error| RunError::Runtime(error.into_runtime_error("source resolution")))?
        {
            return runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(
                &candidate,
            )
            .await
            .map_err(|error| RunError::Runtime(error.into_runtime_error("source resolution")));
        }

        if source_path.extension().is_none() {
            for extension in ["p", "m"] {
                let with_ext = candidate.with_extension(extension);
                if runmat_runtime::builtins::common::path_search::path_is_file(&with_ext)
                    .await
                    .map_err(|error| {
                        RunError::Runtime(error.into_runtime_error("source resolution"))
                    })?
                {
                    return runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(
                        &with_ext,
                    )
                    .await
                    .map_err(|error| {
                        RunError::Runtime(error.into_runtime_error("source resolution"))
                    });
                }
            }
        }

        let resolved = resolve_project_source_input_from(&cwd, Path::new(path)).map_err(|err| {
            RunError::Runtime(
                build_runtime_error(format!(
                    "failed to resolve source input '{}' from working directory {}: {}",
                    path,
                    cwd.display(),
                    err
                ))
                .with_identifier("RunMat:EntrypointResolveFailed")
                .build(),
            )
        })?;
        let resolved = if resolved.is_absolute() {
            resolved
        } else {
            cwd.join(resolved)
        };
        runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(&resolved)
            .await
            .map_err(|error| RunError::Runtime(error.into_runtime_error("source resolution")))
    }

    #[cfg(target_arch = "wasm32")]
    {
        use std::path::PathBuf;

        let cwd = runmat_filesystem::current_dir().map_err(|err| {
            RunError::Runtime(
                build_runtime_error(format!(
                    "failed to resolve current working directory while resolving source path '{path}': {err}"
                ))
                .with_identifier("RunMat:SourceResolveFailed")
                .with_source(err)
                .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                .build(),
            )
        })?;
        let source_path = PathBuf::from(path);
        let candidate = crate::diagnostic_path::resolve_against_base(path, &cwd);

        if runmat_runtime::builtins::common::path_search::path_is_file(&candidate)
            .await
            .map_err(|error| RunError::Runtime(error.into_runtime_error("source resolution")))?
        {
            return Ok(
                runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(&candidate)
                    .await
                    .map_err(|error| {
                        RunError::Runtime(error.into_runtime_error("source resolution"))
                    })?,
            );
        }

        if source_path.extension().is_none() {
            for extension in ["p", "m"] {
                let with_ext = candidate.with_extension(extension);
                if runmat_runtime::builtins::common::path_search::path_is_file(&with_ext)
                    .await
                    .map_err(|error| {
                        RunError::Runtime(error.into_runtime_error("source resolution"))
                    })?
                {
                    return Ok(
                        runmat_runtime::builtins::io::repl_fs::pcode::prefer_pcode_source_path(
                            &with_ext,
                        )
                        .await
                        .map_err(|error| {
                            RunError::Runtime(error.into_runtime_error("source resolution"))
                        })?,
                    );
                }
            }
        }

        Ok(candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::apply_requested_output_policy;
    #[cfg(not(target_arch = "wasm32"))]
    use super::discover_known_project_symbols;
    #[cfg(not(target_arch = "wasm32"))]
    use super::source_input_text;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::abi::SourceInput;
    use crate::abi::{ExecutionOutcome, RuntimeFlow};
    #[cfg(not(target_arch = "wasm32"))]
    use crate::RunError;
    use runmat_hir::RequestedOutputCount;
    use runmat_value::Value;
    #[cfg(not(target_arch = "wasm32"))]
    use std::fs;
    #[cfg(not(target_arch = "wasm32"))]
    use std::path::{Path, PathBuf};
    #[cfg(not(target_arch = "wasm32"))]
    use std::sync::Arc;

    #[test]
    fn requested_output_policy_maps_checked_zero_one_and_many_results() {
        let outcome = |flow| ExecutionOutcome {
            flow,
            ..ExecutionOutcome::default()
        };

        let zero = apply_requested_output_policy(
            outcome(RuntimeFlow::output_list(vec![Value::Num(1.0)])),
            &RequestedOutputCount::Zero,
        )
        .unwrap();
        assert!(matches!(zero.flow, RuntimeFlow::NoValue));

        let one = apply_requested_output_policy(
            outcome(RuntimeFlow::output_list(vec![
                Value::Num(1.0),
                Value::Num(2.0),
            ])),
            &RequestedOutputCount::One,
        )
        .unwrap();
        assert!(matches!(one.flow, RuntimeFlow::Single(Value::Num(1.0))));

        let many = apply_requested_output_policy(
            outcome(RuntimeFlow::Single(Value::Num(1.0))),
            &RequestedOutputCount::Exactly(3),
        )
        .unwrap();
        assert!(matches!(
            many.flow,
            RuntimeFlow::OutputList(ref values) if values.as_slice() == [Value::Num(1.0)]
        ));
    }

    #[test]
    fn requested_output_policy_rejects_legacy_sequences_at_any_depth() {
        for flow in [
            RuntimeFlow::Single(Value::output_list(vec![Value::Num(1.0)])),
            RuntimeFlow::output_list(vec![Value::output_list(vec![Value::Num(1.0)])]),
        ] {
            let error = apply_requested_output_policy(
                ExecutionOutcome {
                    flow,
                    ..ExecutionOutcome::default()
                },
                &RequestedOutputCount::Zero,
            )
            .unwrap_err();
            let RunError::Runtime(error) = error else {
                panic!("expected runtime error");
            };
            assert_eq!(error.identifier(), Some("RunMat:NestedLegacyOutputList"));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    struct CwdGuard {
        original: PathBuf,
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn cwd_lock() -> std::sync::MutexGuard<'static, ()> {
        runmat_filesystem::provider_override_lock()
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn push_cwd(path: &Path) -> CwdGuard {
        let original = std::env::current_dir().expect("read cwd");
        std::env::set_current_dir(path).expect("set cwd");
        CwdGuard { original }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_resolves_named_manifest_entrypoint() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[entrypoints.main]
path = "src/main"
"#,
        )
        .unwrap();
        let _cwd = push_cwd(tmp.path());
        let resolved =
            futures::executor::block_on(source_input_text(SourceInput::Path("main".to_string())))
                .expect("named entrypoint should resolve");
        assert_eq!(
            PathBuf::from(&resolved.display_name),
            PathBuf::from("src").join("main.m")
        );
        let resolved_path = std::path::PathBuf::from(
            resolved
                .fullpath_name
                .as_deref()
                .expect("path source should carry fullpath name"),
        )
        .canonicalize()
        .unwrap();
        let expected = tmp.path().join("src/main.m").canonicalize().unwrap();
        assert_eq!(
            resolved_path, expected,
            "resolved source path should match manifest entrypoint target"
        );
        assert_eq!(resolved.text, "x = 1;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_infers_m_extension_for_relative_path() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        let _cwd = push_cwd(tmp.path());

        let resolved = futures::executor::block_on(source_input_text(SourceInput::Path(
            "src/main".to_string(),
        )))
        .expect("path without extension should infer .m");

        assert_eq!(
            PathBuf::from(&resolved.display_name),
            PathBuf::from("src").join("main.m")
        );
        assert_eq!(resolved.text.trim(), "x = 1;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_prefers_runmat_pcode_over_m_extension() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        let encoded = runmat_runtime::builtins::io::repl_fs::pcode::encode_pcode_source(
            "x = 2;",
            "src/main.m",
            runmat_runtime::builtins::io::repl_fs::pcode::PcodeAlgorithm::R2007b,
        );
        fs::write(tmp.path().join("src/main.p"), encoded).unwrap();
        let _cwd = push_cwd(tmp.path());

        let resolved = futures::executor::block_on(source_input_text(SourceInput::Path(
            "src/main".to_string(),
        )))
        .expect("path without extension should prefer .p over .m");

        assert_eq!(
            PathBuf::from(&resolved.display_name),
            PathBuf::from("src").join("main.p")
        );
        assert_eq!(resolved.text.trim(), "x = 2;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_prefers_runmat_pcode_over_explicit_m_path() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        let encoded = runmat_runtime::builtins::io::repl_fs::pcode::encode_pcode_source(
            "x = 3;",
            "src/main.m",
            runmat_runtime::builtins::io::repl_fs::pcode::PcodeAlgorithm::R2007b,
        );
        fs::write(tmp.path().join("src/main.p"), encoded).unwrap();
        let _cwd = push_cwd(tmp.path());

        let resolved = futures::executor::block_on(source_input_text(SourceInput::Path(
            "src/main.m".to_string(),
        )))
        .expect("explicit .m path should prefer sibling .p");

        assert_eq!(
            PathBuf::from(&resolved.display_name),
            PathBuf::from("src").join("main.p")
        );
        assert_eq!(resolved.text.trim(), "x = 3;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_manifest_entrypoint_prefers_runmat_pcode_over_m_path() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.m"), "x = 1;").unwrap();
        let encoded = runmat_runtime::builtins::io::repl_fs::pcode::encode_pcode_source(
            "x = 4;",
            "src/main.m",
            runmat_runtime::builtins::io::repl_fs::pcode::PcodeAlgorithm::R2007b,
        );
        fs::write(tmp.path().join("src/main.p"), encoded).unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[entrypoints.main]
path = "src/main"
"#,
        )
        .unwrap();
        let _cwd = push_cwd(tmp.path());

        let resolved =
            futures::executor::block_on(source_input_text(SourceInput::Path("main".to_string())))
                .expect("named entrypoint should resolve to P-code sibling");

        assert_eq!(
            PathBuf::from(&resolved.display_name),
            PathBuf::from("src").join("main.p")
        );
        assert_eq!(resolved.text.trim(), "x = 4;");
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_infers_m_extension_from_memory_provider() {
        let _guard = cwd_lock();
        let provider = runmat_filesystem::MemoryFsProvider::new();
        provider.write_project_path("/main.m", b"x = 1;").unwrap();

        runmat_filesystem::with_provider_override(Arc::new(provider), || {
            let resolved = futures::executor::block_on(source_input_text(SourceInput::Path(
                "main".to_string(),
            )))
            .expect("memory provider should resolve path without extension");

            assert_eq!(
                PathBuf::from(&resolved.display_name),
                PathBuf::from("main.m")
            );
            assert_eq!(resolved.text, "x = 1;");
        });
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn source_input_path_errors_for_invalid_named_entrypoint_target() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["src"]

[entrypoints.server]
module = "app.server"
function = "main"
"#,
        )
        .unwrap();
        let _cwd = push_cwd(tmp.path());
        let err =
            futures::executor::block_on(source_input_text(SourceInput::Path("server".to_string())))
                .expect_err("invalid module/function entrypoint should report resolve error");
        let RunError::Runtime(runtime_err) = err else {
            panic!("expected runtime error");
        };
        assert_eq!(
            runtime_err.identifier.as_deref(),
            Some("RunMat:EntrypointResolveFailed")
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn discover_known_project_symbols_reads_manifest_source_context() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("+stats")).unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["."]
"#,
        )
        .unwrap();
        fs::write(
            tmp.path().join("+stats/summarize.m"),
            "function y = summarize(x); y = x; end",
        )
        .unwrap();
        fs::write(tmp.path().join("main.m"), "x = 1;").unwrap();
        let _cwd = push_cwd(tmp.path());

        let symbols = discover_known_project_symbols(Some(
            tmp.path().join("main.m").to_string_lossy().as_ref(),
        ));
        assert!(
            symbols.contains("stats.summarize"),
            "source-context discovery should include project symbols for eval-hook lowering"
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn discover_known_project_symbols_includes_dependency_alias_qualified_names() {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().unwrap();
        let dep_root = tmp.path().join("deps/statslib");
        fs::create_dir_all(&dep_root).unwrap();
        fs::write(
            tmp.path().join("runmat.toml"),
            r#"
[package]
name = "demo"

[sources]
roots = ["."]

[dependencies]
statsdep = { path = "deps/statslib" }
"#,
        )
        .unwrap();
        fs::write(
            dep_root.join("runmat.toml"),
            r#"
[package]
name = "statslib"

[sources]
roots = ["."]
"#,
        )
        .unwrap();
        fs::write(
            dep_root.join("summarize.m"),
            "function y = summarize(x); y = x; end",
        )
        .unwrap();
        fs::write(tmp.path().join("main.m"), "x = 1;").unwrap();
        let _cwd = push_cwd(tmp.path());

        let symbols = discover_known_project_symbols(Some(
            tmp.path().join("main.m").to_string_lossy().as_ref(),
        ));
        assert!(
            symbols.contains("summarize"),
            "expected base dependency symbol in known-project discovery"
        );
        assert!(
            !symbols.contains("statslib.summarize"),
            "dependency package identity must not become a MATLAB namespace"
        );
        assert!(
            symbols.contains("statsdep.summarize"),
            "expected dependency-alias-qualified symbol in known-project discovery"
        );
    }
}
