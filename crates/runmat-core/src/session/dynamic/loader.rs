use super::super::*;
use std::rc::Rc;
use tracing::Instrument;

#[path = "loader/compile.rs"]
mod compile;
#[path = "loader/dependencies.rs"]
mod dependencies;
use dependencies::selected_constructor_dependencies;

pub(in crate::session) async fn load_dynamic_function(
    runtime: runmat_runtime::context::RuntimeContext,
    name: String,
    args: runmat_runtime::call::RootedFunctionArguments,
    requested_outputs: usize,
    phase: runmat_runtime::user_functions::DynamicFunctionLoadPhase,
    input_callsite: Option<runmat_runtime::callsite::CallsiteInfo>,
    environment: Rc<DynamicFunctionEnvironment>,
) -> runmat_runtime::user_functions::DynamicFunctionLoadAttempt {
    use std::ops::ControlFlow;
    #[cfg(target_arch = "wasm32")]
    let _ = &runtime;
    let DynamicFunctionEnvironment {
        compat,
        top_level_await_enabled,
        cache,
        project_handoff,
        source_evaluator,
        #[cfg(not(target_arch = "wasm32"))]
        mex_runtime,
    } = environment.as_ref();
    let compat = *compat;
    let top_level_await_enabled = *top_level_await_enabled;
    #[cfg(not(target_arch = "wasm32"))]
    if phase == runmat_runtime::user_functions::DynamicFunctionLoadPhase::BeforeSemantic {
        use runmat_runtime::user_functions::NativeResolutionScope;
        return match runtime.native_resolution_scope() {
            NativeResolutionScope::Selected(route)
            | NativeResolutionScope::SelectedOrdinary { route, .. } => {
                runmat_runtime::user_functions::await_dynamic_function_load(route.invoke(
                    runtime.clone(),
                    name,
                    args,
                    requested_outputs,
                ))
                .await
            }
            NativeResolutionScope::SelectedClosed => ControlFlow::Continue(args),
            NativeResolutionScope::Session => {
                mex_runtime
                    .load_and_call(&name, args, requested_outputs, runtime)
                    .await
            }
        };
    }
    if phase == runmat_runtime::user_functions::DynamicFunctionLoadPhase::BeforeSemantic {
        return ControlFlow::Continue(args);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let ordinary_mex = match runtime.native_resolution_scope() {
        runmat_runtime::user_functions::NativeResolutionScope::SelectedOrdinary {
            mex,
            source_owner,
            ..
        } if source_owner.is_bound_in(&runtime) => Some(Rc::clone(mex)),
        runmat_runtime::user_functions::NativeResolutionScope::SelectedOrdinary { .. } => {
            return ControlFlow::Break(Err(build_runtime_error(
                "ordinary MEX source authority differs from the current source binding",
            )
            .with_identifier("RunMat:MEX:SourceAuthority")
            .build()));
        }
        _ => None,
    };
    let resolve_span = info_span!(
        "runtime.callable.resolve",
        call_kind = "runtime-name",
        source_kind = "m-file"
    );
    let source_search_span = info_span!("runtime.source_search", resolution_purpose = "call");
    let path_result = if let runmat_runtime::user_functions::DynamicFunctionLoadPhase::Handle {
        selected_source: Some(selected),
        ..
    } = &phase
    {
        Ok(Some(selected.clone()))
    } else {
        runtime
            .scope(async {
                runmat_runtime::builtins::common::path_search::find_file_with_extensions_with_origin(
                    &name,
                    &[".m"],
                    "function resolution",
                )
                .await
            })
            .instrument(source_search_span)
            .instrument(resolve_span)
            .await
    };
    let selected = match path_result {
        Ok(Some(selected)) => selected,
        Ok(None) => return ControlFlow::Continue(args),
        Err(error) => {
            return ControlFlow::Break(Err(error.into_identified_runtime_error(
                "function resolution",
                "RunMat:FunctionResolution",
            )))
        }
    };
    let path = selected.path().to_path_buf();
    ControlFlow::Break(
        async move {
            let path_origin = selected.origin();
            let source_text =
                runmat_runtime::builtins::io::repl_fs::pcode::read_source_text_async(&path)
                    .await
                    .map_err(|error| {
                        build_runtime_error(format!(
                            "Could not read function source '{}': {error}",
                            path.display()
                        ))
                        .with_identifier("RunMat:FunctionSourceRead")
                        .with_source(error)
                        .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                        .build()
                    })?;
            let path_name = path.to_string_lossy();
            let project_context = if let Some(handoff) = project_handoff.as_ref() {
                Some(
                    super::super::compile::project_compilation_context_from_handoff(
                        path_name.as_ref(),
                        handoff,
                    ),
                )
                .transpose()
            } else {
                super::super::compile::discover_project_compilation_context_async(
                    path_name.as_ref(),
                )
                .await
            }
            .map_err(|error| {
                build_runtime_error(format!(
                    "Could not construct project context for '{}': {error}",
                    path.display()
                ))
                .with_identifier("RunMat:ProjectComposition")
                .with_source(error)
                .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                .build()
            })?;
            let inventory_revision = project_context
                .as_ref()
                .map(|context| context.dynamic_inventory_revision())
                .unwrap_or_else(|| runmat_execution::Digest::sha256("empty-source-inventory"));
            let primary_revision = runmat_execution::Digest::sha256(source_text.as_bytes());
            let cached_candidate = cache.get(&path).filter(|entry| {
                entry.selection.matches_selection(
                    &primary_revision,
                    &inventory_revision,
                    path_origin,
                )
            });
            let cached = if let Some(entry) = cached_candidate {
                super::reusable_dynamic_product(&entry)
                    .await
                    .then_some(entry)
            } else {
                None
            };

            let invocation = if let Some(entry) = cached {
                let constructor_owner = super::constructor::retained_constructor_owner(
                    entry.admitted.as_ref(),
                    entry.entrypoint_origin,
                )?;
                super::constructor::require_requested_source_constructor(
                    &phase,
                    constructor_owner.as_ref(),
                )?;
                let (super_dependencies, super_dependency_origins, metadata_units) = if let Some(owner) = constructor_owner.as_ref() {
                    runtime.scope(selected_constructor_dependencies(
                        cache, entry.admitted.as_ref(), &runtime, owner,
                        compat, top_level_await_enabled, project_handoff.as_ref(),
                    )).await?
                } else {
                    (Vec::new(), HashSet::new(), Vec::new())
                };
                debug!(
                    cache = "hit",
                    "reusing dynamically compiled function source"
                );
                if entry.super_dependency_origins == super_dependency_origins
                    && source_evaluator.retains_binding(&entry.binding)?
                {
                    let selected_runtime = runtime
                        .with_retained_super_constructors(
                            entry.super_constructors.clone(),
                            entry.entrypoint_origin,
                        )?;
                    (
                        selected_runtime,
                        entry.binding,
                        None,
                        constructor_owner,
                        entry.admitted,
                        metadata_units,
                    )
                } else {
                    let bound = entry
                        .admitted
                        .as_ref()
                        .clone()
                        .bind_runtime_sources_with_records(runtime, entry.source_records.to_vec())
                        .map_err(|error| {
                            build_runtime_error(error.to_string())
                                .with_identifier("RunMat:ExecutableSourceIdentity")
                                .with_source(error)
                                .build()
                        })?;
                    let selected_runtime = bound.runtime().clone();
                    let admission = super::super::source_evaluation::admit_dynamic_callable(
                        source_evaluator,
                        &selected_runtime,
                        entry.unit_bytes.as_ref(),
                        entry.entrypoint_origin,
                        bound,
                        super_dependencies,
                    )?;
                    let binding = admission.binding().clone();
                    let super_constructors = admission.super_constructors().to_vec();
                    let selected_runtime = selected_runtime
                        .with_retained_super_constructors(
                            super_constructors.clone(),
                            entry.entrypoint_origin,
                        )?;
                    cache.replace_binding(
                        &path,
                        binding.clone(),
                        super_constructors,
                        super_dependency_origins,
                    );
                    (
                        selected_runtime,
                        binding,
                        Some(admission),
                        constructor_owner,
                        entry.admitted,
                        metadata_units,
                    )
                }
            } else {
                // A failed revision probe recomposes the selected source set.
                // Loose private and companion files have no frozen project
                // revision, so primary text and path identity alone cannot
                // prove that a cached executable still represents the program.
                let compile::CompiledDynamicProduct {
                    unit, unit_bytes, admitted, source_records, selection, dependencies,
                } = compile::compile_dynamic_product(compile::DynamicCompileRequest {
                    name: &name,
                    path: &path,
                    path_origin,
                    source_text: &source_text,
                    project_context: project_context.as_ref(),
                    compat,
                    top_level_await_enabled,
                    project_handoff: project_handoff.as_ref(),
                    phase: &phase,
                    dependency_only: false,
                }).await?;
                let entrypoint_origin = match selection {
                    super::super::DynamicDependencyIdentity::Callable(origin) => origin,
                    super::super::DynamicDependencyIdentity::DeclarationOnly { .. } => {
                        return Err(build_runtime_error(
                            "declaration-only product cannot be invoked as a function source",
                        )
                        .with_identifier("RunMat:CallableArtifact:ExecutableAdmission")
                        .build());
                    }
                };
                let bound = admitted
                    .as_ref()
                    .clone()
                    .bind_runtime_sources_with_records(runtime, source_records.to_vec())
                    .map_err(|error| {
                        build_runtime_error(error.to_string())
                            .with_identifier("RunMat:ExecutableSourceIdentity")
                            .with_source(error)
                            .build()
                    })?;
                let constructor_owner = super::constructor::admitted_constructor_owner(
                    &unit,
                    &bound,
                    entrypoint_origin,
                )?;
                super::constructor::require_requested_source_constructor(
                    &phase,
                    constructor_owner.as_ref(),
                )?;
                let selected_runtime = bound.runtime().clone();
                let (super_dependencies, super_dependency_origins, metadata_units) = if let Some(owner) = constructor_owner.as_ref() {
                    selected_runtime.scope(selected_constructor_dependencies(
                        cache, admitted.as_ref(), &selected_runtime, owner,
                        compat, top_level_await_enabled, project_handoff.as_ref(),
                    )).await?
                } else {
                    (Vec::new(), HashSet::new(), Vec::new())
                };
                let admission = super::super::source_evaluation::admit_dynamic_callable(
                    source_evaluator,
                    &selected_runtime,
                    unit_bytes.as_ref(),
                    entrypoint_origin,
                    bound,
                    super_dependencies,
                )?;
                let binding = admission.binding().clone();
                let super_constructors = admission.super_constructors().to_vec();
                let selected_runtime = selected_runtime
                    .with_retained_super_constructors(
                        super_constructors.clone(),
                        entrypoint_origin,
                    )?;
                let publish_span =
                    info_span!("runtime.function_registry.publish", source_kind = "m-file");
                let _publish_guard = publish_span.enter();
                cache.insert(
                    path.clone(),
                    DynamicFunctionCacheEntry {
                        selection: DynamicFunctionCacheSelection {
                            primary_revision,
                            inventory_revision,
                            path_origin: (*path_origin).clone(),
                        },
                        unit_bytes: Arc::clone(&unit_bytes),
                        admitted: Rc::clone(&admitted),
                        source_records,
                        entrypoint_origin,
                        binding: binding.clone(),
                        super_constructors,
                        super_dependency_origins,
                        dependencies,
                    },
                );
                (
                    selected_runtime,
                    binding,
                    Some(admission),
                    constructor_owner,
                    admitted,
                    metadata_units,
                )
            };
            let (
                selected_runtime,
                binding,
                admission_guard,
                constructor_owner,
                admitted_unit,
                metadata_units,
            ) = invocation;
            #[cfg(not(target_arch = "wasm32"))]
            let selected_runtime = if let Some(mex) = ordinary_mex {
                let selected = source_evaluator
                    .selected_binding_view(&binding)?
                    .ok_or_else(|| {
                        build_runtime_error("dynamic function lacks its exact selected body")
                            .with_identifier("RunMat:MEX:SourceAuthority")
                            .build()
                    })?;
                let lease = binding.authority.local_lease(binding.function).ok_or_else(|| {
                    build_runtime_error("dynamic function lacks its exact definition lease")
                        .with_identifier("RunMat:MEX:SourceAuthority")
                        .build()
                })?;
                let body = selected
                    .get_owned(runmat_hir::FunctionId(binding.function), lease)
                    .ok_or_else(|| {
                        build_runtime_error("dynamic function lost its exact selected body")
                            .with_identifier("RunMat:MEX:SourceAuthority")
                            .build()
                    })?;
                let source_owner = body.selected_source_owner.as_ref().ok_or_else(|| {
                    build_runtime_error("dynamic function lacks an issued source owner")
                        .with_identifier("RunMat:MEX:SourceAuthority")
                        .build()
                })?;
                if !source_owner.is_issued_in(&selected_runtime)
                    || body
                        .source_id
                        .and_then(runmat_vm::BytecodeSourceRef::runtime)
                        .is_none_or(|source| !source_owner.owns_source(source))
                {
                    return Err(build_runtime_error(
                        "dynamic function differs from its exact issued source",
                    )
                    .with_identifier("RunMat:MEX:SourceAuthority")
                    .build());
                }
                // Cached definitions retain their exact child source table
                // but the caller's context still carries its parent binding.
                // Rebind that table before issuing any child native route.
                let selected_runtime = source_owner
                    .bind_runtime(selected_runtime)
                    .map_err(|error| {
                        build_runtime_error(error.to_string())
                            .with_identifier("RunMat:MEX:SourceAuthority")
                            .build()
                    })?;
                selected_runtime.with_ordinary_selected_native_route(
                    super::super::source_evaluation::ordinary_native_route(
                        None,
                        Rc::clone(&mex),
                        source_owner.clone(),
                    ),
                    mex,
                    source_owner.clone(),
                )
            } else {
                selected_runtime
            };
            if matches!(phase, runmat_runtime::user_functions::DynamicFunctionLoadPhase::Handle { .. }) {
                let function = runmat_hir::FunctionId(binding.function);
                let selected = source_evaluator
                    .selected_binding_view(&binding)?
                    .ok_or_else(|| {
                        build_runtime_error("selected handle lacks its exact admitted definition")
                            .with_identifier("RunMat:SelectedHandle:MissingDefinition")
                            .build()
                    })?;
                let lease = binding.authority.local_lease(binding.function).ok_or_else(|| {
                    build_runtime_error("selected handle lacks its exact definition lease")
                        .with_identifier("RunMat:SelectedHandle:MissingDefinition")
                        .build()
                })?;
                let body = selected.get_owned(function, lease).ok_or_else(|| {
                    build_runtime_error("selected handle lacks its exact admitted definition")
                        .with_identifier("RunMat:SelectedHandle:MissingDefinition")
                        .build()
                })?;
                let missing_source = || {
                    build_runtime_error("selected handle lacks its exact issued source")
                        .with_identifier("RunMat:SelectedHandle:MissingSource")
                        .build()
                };
                let source_owner = body.selected_source_owner.as_ref().ok_or_else(missing_source)?;
                if !source_owner.is_live() {
                    return Err(missing_source());
                }
                let ordinal = body
                    .source_id
                    .and_then(runmat_vm::BytecodeSourceRef::runtime)
                    .and_then(|source| source_owner.source_ordinal_for(source))
                    .ok_or_else(missing_source)?;
                let issued = source_owner
                    .issued_source_info_for(ordinal)
                    .ok_or_else(missing_source)?;
                let source_file = issued
                    .fullpath_name
                    .as_ref()
                    .map(|path| path.to_string())
                    .or_else(|| body.source_file.clone());
                let handle = Value::bound_function_handle(
                    name,
                    binding.function,
                    binding.authority,
                    body.source_role,
                    source_file,
                );
                let sequence = runmat_value::ValueSequence::single(handle).map_err(|error| {
                    build_runtime_error(error.to_string())
                        .with_identifier("RunMat:SelectedHandle:InvalidOutput")
                        .build()
                })?;
                let rooted = runmat_runtime::execution::RootedValueSequence::new(sequence)
                    .map_err(|error| error.into_runtime_error())?;
                return super::finish_dynamic_invocation(Ok(rooted), admission_guard, |error| {
                    runmat_runtime::execution::preserve_source_call_stack_in_context(
                        error,
                        &selected_runtime,
                    );
                });
            }
            #[cfg(test)]
            let constructor_trace = std::env::var("RUNMAT_CONSTRUCTOR_TRACE").as_deref() == Ok("1")
                && constructor_owner.is_some();
            #[cfg(test)]
            if constructor_trace {
                let entrypoint = admitted_unit.entrypoint_function();
                let body = entrypoint.and_then(|function| admitted_unit.registry().get(function));
                eprintln!(
                    "[constructor-source] selected={:?} entrypoint={:?} body={:?} before_globals={:?}",
                    constructor_owner,
                    entrypoint,
                    body.map(|body| (
                        &body.display_name,
                        &body.class_method_owner,
                        &body.binding_origin,
                        &body.instructions,
                    )),
                    selected_runtime.session_global_names(),
                );
            }
            let result =
                runmat_runtime::callsite::with_function_input_callsite(input_callsite, async {
                    selected_runtime
                        .clone()
                        .scope(async {
                            let mut selected_units = Vec::with_capacity(metadata_units.len() + 1);
                            selected_units.extend(metadata_units.iter().map(|unit| unit.as_ref()));
                            selected_units.push(admitted_unit.as_ref());
                            if let Some(owner) = constructor_owner {
                                runmat_vm::prepare_admitted_execution_metadata_for_class_batch(
                                    &selected_units, &owner.declaring_class,
                                )?;
                                runmat_runtime::dispatcher::call_rooted_bound_constructor_sequence_in_context(
                                    &selected_runtime,
                                    owner,
                                    binding,
                                    args,
                                    requested_outputs,
                                )
                                .await
                            } else {
                                runmat_runtime::user_functions::try_call_rooted_bound_function_with_environment_in_context(
                                    &selected_runtime,
                                    binding,
                                    args,
                                    requested_outputs,
                                    None,
                                )
                                .await
                                .ok_or_else(|| {
                                    build_runtime_error("selected runtime has no exact bound invoker")
                                        .with_identifier("RunMat:BoundFunction:UnavailableInvoker")
                                        .build()
                                })?
                            }
                        })
                        .await
                })
                .await
                .and_then(std::convert::identity);
            #[cfg(test)]
            if constructor_trace {
                let names = selected_runtime.session_global_names();
                eprintln!("[constructor-source] after_globals={names:?}");
                if let Ok(names) = names {
                    for name in names {
                        eprintln!(
                            "[constructor-source] global {name}={:?}",
                            selected_runtime.session_global_value(&name)
                        );
                    }
                }
            }
            super::finish_dynamic_invocation(result, admission_guard, |error| {
                runmat_runtime::execution::preserve_source_call_stack_in_context(
                    error,
                    &selected_runtime,
                );
            })
        }
        .await,
    )
}

pub(in crate::session) struct DynamicFunctionEnvironment {
    pub(in crate::session) compat: CompatMode,
    pub(in crate::session) top_level_await_enabled: bool,
    pub(in crate::session) cache: Rc<super::super::dynamic_cache::DynamicFunctionCache>,
    pub(in crate::session) project_handoff: Option<runmat_package::FrozenProjectHandoff>,
    pub(in crate::session) source_evaluator:
        Rc<super::super::source_evaluation::SessionSourceEvaluator>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(in crate::session) mex_runtime: std::rc::Rc<runmat_runtime::foreign::MexRuntimeSession>,
}
