use super::super::*;
use std::rc::Rc;
use tracing::Instrument;

pub(in crate::session) async fn load_dynamic_function(
    runtime: runmat_runtime::context::RuntimeContext,
    name: String,
    args: runmat_runtime::execution::RootedValues,
    requested_outputs: usize,
    phase: runmat_runtime::user_functions::DynamicFunctionLoadPhase,
    input_callsite: Option<runmat_runtime::callsite::CallsiteInfo>,
    environment: DynamicFunctionEnvironment,
) -> Option<Result<runmat_value::ValueSequence, RuntimeError>> {
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
    } = environment;
    #[cfg(not(target_arch = "wasm32"))]
    if phase == runmat_runtime::user_functions::DynamicFunctionLoadPhase::BeforeSemantic {
        if let Some(result) = mex_runtime
            .load_and_call(
                &name,
                args.values().to_vec(),
                requested_outputs,
                runtime.clone(),
            )
            .await
        {
            return Some(result);
        }
    }
    if phase == runmat_runtime::user_functions::DynamicFunctionLoadPhase::BeforeSemantic {
        return None;
    }
    let resolve_span = info_span!(
        "runtime.callable.resolve",
        call_kind = "runtime-name",
        source_kind = "m-file"
    );
    let source_search_span = info_span!("runtime.source_search", resolution_purpose = "call");
    let path_result = async {
        runmat_runtime::builtins::common::path_search::find_file_with_extensions_with_origin(
            &name,
            &[".m"],
            "function resolution",
        )
        .await
    }
    .instrument(source_search_span)
    .instrument(resolve_span)
    .await;
    let selected = match path_result {
        Ok(Some(selected)) => selected,
        Ok(None) => return None,
        Err(error) => {
            return Some(Err(build_runtime_error(error)
                .with_identifier("RunMat:FunctionResolution")
                .build()))
        }
    };
    let path = selected.path().to_path_buf();
    Some(
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
                .build()
            })?;
            let source_catalog = project_context
                .as_ref()
                .and_then(|context| context.source_catalog.clone());
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

            let (selected_runtime, binding, admission_guard) = if let Some(entry) = cached {
                debug!(
                    cache = "hit",
                    "reusing dynamically compiled function source"
                );
                if source_evaluator.retains_binding(&entry.binding)? {
                    (runtime, entry.binding, None)
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
                        &source_evaluator,
                        &selected_runtime,
                        entry.unit_bytes.as_ref(),
                        entry.entrypoint_origin,
                        bound,
                    )?;
                    let binding = admission.binding().clone();
                    cache.replace_binding(&path, binding.clone());
                    (selected_runtime, binding, Some(admission))
                }
            } else {
                // A failed revision probe recomposes the selected source set.
                // Loose private and companion files have no frozen project
                // revision, so primary text and path identity alone cannot
                // prove that a cached executable still represents the program.
                let mut ast = parse_with_options(&source_text, ParserOptions::new(compat))
                    .map_err(|error| {
                        build_runtime_error(format!(
                            "Could not parse function source '{}': {error}",
                            path.display()
                        ))
                        .with_identifier("RunMat:FunctionParseError")
                        .build()
                    })?;
                let mut companion = if let Some(context) = project_context.as_ref() {
                    super::super::compile::compose_project_compilation_context_async(
                        context, compat,
                    )
                    .await
                    .map_err(|error| {
                        build_runtime_error(format!(
                            "Could not compose function source '{}': {error}",
                            path.display()
                        ))
                        .with_identifier("RunMat:FunctionCompositionError")
                        .build()
                    })?
                } else {
                    super::super::compile::CompanionSourceDiscovery::default()
                };
                let primary_function_names =
                    super::super::compile::source_context_function_names_in_statements(&ast.body)
                        .into_iter()
                        .collect::<HashSet<_>>();
                companion.apply_function_precedence(&primary_function_names);
                let primary_statement_count = ast.body.len();
                let executable_source =
                    super::super::compile::program_sources::ProgramSourceTable::dynamic_root(
                        path_name.as_ref(),
                        &source_text,
                        project_handoff.as_ref(),
                        path_origin,
                    )
                    .map_err(|error| {
                        build_runtime_error(error)
                            .with_identifier("RunMat:ExecutableSourceIdentity")
                            .build()
                    })?;
                let source_table =
                    super::super::compile::program_sources::ProgramSourceTable::build_dynamic(
                        &executable_source,
                        path_name.as_ref(),
                        &companion,
                        primary_statement_count,
                        project_handoff.as_ref(),
                        path_origin,
                    )
                    .map_err(|error| {
                        build_runtime_error(error)
                            .with_identifier("RunMat:ExecutableSourceIdentity")
                            .build()
                    })?;
                if !companion.statements.is_empty() {
                    ast.body
                        .extend(companion.statements.drain(..).map(|entry| entry.statement));
                }
                let revision = super::dynamic_executable_revision(
                    &executable_source,
                    &source_table.source_map,
                    compat,
                )?;
                debug!(
                    cache = "miss",
                    "compiling dynamically resolved function source"
                );
                let dynamic_compile_span =
                    info_span!("runtime.dynamic_compile", source_kind = "m-file");
                let _dynamic_compile_guard = dynamic_compile_span.enter();
                let known_project_symbols = source_catalog
                    .as_ref()
                    .map(|catalog| &catalog.symbols)
                    .cloned()
                    .unwrap_or_default();
                let primary_source = source_table.primary_source();
                let mut frontend = runmat_static_analysis::frontend::analyze_program_with_catalog(
                    &ast,
                    &LoweringContext::new(&HashMap::new())
                        .with_known_project_symbols(&known_project_symbols)
                        .with_project_symbol_aliases(&companion.project_symbol_aliases)
                        .with_private_functions(
                            &companion.private_function_owners,
                            &companion.private_function_aliases,
                        )
                        .with_runmat_extensions_enabled(compat.allows_runmat_extensions())
                        .with_top_level_await_enabled(top_level_await_enabled)
                        .with_top_level_sources(&source_table.statement_sources)
                        .with_primary_source(primary_source),
                    source_catalog.as_ref(),
                );
                if let Some(error) = frontend.lowering_failure {
                    return Err(build_runtime_error(format!(
                        "Could not lower function source '{}': {error}",
                        path.display()
                    ))
                    .with_identifier(
                        error
                            .identifier
                            .as_deref()
                            .unwrap_or("RunMat:FunctionLoweringError"),
                    )
                    .build());
                }
                if let Some(error) = frontend.compile_failure {
                    return Err(error.into());
                }
                if frontend.has_errors() {
                    let diagnostic = frontend
                        .diagnostics
                        .iter()
                        .find(|diagnostic| {
                            diagnostic.severity == runmat_hir::HirDiagnosticSeverity::Error
                        })
                        .map(|diagnostic| diagnostic.message.clone())
                        .unwrap_or_else(|| "static analysis failed".to_string());
                    return Err(build_runtime_error(format!(
                        "Could not compile function source '{}': {diagnostic}",
                        path.display()
                    ))
                    .with_identifier("RunMat:FunctionCompileError")
                    .build());
                }
                let mut bytecode = frontend.bytecode.take().ok_or_else(|| {
                    build_runtime_error(format!(
                        "Canonical frontend produced no bytecode for '{}'",
                        path.display()
                    ))
                    .with_identifier("RunMat:FunctionCompileError")
                    .build()
                })?;
                let lowering = frontend.lowering.as_ref().ok_or_else(|| {
                    build_runtime_error(format!(
                        "Canonical frontend produced no source lowering for '{}'",
                        path.display()
                    ))
                    .with_identifier("RunMat:FunctionCompileError")
                    .build()
                })?;
                source_table
                    .validate_compiled_sources(&bytecode, &lowering.assembly)
                    .map_err(|error| {
                        build_runtime_error(error)
                            .with_identifier("RunMat:ExecutableSourceIdentity")
                            .build()
                    })?;
                let dependencies = super::dynamic_product_dependencies(&companion.observed_sources);
                let source_map = source_table.source_map;
                let source_records =
                    Arc::<[runmat_runtime::source_context::ProgramSourceRecord]>::from(
                        source_map.runtime_records(),
                    );
                super::attach_dynamic_function_origins(
                    &mut bytecode,
                    &revision.program_revision,
                    &source_map,
                )
                .map_err(|error| {
                    build_runtime_error(error.to_string())
                        .with_identifier("RunMat:ExecutableFunctionIdentity")
                        .with_source(error)
                        .build()
                })?;
                let leaf_name = name.rsplit('.').next().unwrap_or(name.as_str());
                bytecode.bound_functions = bytecode.function_registry.functions.clone();
                let mir = frontend.mir.take().ok_or_else(|| {
                    build_runtime_error(format!(
                        "Canonical frontend produced no MIR for '{}'",
                        path.display()
                    ))
                    .with_identifier("RunMat:FunctionCompileError")
                    .build()
                })?;
                let analysis = frontend.facts.take().ok_or_else(|| {
                    build_runtime_error(format!(
                        "Canonical frontend produced no analysis for '{}'",
                        path.display()
                    ))
                    .with_identifier("RunMat:FunctionCompileError")
                    .build()
                })?;
                crate::ExecutableUnit::retain_vm_layout(&mut bytecode, &lowering.assembly, &mir)
                    .map_err(|error| {
                        build_runtime_error(format!(
                            "failed to retain executable VM layout: {error:?}"
                        ))
                        .with_identifier("RunMat:ExecutableLayout")
                        .build()
                    })?;
                let unit = crate::ExecutableUnit::new(
                    executable_source,
                    revision,
                    source_map,
                    mir,
                    analysis,
                    bytecode,
                )
                .map_err(|error| {
                    build_runtime_error(error)
                        .with_identifier("RunMat:ExecutableProduct")
                        .build()
                })?;
                let unit_bytes = unit
                    .portable_envelope_for(Some(leaf_name))
                    .and_then(|envelope| {
                        envelope
                            .canonical_bytes()
                            .map_err(|error| error.to_string())
                    })
                    .map_err(|error| {
                        build_runtime_error(error)
                            .with_identifier("RunMat:CallableArtifact:ExecutableProduct")
                            .build()
                    })?;
                let admitted = runmat_vm::admit_executable_unit(&unit_bytes).map_err(|error| {
                    build_runtime_error(error.to_string())
                        .with_identifier("RunMat:CallableArtifact:ExecutableAdmission")
                        .with_source(error)
                        .build()
                })?;
                let entrypoint_origin = admitted.entrypoint_origin().map_err(|error| {
                    build_runtime_error(error.to_string())
                        .with_identifier("RunMat:CallableArtifact:ExecutableAdmission")
                        .with_source(error)
                        .build()
                })?;
                let unit_bytes = Arc::<[u8]>::from(unit_bytes);
                let admitted = Rc::new(admitted);
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
                let selected_runtime = bound.runtime().clone();
                let admission = super::super::source_evaluation::admit_dynamic_callable(
                    &source_evaluator,
                    &selected_runtime,
                    unit_bytes.as_ref(),
                    entrypoint_origin,
                    bound,
                )?;
                let binding = admission.binding().clone();
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
                        dependencies,
                    },
                );
                (selected_runtime, binding, Some(admission))
            };
            let result =
                runmat_runtime::callsite::with_function_input_callsite(input_callsite, async {
                    runmat_runtime::user_functions::try_call_bound_function_in_context(
                        &selected_runtime,
                        binding,
                        args.values(),
                        requested_outputs,
                    )
                    .await
                    .ok_or_else(|| {
                        build_runtime_error("selected runtime has no exact bound invoker")
                            .with_identifier("RunMat:BoundFunction:UnavailableInvoker")
                            .build()
                    })?
                })
                .await;
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

#[derive(Clone)]
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
