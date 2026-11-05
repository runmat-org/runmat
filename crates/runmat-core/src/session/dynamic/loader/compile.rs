use super::super::super::*;
use std::path::Path;
use std::rc::Rc;

pub(super) struct DynamicCompileRequest<'a> {
    pub name: &'a str,
    pub path: &'a Path,
    pub path_origin: &'a runmat_runtime::builtins::common::path_search::FileCandidateOrigin,
    pub source_text: &'a str,
    pub project_context: Option<&'a super::super::super::compile::ProjectCompilationContext>,
    pub compat: CompatMode,
    pub top_level_await_enabled: bool,
    pub project_handoff: Option<&'a runmat_package::FrozenProjectHandoff>,
    pub phase: &'a runmat_runtime::user_functions::DynamicFunctionLoadPhase,
    /// A dependency is admitted for its class declaration, never invoked.
    pub dependency_only: bool,
}

pub(super) struct CompiledDynamicProduct {
    pub unit: crate::ExecutableUnit,
    pub unit_bytes: Arc<[u8]>,
    pub admitted: Rc<runmat_vm::AdmittedExecutableUnit>,
    pub source_records: Arc<[runmat_runtime::source_context::ProgramSourceRecord]>,
    pub selection: super::super::super::DynamicDependencyIdentity,
    pub dependencies: Arc<[super::super::super::DynamicFunctionDependency]>,
}

pub(super) async fn compile_dynamic_product(
    request: DynamicCompileRequest<'_>,
) -> Result<CompiledDynamicProduct, RuntimeError> {
    let DynamicCompileRequest {
        name,
        path,
        path_origin,
        source_text,
        project_context,
        compat,
        top_level_await_enabled,
        project_handoff,
        phase,
        dependency_only,
    } = request;
    let path_name = path.to_string_lossy();
    let source_catalog = project_context.and_then(|context| context.source_catalog.clone());
    let mut ast = parse_with_options(source_text, ParserOptions::new(compat)).map_err(|error| {
        build_runtime_error(format!(
            "Could not parse function source '{}': {error}",
            path.display()
        ))
        .with_identifier("RunMat:FunctionParseError")
        .build()
    })?;
    if matches!(
        phase,
        runmat_runtime::user_functions::DynamicFunctionLoadPhase::Handle { class: None, .. }
    ) {
        super::super::constructor::qualify_handle_source_classdef(name, &mut ast.body)?;
    } else {
        super::super::constructor::qualify_requested_source_classdef(phase, &mut ast.body)?;
    }
    let mut companion = if let Some(context) = project_context {
        super::super::super::compile::compose_project_compilation_context_async(context, compat)
            .await
            .map_err(|error| {
                build_runtime_error(format!(
                    "Could not compose function source '{}': {error}",
                    path.display()
                ))
                .with_identifier("RunMat:FunctionCompositionError")
                .with_source(error)
                .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
                .build()
            })?
    } else {
        super::super::super::compile::CompanionSourceDiscovery::default()
    };
    let primary_function_names =
        super::super::super::compile::source_context_function_names_in_statements(&ast.body)
            .into_iter()
            .collect::<HashSet<_>>();
    companion.apply_function_precedence(&primary_function_names);
    #[cfg(windows)]
    {
        companion.canonical_search_root =
            canonical_companion_search_root(path_origin, &companion).await?;
    }
    let primary_statement_count = ast.body.len();
    let executable_source =
        super::super::super::compile::program_sources::ProgramSourceTable::dynamic_root(
            path_name.as_ref(),
            source_text,
            project_handoff,
            path_origin,
        )
        .map_err(|error| {
            build_runtime_error(error)
                .with_identifier("RunMat:ExecutableSourceIdentity")
                .build()
        })?;
    let source_table =
        super::super::super::compile::program_sources::ProgramSourceTable::build_dynamic(
            &executable_source,
            path_name.as_ref(),
            &companion,
            primary_statement_count,
            project_handoff,
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
    let revision = super::super::dynamic_executable_revision(
        &executable_source,
        &source_table.source_map,
        compat,
    )?;
    debug!(
        cache = "miss",
        "compiling dynamically resolved function source"
    );
    let dynamic_compile_span = info_span!("runtime.dynamic_compile", source_kind = "m-file");
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
    // Companion definitions are compiled into this executable, but
    // their semantic diagnostics are not a load-time failure of the
    // selected function. The analysis product retains MIR contract
    // diagnostics; actual calls enforce their runtime contract.
    // Parsing, lowering, and bytecode compilation failures above
    // remain fatal, as in the canonical session compiler.
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
    let dependencies = super::super::dynamic_product_dependencies(&companion.observed_sources);
    let source_map = source_table.source_map;
    let source_records = Arc::<[runmat_runtime::source_context::ProgramSourceRecord]>::from(
        source_map.runtime_records(),
    );
    super::super::attach_dynamic_function_origins(
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
    let leaf_name = name.rsplit('.').next().unwrap_or(name);
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
    crate::ExecutableUnit::retain_vm_layout(&mut bytecode, &lowering.assembly, &mir).map_err(
        |error| {
            build_runtime_error(format!("failed to retain executable VM layout: {error:?}"))
                .with_identifier("RunMat:ExecutableLayout")
                .build()
        },
    )?;
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
    let envelope = match phase {
        runmat_runtime::user_functions::DynamicFunctionLoadPhase::RegisteredConstructorSource(
            class,
        )
        | runmat_runtime::user_functions::DynamicFunctionLoadPhase::Handle {
            class: Some(class),
            ..
        } => {
            let constructors = unit
                .mir()
                .functions
                .values()
                .filter(|metadata| {
                    metadata.source == primary_source
                        && metadata.class_method_owner.as_ref().is_some_and(|owner| {
                            owner.declaring_class == *class
                                && owner.role == runmat_types::ClassMethodRole::Constructor
                        })
                })
                .count();
            if dependency_only && constructors == 0 {
                unit.portable_envelope_for_declaration_source(class, primary_source)
            } else {
                unit.portable_envelope_for_constructor_source(class, primary_source)
            }
        }
        runmat_runtime::user_functions::DynamicFunctionLoadPhase::Handle {
            class: None, ..
        } => {
            let constructors = unit
                .mir()
                .functions
                .values()
                .filter(|metadata| metadata.source == primary_source)
                .filter_map(|metadata| metadata.class_method_owner.as_ref())
                .filter(|owner| {
                    owner.role == runmat_types::ClassMethodRole::Constructor && !owner.is_static
                })
                .map(|owner| owner.declaring_class.clone())
                .collect::<std::collections::BTreeSet<_>>();
            if constructors.len() == 1 {
                unit.portable_envelope_for_constructor_source(
                    constructors
                        .iter()
                        .next()
                        .expect("one checked constructor class"),
                    primary_source,
                )
            } else if constructors.is_empty() {
                unit.portable_envelope_for(Some(leaf_name))
            } else {
                Err("selected source has multiple constructor classes".into())
            }
        }
        _ => unit.portable_envelope_for(Some(leaf_name)),
    };
    let unit_bytes = envelope
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
    let selection = if let Some(class) = admitted.declaration_class() {
        super::super::super::DynamicDependencyIdentity::DeclarationOnly {
            class: class.clone(),
            program: admitted
                .envelope()
                .manifest
                .identity
                .program
                .identity_digest()
                .map_err(|error| {
                    build_runtime_error(error.to_string())
                        .with_identifier("RunMat:CallableArtifact:ExecutableAdmission")
                        .with_source(error)
                        .build()
                })?,
        }
    } else {
        super::super::super::DynamicDependencyIdentity::Callable(
            admitted.entrypoint_origin().map_err(|error| {
                build_runtime_error(error.to_string())
                    .with_identifier("RunMat:CallableArtifact:ExecutableAdmission")
                    .with_source(error)
                    .build()
            })?,
        )
    };
    let unit_bytes = Arc::<[u8]>::from(unit_bytes);
    let admitted = Rc::new(admitted);
    Ok(CompiledDynamicProduct {
        unit,
        unit_bytes,
        admitted,
        source_records,
        selection,
        dependencies,
    })
}

#[cfg(windows)]
async fn canonical_companion_search_root(
    origin: &runmat_runtime::builtins::common::path_search::FileCandidateOrigin,
    discovery: &super::super::super::compile::CompanionSourceDiscovery,
) -> Result<Option<std::path::PathBuf>, RuntimeError> {
    use runmat_runtime::builtins::common::path_search::FileCandidateOrigin;
    let FileCandidateOrigin::SearchRoot { access_root, .. } = origin else {
        return Ok(None);
    };
    let needs_canonical_root = discovery.statements.iter().any(|entry| {
        entry
            .source
            .as_ref()
            .and_then(|source| source.fullpath_name.as_deref())
            .is_some_and(|full| Path::new(full).strip_prefix(access_root).is_err())
    });
    if !needs_canonical_root {
        return Ok(None);
    }
    // Project discovery can issue verbatim Windows paths. Resolve the selected
    // root through its filesystem owner before comparing those captured paths.
    runmat_filesystem::canonicalize_async(access_root)
        .await
        .map(Some)
        .map_err(|error| {
            build_runtime_error(format!(
                "Could not resolve selected source root '{}': {error}",
                access_root.display()
            ))
            .with_identifier("RunMat:ExecutableSourceIdentity")
            .with_source(error)
            .with_gpu_gather_retry(runmat_runtime::GpuGatherRetry::Never)
            .build()
        })
}
