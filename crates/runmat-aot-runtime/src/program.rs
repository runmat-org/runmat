use std::{rc::Rc, sync::Arc};

use runmat_native_executor::execute::{NativeExecution, NativeWorkspaceInput};
use runmat_types::ProgramFunctionId;
use runmat_value::{Value, ValueSequence};

#[cfg(test)]
mod tests;
#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "program/s1_aot_direct_call_tests.rs"]
mod s1_aot_direct_call_tests;
mod bound;

pub async fn invoke(
    executor: Rc<runmat_native_executor::NativeExecutor>,
    program: &runmat_native_codegen::aot::AotProgramManifest,
    function: ProgramFunctionId,
    arguments: Vec<Value>,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<NativeExecution, runmat_runtime::RuntimeError> {
    let runtime = install_acceleration_services(runtime);
    let guards = ProgramInvocationGuards::install(Rc::clone(&executor), program, &runtime)
        .map_err(|error| *error)?;
    let active = runmat_runtime::user_functions::push_active_semantic_function(function.0 as usize);
    let execution = executor
        .invoke_async(function, arguments, requested_outputs, runtime)
        .await
        .map_err(native_executor_error);
    drop(active);
    drop(guards);
    execution
}

pub async fn invoke_workspace(
    executor: Rc<runmat_native_executor::NativeExecutor>,
    program: &runmat_native_codegen::aot::AotProgramManifest,
    function: ProgramFunctionId,
    workspace: NativeWorkspaceInput,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<NativeExecution, runmat_runtime::RuntimeError> {
    let runtime = install_acceleration_services(runtime);
    let guards = ProgramInvocationGuards::install(Rc::clone(&executor), program, &runtime)
        .map_err(|error| *error)?;
    let active = runmat_runtime::user_functions::push_active_semantic_function(function.0 as usize);
    let execution = executor
        .invoke_workspace_async(function, workspace, requested_outputs, runtime)
        .await
        .map_err(native_executor_error);
    drop(active);
    drop(guards);
    execution
}

struct ProgramInvocationGuards {
    _semantic: runmat_runtime::user_functions::FunctionInvokerGuard,
    _bound: runmat_runtime::user_functions::BoundFunctionInvokerGuard,
    _lexical: runmat_runtime::user_functions::LexicalFunctionInvokerGuard,
    _resolver: runmat_runtime::user_functions::FunctionResolverGuard,
    _name_candidates: runmat_runtime::user_functions::SemanticNameCandidatesGuard,
    _catalog: runmat_runtime::user_functions::SourceFunctionCatalogGuard,
    _execution_source: runmat_runtime::user_functions::ExecutionSourceResolverGuard,
}

impl ProgramInvocationGuards {
    fn install(
        executor: Rc<runmat_native_executor::NativeExecutor>,
        program: &runmat_native_codegen::aot::AotProgramManifest,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Result<Self, Box<runmat_runtime::RuntimeError>> {
        program
            .validate()
            .map_err(|error| Box::new(native_error(error)))?;
        let bound_functions = Arc::new(bound::AdmittedBoundFunctions::from_program(program)?);
        let mut functions = program
            .functions
            .iter()
            .map(|function| {
                let source_ordinal = usize::try_from(function.source.0)
                    .map_err(|_| native_error("program source ordinal exceeds host indexing"))?;
                let function_id = usize::try_from(function.function.0)
                    .map_err(|_| native_error("program function ordinal exceeds host indexing"))?;
                let source_id = runtime
                    .admitted_program_source_id(function.source)
                    .map_err(source_admission_error)?
                    .ok_or_else(|| {
                        source_admission_error(
                            runmat_runtime::context::ProgramSourceSelectionError::Missing(
                                function.source,
                            ),
                        )
                    })?;
                Ok(runmat_runtime::user_functions::SourceFunctionInfo {
                    source_id,
                    name: function.name.clone(),
                    function: function_id,
                    role: function.role,
                    source_file: program
                        .source_map
                        .entries
                        .get(source_ordinal)
                        .map(|source| source.relative_path.clone()),
                    binding_origin: Some(
                        runmat_execution::FunctionBindingOrigin::from_revision(
                            &program.executable.program,
                            function.function,
                        )
                        .map_err(native_error)?,
                    ),
                    definition_owners: None,
                })
            })
            .collect::<Result<Vec<_>, runmat_runtime::RuntimeError>>()?;
        functions.sort_by_key(|function| function.function);
        // Installation is synchronous, but it must target this program's
        // context state rather than the ambient compatibility slot. Native
        // invocation activates the same context around every callback.
        let _runtime_scope = runtime.enter();
        let previous_semantic = runmat_runtime::user_functions::current_semantic_function_invoker();
        let native_functions = Arc::new(
            program
                .functions
                .iter()
                .map(|function| function.function)
                .collect::<std::collections::BTreeSet<_>>(),
        );
        let semantic_executor = Rc::clone(&executor);
        let semantic_runtime = runtime.clone();
        let semantic_native_functions = native_functions.clone();
        let semantic = runmat_runtime::user_functions::install_local_semantic_function_invoker(
            Rc::new(move |function, arguments, requested_outputs| {
                let executor = Rc::clone(&semantic_executor);
                let runtime = isolated_runtime(semantic_runtime.clone());
                let arguments = arguments.to_vec();
                let previous = previous_semantic.clone();
                let native_functions = Arc::clone(&semantic_native_functions);
                Box::pin(async move {
                    let Ok(function) = u32::try_from(function).map(ProgramFunctionId) else {
                        return Err(native_error(
                            "semantic function identity exceeds native schema",
                        ));
                    };
                    if !native_functions.contains(&function) {
                        if let Some(previous) = previous {
                            return runtime
                                .scope(previous(function.0 as usize, &arguments, requested_outputs))
                                .await;
                        }
                        return Err(native_error(format!(
                            "semantic function {} is unavailable",
                            function.0
                        )));
                    }
                    invoke_native_values(&executor, function, arguments, requested_outputs, runtime)
                        .await
                })
            }),
        );

        let previous_bound = runmat_runtime::user_functions::current_bound_function_invoker();
        let bound_executor = Rc::clone(&executor);
        let bound = runmat_runtime::user_functions::install_local_bound_function_invoker(Rc::new(
            move |runtime, binding, arguments, requested_outputs| {
                let executor = Rc::clone(&bound_executor);
                let functions = Arc::clone(&bound_functions);
                let previous = previous_bound.clone();
                Box::pin(async move {
                    let Some(function) = functions.resolve(&binding).map_err(|error| *error)?
                    else {
                        if let Some(previous) = previous {
                            return previous(runtime, binding, arguments, requested_outputs).await;
                        }
                        return Err(bound::unavailable(
                            "bound function has no admitted execution owner",
                        ));
                    };
                    let runtime = isolated_runtime(runtime);
                    invoke_native_values(
                        &executor,
                        function,
                        arguments.values().to_vec(),
                        requested_outputs,
                        runtime,
                    )
                    .await
                })
            },
        ));

        let previous_lexical = runmat_runtime::user_functions::current_lexical_function_invoker();
        let lexical_executor = Rc::clone(&executor);
        let lexical_runtime = runtime.clone();
        let lexical_native_functions = native_functions;
        let lexical = runmat_runtime::user_functions::install_local_lexical_function_invoker(
            Rc::new(move |call| {
                let executor = Rc::clone(&lexical_executor);
                let runtime = isolated_runtime(lexical_runtime.clone());
                let previous = previous_lexical.clone();
                let native_functions = Arc::clone(&lexical_native_functions);
                Box::pin(async move {
                    let Ok(function) = u32::try_from(call.function).map(ProgramFunctionId) else {
                        return Err(native_error(
                            "lexical function identity exceeds native schema",
                        ));
                    };
                    if !native_functions.contains(&function) {
                        if let Some(previous) = previous {
                            return runtime.scope(previous(call)).await;
                        }
                        return Err(native_error(format!(
                            "lexical function {} is unavailable",
                            function.0
                        )));
                    }
                    let execution = executor
                        .invoke_async_with_captures(
                            function,
                            call.captures,
                            call.arguments,
                            call.requested_outputs,
                            runtime,
                        )
                        .await
                        .map_err(native_executor_error)?;
                    Ok(runmat_runtime::call::lexical::LexicalCallResult {
                        outputs: normalize_outputs(execution.outputs, call.requested_outputs)
                            .map_err(|error| *error)?,
                        captures: execution.captures,
                    })
                })
            }),
        );

        let previous_resolver =
            runmat_runtime::user_functions::current_semantic_function_resolver();
        let previous_candidates =
            runmat_runtime::user_functions::current_semantic_name_candidates();
        let resolver_program = Arc::new(program.clone());
        let source_facts = Arc::new(
            functions
                .iter()
                .map(|info| (info.function, info.clone()))
                .collect::<std::collections::HashMap<_, _>>(),
        );
        let candidates: Arc<runmat_runtime::user_functions::SemanticNameCandidates> =
            Arc::new(move |name| {
                let mut candidates = Vec::new();
                if let Some(function) = resolver_program.resolve_name(name) {
                    let function = function.0 as usize;
                    candidates.push(runmat_runtime::user_functions::SemanticNameCandidate {
                        function,
                        source: source_facts.get(&function).cloned(),
                        definition_owner: None,
                    });
                }
                if let Some(previous) = &previous_candidates {
                    for candidate in previous(name) {
                        if !candidates
                            .iter()
                            .any(|existing| existing.same_definition(&candidate))
                        {
                            candidates.push(candidate);
                        }
                    }
                }
                if let Some(previous) = previous_resolver
                    .as_ref()
                    .and_then(|resolver| resolver(name))
                {
                    if !candidates
                        .iter()
                        .any(|candidate| candidate.function == previous)
                    {
                        candidates.push(runmat_runtime::user_functions::SemanticNameCandidate {
                            function: previous,
                            source: None,
                            definition_owner: None,
                        });
                    }
                }
                candidates
            });
        let resolver_candidates = Arc::clone(&candidates);
        let resolver = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
            Arc::new(move |name| {
                resolver_candidates(name)
                    .into_iter()
                    .next()
                    .map(|candidate| candidate.function)
            }),
        ));
        let name_candidates =
            runmat_runtime::user_functions::install_semantic_name_candidates(Some(candidates));
        let catalog = runmat_runtime::user_functions::install_source_function_catalog(Some(
            Arc::new(functions),
        ));
        let execution_source =
            runmat_runtime::user_functions::install_execution_source_resolver(None);
        Ok(Self {
            _semantic: semantic,
            _bound: bound,
            _lexical: lexical,
            _resolver: resolver,
            _name_candidates: name_candidates,
            _catalog: catalog,
            _execution_source: execution_source,
        })
    }
}

fn isolated_runtime(
    runtime: runmat_runtime::context::RuntimeContext,
) -> runmat_runtime::context::RuntimeContext {
    let ports = runtime.service_ports().clone().without_workspace();
    install_acceleration_services(runtime.with_service_ports(ports))
}

async fn invoke_native_values(
    executor: &runmat_native_executor::NativeExecutor,
    function: ProgramFunctionId,
    arguments: Vec<Value>,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<ValueSequence, runmat_runtime::RuntimeError> {
    let (captures, arguments) = executor
        .split_flattened_capture_arguments(function, arguments)
        .map_err(native_executor_error)?;
    let execution = executor
        .invoke_async_with_captures(function, captures, arguments, requested_outputs, runtime)
        .await
        .map_err(native_executor_error)?;
    normalize_outputs(execution.outputs, requested_outputs).map_err(|error| *error)
}

/// A standalone root owns the same existing placement session used by Core.
/// Supplied services and all nested context clones retain their exact authority.
pub(crate) fn install_acceleration_services(
    runtime: runmat_runtime::context::RuntimeContext,
) -> runmat_runtime::context::RuntimeContext {
    let runtime = if runtime.service_ports().placement().is_none() {
        let ports = runtime.service_ports().clone().with_placement(Rc::new(
            runmat_accelerate::placement::PlacementSession::default(),
        ));
        runtime.with_service_ports(ports)
    } else {
        runtime
    };
    runmat_accelerate::native_auto::install_preparation_service(runtime)
}

fn normalize_outputs(
    outputs: Vec<Value>,
    requested_outputs: usize,
) -> Result<ValueSequence, Box<runmat_runtime::RuntimeError>> {
    if outputs.len() != requested_outputs {
        return Err(Box::new(native_error(format!(
            "native execution produced {} outputs for a request of {requested_outputs}",
            outputs.len()
        ))));
    }
    Ok(if requested_outputs == 1 {
        ValueSequence::single(
            outputs
                .into_iter()
                .next()
                .expect("one output was validated"),
        )
        .map_err(|error| Box::new(native_error(error)))?
    } else {
        ValueSequence::comma_separated(outputs).map_err(|error| Box::new(native_error(error)))?
    })
}

fn native_error(error: impl std::fmt::Display) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(error.to_string())
        .with_identifier("RunMat:AotNativeCall")
        .build()
}

fn native_executor_error(
    error: runmat_native_executor::NativeExecutorError,
) -> runmat_runtime::RuntimeError {
    match error {
        runmat_native_executor::NativeExecutorError::Runtime(original) => *original,
        runmat_native_executor::NativeExecutorError::SourceAdmission(error) => {
            source_admission_error(error)
        }
        other => native_error(other),
    }
}

fn source_admission_error(
    error: runmat_runtime::context::ProgramSourceSelectionError,
) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(error.to_string())
        .with_identifier("RunMat:AotSourceAdmission")
        .with_source(error)
        .build()
}
