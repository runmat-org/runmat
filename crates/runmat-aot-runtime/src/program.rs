use std::{rc::Rc, sync::Arc};

use runmat_native_executor::execute::{NativeExecution, NativeWorkspaceInput};
use runmat_types::ProgramFunctionId;
use runmat_value::Value;

pub async fn invoke(
    executor: Rc<runmat_native_executor::NativeExecutor>,
    program: &runmat_native_codegen::aot::AotProgramManifest,
    function: ProgramFunctionId,
    arguments: Vec<Value>,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<NativeExecution, runmat_runtime::RuntimeError> {
    let guards = ProgramInvocationGuards::install(Rc::clone(&executor), program, &runtime);
    let active = runmat_runtime::user_functions::push_active_semantic_function(function.0 as usize);
    let execution = executor
        .invoke_async(function, arguments, requested_outputs, runtime)
        .await
        .map_err(native_error);
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
    let guards = ProgramInvocationGuards::install(Rc::clone(&executor), program, &runtime);
    let active = runmat_runtime::user_functions::push_active_semantic_function(function.0 as usize);
    let execution = executor
        .invoke_workspace_async(function, workspace, requested_outputs, runtime)
        .await
        .map_err(native_error);
    drop(active);
    drop(guards);
    execution
}

struct ProgramInvocationGuards {
    _semantic: runmat_runtime::user_functions::FunctionInvokerGuard,
    _lexical: runmat_runtime::user_functions::LexicalFunctionInvokerGuard,
    _resolver: runmat_runtime::user_functions::FunctionResolverGuard,
    _catalog: runmat_runtime::user_functions::SourceFunctionCatalogGuard,
}

impl ProgramInvocationGuards {
    fn install(
        executor: Rc<runmat_native_executor::NativeExecutor>,
        program: &runmat_native_codegen::aot::AotProgramManifest,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> Self {
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
                    let execution = executor
                        .invoke_async(function, arguments, requested_outputs, runtime)
                        .await
                        .map_err(native_error)?;
                    normalize_outputs(execution.outputs, requested_outputs).map_err(|error| *error)
                })
            }),
        );

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
                        .map_err(native_error)?;
                    Ok(runmat_runtime::call::lexical::LexicalCallResult {
                        value: normalize_outputs(execution.outputs, call.requested_outputs)
                            .map_err(|error| *error)?,
                        captures: execution.captures,
                    })
                })
            }),
        );

        let previous_resolver =
            runmat_runtime::user_functions::current_semantic_function_resolver();
        let resolver_program = Arc::new(program.clone());
        let resolver = runmat_runtime::user_functions::install_semantic_function_resolver(Some(
            Arc::new(move |name| {
                resolver_program
                    .resolve_name(name)
                    .map(|function| function.0 as usize)
                    .or_else(|| {
                        previous_resolver
                            .as_ref()
                            .and_then(|resolver| resolver(name))
                    })
            }),
        ));
        let mut functions = program
            .functions
            .iter()
            .map(
                |function| runmat_runtime::user_functions::SourceFunctionInfo {
                    source_id: runmat_types::SourceId(function.source.0 as usize),
                    name: function.name.clone(),
                    function: function.function.0 as usize,
                },
            )
            .collect::<Vec<_>>();
        functions.sort_by_key(|function| function.function);
        let catalog = runmat_runtime::user_functions::install_source_function_catalog(Some(
            Arc::new(functions),
        ));
        Self {
            _semantic: semantic,
            _lexical: lexical,
            _resolver: resolver,
            _catalog: catalog,
        }
    }
}

fn isolated_runtime(
    runtime: runmat_runtime::context::RuntimeContext,
) -> runmat_runtime::context::RuntimeContext {
    let ports = runtime.service_ports().clone().without_workspace();
    runtime.with_service_ports(ports)
}

fn normalize_outputs(
    mut outputs: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, Box<runmat_runtime::RuntimeError>> {
    if outputs.len() != requested_outputs {
        return Err(Box::new(native_error(format!(
            "native execution produced {} outputs for a request of {requested_outputs}",
            outputs.len()
        ))));
    }
    Ok(match requested_outputs {
        0 => Value::OutputList(Vec::new()),
        1 => outputs.remove(0),
        _ => Value::OutputList(outputs),
    })
}

fn native_error(error: impl std::fmt::Display) -> runmat_runtime::RuntimeError {
    runmat_runtime::build_runtime_error(error.to_string())
        .with_identifier("RunMat:AotNativeCall")
        .build()
}
