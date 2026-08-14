use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    sync::Arc,
};

use runmat_runtime::user_functions;
use runmat_types::ProgramFunctionId;
use runmat_value::Value;

use crate::ExecutableUnit;

pub(crate) struct NativeExecution {
    pub value: Value,
    pub loop_backedges: BTreeMap<runmat_types::ProgramPointId, u64>,
}

// Semantic invokers use Runtime's established Arc callback ABI, while the
// executable-memory owner, RuntimeContext, and GC Values are deliberately
// invocation-thread confined. This callback never crosses that thread.
#[allow(clippy::arc_with_non_send_sync)]
pub(crate) async fn invoke(
    unit: &ExecutableUnit,
    published: runmat_jit::entry::PublishedEntry,
    preferred_function: Option<&str>,
    arguments: Vec<Value>,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<NativeExecution, runmat_runtime::RuntimeError> {
    runmat_vm::prepare_native_execution_metadata(unit.bytecode())?;
    let function = preferred_function
        .map(|name| {
            unit.native_function_id(name)
                .ok_or_else(|| super::error::stage("NativeProcedure", name))
                .and_then(program_function)
        })
        .transpose()?
        .unwrap_or(published.entrypoint);
    let registry = Arc::new(unit.functions().clone());
    let native_functions = unit
        .mir()
        .bodies
        .keys()
        .map(|function| program_function(*function))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let capture_bindings = unit
        .mir()
        .functions
        .iter()
        .map(|(function, metadata)| {
            Ok((
                program_function(*function)?,
                metadata
                    .captures
                    .iter()
                    .map(|capture| capture.binding)
                    .collect::<Vec<_>>(),
            ))
        })
        .collect::<Result<BTreeMap<_, _>, runmat_runtime::RuntimeError>>()?;

    let previous_invoker = user_functions::current_semantic_function_invoker();
    let nested_executor = Rc::clone(&published.executor);
    let nested_runtime = runtime.clone();
    let semantic_capture_bindings = Arc::new(capture_bindings.clone());
    let semantic_native_functions = native_functions.clone();
    let semantic_unit = unit.clone();
    let invoker =
        user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, arguments, requested_outputs| {
                let arguments = arguments.to_vec();
                let previous_invoker = previous_invoker.clone();
                let executor = Rc::clone(&nested_executor);
                let runtime = nested_runtime.clone();
                let capture_bindings = Arc::clone(&semantic_capture_bindings);
                let unit = semantic_unit.clone();
                let is_native = u32::try_from(function)
                    .map(ProgramFunctionId)
                    .is_ok_and(|function| semantic_native_functions.contains(&function));
                Box::pin(async move {
                    if is_native {
                        let function = program_function(runmat_hir::FunctionId(function))?;
                        let bindings = capture_bindings.get(&function).cloned().unwrap_or_default();
                        if arguments.len() < bindings.len() {
                            return Err(super::error::stage(
                                "NativeCaptureArity",
                                "closure invocation omitted lexical captures",
                            ));
                        }
                        let captures =
                            bindings
                                .into_iter()
                                .zip(arguments.iter().cloned())
                                .map(|(binding, value)| {
                                    runmat_runtime::call::lexical::LexicalCapture { binding, value }
                                })
                                .collect::<Vec<_>>();
                        let arguments = arguments[captures.len()..].to_vec();
                        let execution = super::deopt::invoke(
                            &unit,
                            executor,
                            function,
                            captures,
                            arguments,
                            requested_outputs,
                            runtime,
                        )
                        .await?;
                        normalize_outputs(execution.outputs, requested_outputs)
                    } else if let Some(previous_invoker) = previous_invoker {
                        previous_invoker(function, &arguments, requested_outputs).await
                    } else {
                        Err(super::error::stage(
                            "UndefinedFunction",
                            format!("semantic function {function} is unavailable"),
                        ))
                    }
                })
            },
        )));

    let previous_external_invoker = user_functions::current_external_function_invoker();
    let external_registry = Arc::clone(&registry);
    let external_runtime = runtime.clone();
    let external_invoker =
        user_functions::install_external_function_invoker(Some(Arc::new(move |call| {
            let previous = previous_external_invoker.clone();
            let registry = Arc::clone(&external_registry);
            let runtime = external_runtime.clone();
            Box::pin(async move {
                if registry
                    .get(runmat_hir::FunctionId(call.function))
                    .is_some()
                {
                    return runmat_vm::invoke_semantic_function_value_in_context(
                        call.function,
                        &call.arguments,
                        call.requested_outputs,
                        &registry,
                        runtime,
                    )
                    .await;
                }
                if let Some(previous) = previous {
                    return previous(call).await;
                }
                Err(super::error::stage(
                    "UndefinedFunction",
                    format!("external function '{}' is unavailable", call.display_name),
                ))
            })
        })));

    let previous_lexical_invoker = user_functions::current_lexical_function_invoker();
    let lexical_executor = Rc::clone(&published.executor);
    let lexical_runtime = runtime.clone();
    let lexical_native_functions = native_functions.clone();
    let lexical_unit = unit.clone();
    let lexical_invoker =
        user_functions::install_lexical_function_invoker(Some(Arc::new(move |call| {
            let previous = previous_lexical_invoker.clone();
            let executor = Rc::clone(&lexical_executor);
            let runtime = lexical_runtime.clone();
            let unit = lexical_unit.clone();
            let is_native = u32::try_from(call.function)
                .map(ProgramFunctionId)
                .is_ok_and(|function| lexical_native_functions.contains(&function));
            Box::pin(async move {
                if !is_native {
                    if let Some(previous) = previous {
                        return previous(call).await;
                    }
                    return Err(super::error::stage(
                        "UndefinedFunction",
                        format!("lexical function {} is unavailable", call.function),
                    ));
                }
                let function = program_function(runmat_hir::FunctionId(call.function))?;
                let execution = super::deopt::invoke(
                    &unit,
                    executor,
                    function,
                    call.captures,
                    call.arguments,
                    call.requested_outputs,
                    runtime,
                )
                .await?;
                Ok(runmat_runtime::call::lexical::LexicalCallResult {
                    value: normalize_outputs(execution.outputs, call.requested_outputs)?,
                    captures: execution.captures,
                })
            })
        })));

    let previous_resolver = user_functions::current_semantic_function_resolver();
    let resolver_registry = Arc::clone(&registry);
    let resolver =
        user_functions::install_semantic_function_resolver(Some(Arc::new(move |name| {
            resolver_registry
                .resolve_name(name)
                .map(|function| function.0)
                .or_else(|| {
                    previous_resolver
                        .as_ref()
                        .and_then(|resolver| resolver(name))
                })
        })));
    let mut source_functions = registry
        .functions
        .values()
        .filter_map(|function| {
            function
                .source_id
                .map(|source_id| user_functions::SourceFunctionInfo {
                    source_id,
                    name: function.display_name.clone(),
                    function: function.function.0,
                })
        })
        .collect::<Vec<_>>();
    source_functions.sort_by_key(|function| function.function);
    let catalog = user_functions::install_source_function_catalog(Some(Arc::new(source_functions)));
    let active = user_functions::push_active_semantic_function(function.0 as usize);
    let execution = super::deopt::invoke(
        unit,
        published.executor,
        function,
        Vec::new(),
        arguments,
        requested_outputs,
        runtime,
    )
    .await;
    drop(active);
    drop(catalog);
    drop(resolver);
    drop(lexical_invoker);
    drop(external_invoker);
    drop(invoker);
    let execution = execution?;
    Ok(NativeExecution {
        value: normalize_outputs(execution.outputs, requested_outputs)?,
        loop_backedges: execution.loop_backedges,
    })
}

fn program_function(
    function: runmat_hir::FunctionId,
) -> Result<ProgramFunctionId, runmat_runtime::RuntimeError> {
    u32::try_from(function.0)
        .map(ProgramFunctionId)
        .map_err(|_| {
            super::error::stage("NativeProcedure", "function identity exceeds native schema")
        })
}

fn normalize_outputs(
    mut outputs: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, runmat_runtime::RuntimeError> {
    if outputs.len() != requested_outputs {
        return Err(super::error::stage(
            "NativeOutputArity",
            format!(
                "native execution produced {} outputs for a request of {requested_outputs}",
                outputs.len()
            ),
        ));
    }
    Ok(match requested_outputs {
        0 => Value::OutputList(Vec::new()),
        1 => outputs.remove(0),
        _ => Value::OutputList(outputs),
    })
}
