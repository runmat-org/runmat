use std::{collections::BTreeSet, rc::Rc, sync::Arc};

use runmat_runtime::user_functions;
use runmat_types::ProgramFunctionId;
use runmat_value::Value;

use crate::ExecutableUnit;

// Semantic invokers use Runtime's established Arc callback ABI, while the
// executable-memory owner, RuntimeContext, and GC Values are deliberately
// invocation-thread confined. This callback never crosses that thread.
#[allow(clippy::arc_with_non_send_sync)]
pub(crate) async fn invoke(
    unit: &ExecutableUnit,
    preferred_function: Option<&str>,
    arguments: Vec<Value>,
    requested_outputs: usize,
    runtime: runmat_runtime::context::RuntimeContext,
) -> Result<Value, runmat_runtime::RuntimeError> {
    let compiled = super::compile::compile(unit, preferred_function)?;
    let function = preferred_function
        .map(|name| {
            unit.functions()
                .resolve_name(name)
                .ok_or_else(|| super::error::stage("NativeProcedure", name))
                .and_then(program_function)
        })
        .transpose()?
        .unwrap_or(compiled.entrypoint);
    let registry = Arc::new(unit.functions().clone());
    let native_functions = unit
        .mir()
        .bodies
        .keys()
        .map(|function| program_function(*function))
        .collect::<Result<BTreeSet<_>, _>>()?;

    let previous_invoker = user_functions::current_semantic_function_invoker();
    let nested_executor = Rc::clone(&compiled.executor);
    let nested_runtime = runtime.clone();
    let invoker = user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, arguments, requested_outputs| {
            let arguments = arguments.to_vec();
            let previous_invoker = previous_invoker.clone();
            let executor = Rc::clone(&nested_executor);
            let runtime = nested_runtime.clone();
            let is_native = u32::try_from(function)
                .map(ProgramFunctionId)
                .is_ok_and(|function| native_functions.contains(&function));
            Box::pin(async move {
                if is_native {
                    let function = program_function(runmat_hir::FunctionId(function))?;
                    let execution = executor
                        .invoke_async(function, arguments, requested_outputs, runtime)
                        .await
                        .map_err(super::error::from_jit_error)?;
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
    let execution = compiled
        .executor
        .invoke_async(function, arguments, requested_outputs, runtime)
        .await
        .map_err(super::error::from_jit_error);
    drop(active);
    drop(catalog);
    drop(resolver);
    drop(invoker);
    normalize_outputs(execution?.outputs, requested_outputs)
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
