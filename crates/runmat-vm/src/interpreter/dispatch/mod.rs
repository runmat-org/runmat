mod arithmetic;
mod arrays;
mod calls;
mod control_flow;
mod exceptions;
mod indexing;
mod instruction;
mod object;
mod stack;

use crate::bytecode::Instr;
use crate::interpreter::debug;
use crate::runtime::workspace::{
    refresh_workspace_state, workspace_slot_assigned, workspace_slot_name,
};
use runmat_runtime::RuntimeError;
use runmat_value::{ObjectInstance, StructValue, Tensor, Value};
use std::collections::{HashMap, HashSet};

pub use arrays::{
    create_matrix, create_matrix_dynamic, create_range, pack_to_col, pack_to_row, unpack,
};
pub use calls::{
    build_builtin_expand_multi_args, build_feval_expand_multi_args,
    build_user_function_expand_multi_args, handle_builtin_call_multi, handle_create_closure,
    handle_create_semantic_closure, handle_load_method, handle_load_static_property,
    handle_method_or_member_index_expand_multi_call, handle_method_or_member_index_multi_call,
    handle_prepared_user_function_call, handle_register_class, handle_user_function_call,
    handle_workspace_first_prepared_call, BuiltinHandling, UserCallHandling,
};
pub use control_flow::{apply_control_flow_action, DispatchDecision};
pub use exceptions::{redirect_exception_to_catch, ExceptionHandling};
pub use runmat_runtime::condition::logical_truth_from_value;
pub use stack::{
    emit_stack_top, emit_var, load_bool, load_char_row, load_complex, load_const, load_int,
    load_local, load_string, load_var, store_local, store_var,
};

fn builtin_constant_value(name: &str) -> Option<Value> {
    runmat_builtins::constants()
        .into_iter()
        .find(|constant| constant.name == name)
        .map(|constant| constant.value.clone())
}

pub enum DispatchHandled {
    Generic(DispatchDecision),
    ReturnValue(DispatchDecision),
    Return(DispatchDecision),
}

pub struct DispatchMeta<'a> {
    pub instr: &'a Instr,
    pub var_names: &'a HashMap<usize, String>,
    pub function_registry: &'a crate::bytecode::FunctionRegistry,
    pub source_id: Option<runmat_hir::SourceId>,
    pub call_arg_spans: Option<Vec<runmat_hir::Span>>,
    pub call_counts: &'a [(usize, usize)],
    pub current_function_name: &'a str,
}

pub struct DispatchState<'a> {
    pub stack: &'a mut Vec<Value>,
    pub vars: &'a mut Vec<Value>,
    pub context: &'a mut crate::bytecode::program::ExecutionContext,
    pub try_stack: &'a mut Vec<crate::interpreter::state::ActiveTryHandler>,
    pub last_exception: &'a mut Option<runmat_value::MException>,
    pub imports: &'a mut Vec<(Vec<String>, bool)>,
    pub global_aliases: &'a mut HashMap<usize, String>,
    pub persistent_aliases: &'a mut HashMap<usize, String>,
    pub missing_input_slots: &'a mut HashSet<usize>,
    pub pc: &'a mut usize,
}

pub struct DispatchHooks<'a> {
    pub store_var_before_overwrite: &'a mut dyn FnMut(&Value, &Value),
    pub store_var_after_store: &'a mut dyn FnMut(usize, &Value),
    pub store_local_before_local_overwrite: &'a mut dyn FnMut(&Value, &Value),
    pub store_local_before_var_overwrite: &'a mut dyn FnMut(&Value, &Value),
    pub store_local_after_store: &'a mut dyn FnMut(usize, &Value),
    pub store_local_after_fallback_store: &'a mut dyn FnMut(&str, usize, &Value),
}

fn requested_outputs_from_slot(vars: &[Value], slot: usize) -> Result<usize, RuntimeError> {
    let value = vars.get(slot).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "OutputCountSlotOutOfBounds",
            "requested output slot is out of bounds",
        )
    })?;
    match value {
        Value::Num(n) => {
            if !n.is_finite() || *n < 0.0 || (*n - n.round()).abs() > f64::EPSILON {
                return Err(crate::interpreter::errors::mex(
                    "InvalidOutputCountValue",
                    "requested output count slot must contain a nonnegative integer scalar",
                ));
            }
            Ok(*n as usize)
        }
        Value::Int(i) => i.try_to_usize().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "InvalidOutputCountValue",
                "requested output count slot must contain a nonnegative integer scalar",
            )
        }),
        _ => Err(crate::interpreter::errors::mex(
            "InvalidOutputCountValue",
            "requested output count slot must contain a nonnegative integer scalar",
        )),
    }
}

fn initialize_object_with_defaults(class_name: &str) -> ObjectInstance {
    let empty_default = || Value::Tensor(Tensor::new(vec![], vec![0, 0]).expect("empty tensor"));
    if let Some(def) = runmat_runtime::class_registry::get_class(class_name) {
        let mut chain: Vec<runmat_runtime::class_registry::RuntimeClass> = Vec::new();
        let mut visited = HashSet::new();
        let mut cursor: Option<String> = Some(def.name.clone());
        while let Some(name) = cursor {
            if !visited.insert(name.clone()) {
                break;
            }
            if let Some(class_def) = runmat_runtime::class_registry::get_class(&name) {
                chain.push(class_def.clone());
                cursor = class_def.parent.clone();
            } else {
                break;
            }
        }
        chain.reverse();
        let mut object = ObjectInstance::new(def.name.clone());
        for class_def in chain {
            for (property_name, property_def) in class_def.properties {
                if !property_def.is_static {
                    object.properties.insert(
                        property_name,
                        property_def.default_value.unwrap_or_else(empty_default),
                    );
                }
            }
        }
        object
    } else {
        ObjectInstance::new(class_name.to_string())
    }
}

fn pop_aggregate_literal_values(
    stack: &mut Vec<Value>,
    field_count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    let mut values = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        values.push(stack.pop().ok_or_else(|| {
            crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow while building aggregate literal",
            )
        })?);
    }
    values.reverse();
    Ok(values)
}

fn execution_error(error: runmat_runtime::execution::ExecutionServiceError) -> RuntimeError {
    crate::interpreter::errors::mex("ExecutionService", &error.to_string())
}

fn create_async_future_value(
    context: &crate::bytecode::program::ExecutionContext,
    function: runmat_hir::FunctionId,
    requested_outputs: usize,
    arguments: Vec<Value>,
) -> Result<Value, RuntimeError> {
    runmat_runtime::execution::validate_spawn_capture(&Value::OutputList(arguments.clone()))?;
    let function_identity =
        runmat_types::ProgramFunctionId(u32::try_from(function.0).map_err(|_| {
            crate::interpreter::errors::mex(
                "ExecutionProgram",
                "function identity exceeds portable contract",
            )
        })?);
    let program =
        runmat_runtime::execution::capture_deferred_program(&context.runtime, function_identity)?;
    context
        .runtime
        .execution()
        .create_future(runmat_runtime::execution::DeferredCall {
            environment: runmat_runtime::execution::capture_deferred_environment(&context.runtime)?,
            function: function.0,
            arguments,
            requested_outputs,
            program_revision: program
                .as_ref()
                .map(|capture| capture.revision.clone())
                .or_else(|| context.runtime.program_revision().cloned()),
            program: program.map(|capture| capture.bytes),
        })
        .map(Value::Future)
        .map_err(execution_error)
}

async fn await_execution_value(
    context: &crate::bytecode::program::ExecutionContext,
    value: Value,
    function_registry: &crate::bytecode::FunctionRegistry,
) -> Result<Value, RuntimeError> {
    use runmat_runtime::execution::AwaitAction;

    let mut value = value;
    loop {
        match context
            .runtime
            .execution()
            .begin_await(value)
            .map_err(execution_error)?
        {
            AwaitAction::Passthrough(value) | AwaitAction::Completed(value) => return Ok(value),
            AwaitAction::Pending(pending) => {
                yield_once().await;
                value = pending;
            }
            AwaitAction::ExecuteFuture { handle, call } => {
                let descriptor = runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                    runmat_hir::CallableIdentity::BoundFunction(runmat_hir::FunctionId(
                        call.function,
                    )),
                    call.arguments,
                    call.requested_outputs,
                    runmat_hir::CallableFallbackPolicy::None,
                    runmat_runtime::call::descriptor::CallableCallKind::Direct,
                );
                let runtime = context
                    .runtime
                    .clone()
                    .with_invocation_environment(&call.environment)
                    .map_err(|error| {
                        execution_error(runmat_runtime::execution::ExecutionServiceError::Failed(
                            error.to_string(),
                        ))
                    })?;
                let result = runtime
                    .scope(
                        runmat_runtime::call::descriptor::execute_callable_descriptor(descriptor),
                    )
                    .await
                    .map(|value| calls::normalize_requested_outputs(value, call.requested_outputs));
                let stored = result.as_ref().map(Clone::clone).map_err(|error| {
                    runmat_runtime::execution::ExecutionServiceError::Failed(error.to_string())
                });
                context
                    .runtime
                    .execution()
                    .complete_future(&handle, stored)
                    .map_err(execution_error)?;
                let _ = function_registry;
                return result;
            }
        }
    }
}

async fn yield_once() {
    let mut yielded = false;
    futures::future::poll_fn(|context| {
        if yielded {
            std::task::Poll::Ready(())
        } else {
            yielded = true;
            context.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    })
    .await;
}

#[cfg(feature = "native-accel")]
fn clear_popped_value_residency_excluding_live_values(
    popped: &Value,
    stack: &[Value],
    vars: &[Value],
    context: &crate::bytecode::program::ExecutionContext,
) {
    let mut live = Vec::with_capacity(stack.len() + vars.len() + context.locals.len());
    live.extend(stack.iter().cloned());
    live.extend(vars.iter().cloned());
    live.extend(context.locals.iter().cloned());
    if let Err(err) =
        crate::accel::residency::clear_value_excluding(popped, &Value::OutputList(live))
    {
        log::warn!("failed to clear popped GPU residency: {err}");
    }
}

#[cfg(feature = "native-accel")]
fn clear_scope_value_residency_excluding_live_values(
    dropped_local: &Value,
    stack: &[Value],
    vars: &[Value],
    context: &crate::bytecode::program::ExecutionContext,
) {
    let mut live = Vec::with_capacity(stack.len() + vars.len() + context.locals.len());
    live.extend(stack.iter().cloned());
    live.extend(vars.iter().cloned());
    live.extend(context.locals.iter().cloned());
    if let Err(err) =
        crate::accel::residency::clear_value_excluding(dropped_local, &Value::OutputList(live))
    {
        log::warn!("failed to clear dropped local GPU residency: {err}");
    }
}

#[cfg(feature = "native-accel")]
fn clear_overwritten_var_residency_excluding_live_values(
    overwritten: &Value,
    overwritten_index: usize,
    stack: &[Value],
    vars: &[Value],
    context: &crate::bytecode::program::ExecutionContext,
) {
    let mut live = Vec::with_capacity(stack.len() + vars.len() + context.locals.len());
    live.extend(stack.iter().cloned());
    for (idx, value) in vars.iter().enumerate() {
        if idx != overwritten_index {
            live.push(value.clone());
        }
    }
    live.extend(context.locals.iter().cloned());
    if let Err(err) =
        crate::accel::residency::clear_value_excluding(overwritten, &Value::OutputList(live))
    {
        log::warn!("failed to clear overwritten variable GPU residency: {err}");
    }
}

#[cfg(feature = "native-accel")]
fn clear_overwritten_local_residency_excluding_live_values(
    overwritten: &Value,
    overwritten_local_index: usize,
    stack: &[Value],
    vars: &[Value],
    context: &crate::bytecode::program::ExecutionContext,
) {
    let mut live = Vec::with_capacity(stack.len() + vars.len() + context.locals.len());
    live.extend(stack.iter().cloned());
    live.extend(vars.iter().cloned());
    for (idx, value) in context.locals.iter().enumerate() {
        if idx != overwritten_local_index {
            live.push(value.clone());
        }
    }
    if let Err(err) =
        crate::accel::residency::clear_value_excluding(overwritten, &Value::OutputList(live))
    {
        log::warn!("failed to clear overwritten local GPU residency: {err}");
    }
}

pub async fn dispatch_instruction<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
    hooks: DispatchHooks<'a>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    if indexing::dispatch_indexing(
        meta.instr,
        state.stack,
        state.vars,
        meta.function_registry,
        *state.pc,
    )
    .await?
        || object::dispatch_object(meta.instr, state.stack, meta.current_function_name).await?
        || arithmetic::dispatch_arithmetic(meta.instr, state.stack).await?
    {
        return Ok(Some(DispatchHandled::Generic(
            DispatchDecision::FallThrough,
        )));
    }
    instruction::dispatch(meta, state, hooks).await
}

#[cfg(test)]
mod tests {
    use super::logical_truth_from_value;
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Tensor, Value};

    #[test]
    fn logical_truth_reads_typed_integer_tensor_storage_exactly() {
        let zero = Tensor::new_integer(IntegerStorage::U64(vec![0]), vec![1, 1])
            .expect("zero integer tensor");
        assert!(!block_on(logical_truth_from_value(
            &Value::Tensor(zero),
            "if condition"
        ))
        .unwrap());

        let nonzero = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1])
            .expect("nonzero integer tensor");
        assert!(block_on(logical_truth_from_value(
            &Value::Tensor(nonzero),
            "if condition"
        ))
        .unwrap());
    }

    #[test]
    fn logical_truth_accepts_typed_integer_scalar_with_cleared_mirror() {
        let value = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("integer scalar");

        assert!(block_on(logical_truth_from_value(
            &Value::Tensor(value),
            "if condition"
        ))
        .unwrap());
    }
}
