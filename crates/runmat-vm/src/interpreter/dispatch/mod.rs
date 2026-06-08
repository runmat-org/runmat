mod arithmetic;
mod arrays;
mod calls;
mod control_flow;
mod exceptions;
mod indexing;
mod object;
mod stack;

use crate::bytecode::Instr;
use crate::interpreter::debug;
use crate::runtime::workspace::{
    refresh_workspace_state, workspace_slot_assigned, workspace_slot_name,
};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{IntValue, ObjectInstance, StructValue, Tensor, Value};
use runmat_runtime::dispatcher::gather_if_needed_async;
use runmat_runtime::RuntimeError;
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
    BuiltinHandling, UserCallHandling,
};
pub use control_flow::{apply_control_flow_action, DispatchDecision};
pub use exceptions::{redirect_exception_to_catch, ExceptionHandling};
pub use stack::{
    emit_stack_top, emit_var, load_bool, load_char_row, load_complex, load_const, load_local,
    load_string, load_var, store_local, store_var,
};

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
    pub try_stack: &'a mut Vec<(usize, Option<usize>)>,
    pub last_exception: &'a mut Option<runmat_builtins::MException>,
    pub imports: &'a mut Vec<(Vec<String>, bool)>,
    pub global_aliases: &'a mut HashMap<usize, String>,
    pub persistent_aliases: &'a mut HashMap<usize, String>,
    pub missing_input_slots: &'a mut HashSet<usize>,
    pub pc: &'a mut usize,
}

pub struct DispatchHooks<'a> {
    pub clear_value_residency: &'a mut dyn FnMut(&Value),
    pub store_var_before_overwrite: &'a mut dyn FnMut(&Value, &Value),
    pub store_var_after_store: &'a mut dyn FnMut(usize, &Value),
    pub store_local_before_local_overwrite: &'a mut dyn FnMut(&Value, &Value),
    pub store_local_before_var_overwrite: &'a mut dyn FnMut(&Value, &Value),
    pub store_local_after_store: &'a mut dyn FnMut(usize, &Value),
    pub store_local_after_fallback_store: &'a mut dyn FnMut(&str, usize, &Value),
}

pub async fn logical_truth_from_value(value: &Value, label: &str) -> Result<bool, RuntimeError> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => Ok(*n != 0.0),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::LogicalArray(array) => Err(crate::interpreter::errors::mex(
            "InvalidConditionType",
            &format!(
                "{label}: expected scalar logical or numeric value, got logical array with {} elements",
                array.data.len()
            ),
        )),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0] != 0.0),
        Value::Tensor(tensor) => Err(crate::interpreter::errors::mex(
            "InvalidConditionType",
            &format!(
                "{label}: expected scalar logical or numeric value, got numeric array with {} elements",
                tensor.data.len()
            ),
        )),
        Value::GpuTensor(_) => {
            let gathered = gather_if_needed_async(value)
                .await
                .map_err(|e| format!("{label}: {e}"))?;
            Box::pin(logical_truth_from_value(&gathered, label)).await
        }
        other => Err(crate::interpreter::errors::mex(
            "InvalidConditionType",
            &format!("{label}: expected scalar logical or numeric value, got {other:?}"),
        )),
    }
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
        Value::Int(i) => usize::try_from(i.to_i64()).map_err(|_| {
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

fn for_each_gpu_handle_in_value(
    value: &Value,
    f: &mut impl FnMut(&GpuTensorHandle) -> Result<(), RuntimeError>,
) -> Result<(), RuntimeError> {
    let mut visited_handle_targets = HashSet::new();
    for_each_gpu_handle_in_value_with_visited(value, f, &mut visited_handle_targets)
}

fn for_each_gpu_handle_in_value_with_visited(
    value: &Value,
    f: &mut impl FnMut(&GpuTensorHandle) -> Result<(), RuntimeError>,
    visited_handle_targets: &mut HashSet<usize>,
) -> Result<(), RuntimeError> {
    match value {
        Value::GpuTensor(handle) => f(handle),
        Value::Cell(cell) => {
            for elem in &cell.data {
                for_each_gpu_handle_in_value_with_visited(elem, f, visited_handle_targets)?;
            }
            Ok(())
        }
        Value::Struct(struct_value) => {
            for elem in struct_value.fields.values() {
                for_each_gpu_handle_in_value_with_visited(elem, f, visited_handle_targets)?;
            }
            Ok(())
        }
        Value::Object(object_value) => {
            for elem in object_value.properties.values() {
                for_each_gpu_handle_in_value_with_visited(elem, f, visited_handle_targets)?;
            }
            Ok(())
        }
        Value::Closure(closure) => {
            for capture in &closure.captures {
                for_each_gpu_handle_in_value_with_visited(capture, f, visited_handle_targets)?;
            }
            Ok(())
        }
        Value::OutputList(values) => {
            for elem in values {
                for_each_gpu_handle_in_value_with_visited(elem, f, visited_handle_targets)?;
            }
            Ok(())
        }
        Value::HandleObject(handle) => {
            let raw_target = unsafe { handle.target.as_raw() } as usize;
            if visited_handle_targets.insert(raw_target) {
                let target = unsafe { &*handle.target.as_raw() };
                for_each_gpu_handle_in_value_with_visited(target, f, visited_handle_targets)?;
            }
            Ok(())
        }
        Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::ComplexTensor(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::ClassRef(_)
        | Value::MException(_) => Ok(()),
    }
}

fn enforce_spawn_value_concurrency_policy(value: &Value) -> Result<(), RuntimeError> {
    for_each_gpu_handle_in_value(value, &mut |handle| {
        let provider = runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
            crate::interpreter::errors::mex(
                "SpawnProviderUnavailable",
                &format!(
                    "spawn cannot capture GPU handle buffer {} (device {}) without an active provider",
                    handle.buffer_id, handle.device_id
                ),
            )
        })?;
        let policy = provider.spawn_handle_concurrency();
        if matches!(
            policy,
            runmat_accelerate_api::SpawnHandleConcurrency::Reject
        ) {
            return Err(crate::interpreter::errors::mex(
                "SpawnGpuHandleUnsupported",
                &format!(
                    "spawn cannot capture GPU handle buffer {} on provider '{}' (spawn_handle_concurrency={})",
                    handle.buffer_id,
                    provider.device_info(),
                    policy.as_str()
                ),
            ));
        }
        Ok(())
    })
}

const SPAWN_TASK_KIND_FIELD: &str = "__runmat_spawn_kind";
const SPAWN_TASK_ID_FIELD: &str = "__runmat_spawn_id";
const SPAWN_TASK_PAYLOAD_FIELD: &str = "__runmat_spawn_payload";
const SPAWN_TASK_KIND_VALUE: &str = "task";
const FUTURE_KIND_FIELD: &str = "__runmat_future_kind";
const FUTURE_FUNCTION_FIELD: &str = "__runmat_future_function";
const FUTURE_REQUESTED_OUTPUTS_FIELD: &str = "__runmat_future_requested_outputs";
const FUTURE_ARGS_FIELD: &str = "__runmat_future_args";
const FUTURE_KIND_VALUE: &str = "async_future";

fn allocate_spawn_task_id(context: &mut crate::bytecode::program::ExecutionContext) -> u64 {
    loop {
        let candidate = context.next_spawn_task_id;
        context.next_spawn_task_id = context.next_spawn_task_id.wrapping_add(1);
        if context.spawned_task_ids.insert(candidate) {
            return candidate;
        }
    }
}

fn wrap_spawned_value(
    context: &mut crate::bytecode::program::ExecutionContext,
    value: Value,
) -> Value {
    let task_id = allocate_spawn_task_id(context);
    let mut task = StructValue::new();
    task.fields.insert(
        SPAWN_TASK_KIND_FIELD.to_string(),
        Value::String(SPAWN_TASK_KIND_VALUE.to_string()),
    );
    task.fields.insert(
        SPAWN_TASK_ID_FIELD.to_string(),
        Value::Int(IntValue::U64(task_id)),
    );
    task.fields
        .insert(SPAWN_TASK_PAYLOAD_FIELD.to_string(), value);
    Value::Struct(task)
}

fn create_async_future_value(
    function: runmat_hir::FunctionId,
    requested_outputs: usize,
    args: Vec<Value>,
) -> Value {
    let mut future = StructValue::new();
    future.fields.insert(
        FUTURE_KIND_FIELD.to_string(),
        Value::String(FUTURE_KIND_VALUE.to_string()),
    );
    future.fields.insert(
        FUTURE_FUNCTION_FIELD.to_string(),
        Value::Int(IntValue::U64(function.0 as u64)),
    );
    future.fields.insert(
        FUTURE_REQUESTED_OUTPUTS_FIELD.to_string(),
        Value::Int(IntValue::U64(requested_outputs as u64)),
    );
    future
        .fields
        .insert(FUTURE_ARGS_FIELD.to_string(), Value::OutputList(args));
    Value::Struct(future)
}

fn initialize_object_with_defaults(class_name: &str) -> ObjectInstance {
    let empty_default = || Value::Tensor(Tensor::new(vec![], vec![0, 0]).expect("empty tensor"));
    if let Some(def) = runmat_builtins::get_class(class_name) {
        let mut chain: Vec<runmat_builtins::ClassDef> = Vec::new();
        let mut visited = HashSet::new();
        let mut cursor: Option<String> = Some(def.name.clone());
        while let Some(name) = cursor {
            if !visited.insert(name.clone()) {
                break;
            }
            if let Some(class_def) = runmat_builtins::get_class(&name) {
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

async fn resolve_semantic_future_value(value: Value) -> Result<Value, RuntimeError> {
    let Value::Struct(future) = value else {
        return Ok(value);
    };
    let is_future = matches!(
        future.fields.get(FUTURE_KIND_FIELD),
        Some(Value::String(kind)) if kind == FUTURE_KIND_VALUE
    );
    if !is_future {
        return Ok(Value::Struct(future));
    }
    let function = match future.fields.get(FUTURE_FUNCTION_FIELD) {
        Some(Value::Int(IntValue::U64(id))) => runmat_hir::FunctionId(*id as usize),
        _ => {
            return Err(crate::interpreter::errors::mex(
                "AwaitOperandInvalid",
                "future descriptor is missing a valid semantic function identifier",
            ))
        }
    };
    let requested_outputs = match future.fields.get(FUTURE_REQUESTED_OUTPUTS_FIELD) {
        Some(Value::Int(IntValue::U64(count))) => *count as usize,
        _ => {
            return Err(crate::interpreter::errors::mex(
                "AwaitOperandInvalid",
                "future descriptor is missing a valid requested output count",
            ))
        }
    };
    let args = match future.fields.get(FUTURE_ARGS_FIELD) {
        Some(Value::OutputList(args)) => args.clone(),
        _ => {
            return Err(crate::interpreter::errors::mex(
                "AwaitOperandInvalid",
                "future descriptor is missing argument payload values",
            ))
        }
    };

    let descriptor = crate::call::descriptor::CallableDescriptor::resolved(
        runmat_hir::CallableIdentity::BoundFunction(function),
        args,
        requested_outputs,
        runmat_hir::CallableFallbackPolicy::None,
        crate::call::descriptor::CallableCallKind::Direct,
    );
    let value = crate::call::descriptor::execute_callable_descriptor(descriptor).await?;
    Ok(calls::normalize_requested_outputs(value, requested_outputs))
}

fn unwrap_spawned_value(
    context: &mut crate::bytecode::program::ExecutionContext,
    value: Value,
) -> Result<Value, RuntimeError> {
    let Value::Struct(task) = value else {
        // Await preserves pass-through behavior for non-task values.
        return Ok(value);
    };
    let is_spawn_task = matches!(
        task.fields.get(SPAWN_TASK_KIND_FIELD),
        Some(Value::String(kind)) if kind == SPAWN_TASK_KIND_VALUE
    );
    if !is_spawn_task {
        return Ok(Value::Struct(task));
    }
    let task_id = match task.fields.get(SPAWN_TASK_ID_FIELD) {
        Some(Value::Int(IntValue::U64(id))) => *id,
        _ => {
            return Err(crate::interpreter::errors::mex(
                "AwaitOperandInvalid",
                "await task handle is missing a valid task identifier",
            ))
        }
    };
    if !context.spawned_task_ids.remove(&task_id) {
        return Err(crate::interpreter::errors::mex(
            "AwaitOperandInvalid",
            "await task handle is stale or was already consumed",
        ));
    }
    task.fields
        .get(SPAWN_TASK_PAYLOAD_FIELD)
        .cloned()
        .ok_or_else(|| {
            crate::interpreter::errors::mex(
                "AwaitOperandInvalid",
                "await task handle is missing payload value",
            )
        })
}

fn spawn_task_id_from_value(value: &Value) -> Option<u64> {
    let Value::Struct(task) = value else {
        return None;
    };
    let is_spawn_task = matches!(
        task.fields.get(SPAWN_TASK_KIND_FIELD),
        Some(Value::String(kind)) if kind == SPAWN_TASK_KIND_VALUE
    );
    if !is_spawn_task {
        return None;
    }
    match task.fields.get(SPAWN_TASK_ID_FIELD) {
        Some(Value::Int(IntValue::U64(id))) => Some(*id),
        _ => None,
    }
}

fn collect_spawn_task_ids_in_value(value: &Value, output: &mut HashSet<u64>) {
    let mut visited_handle_targets = HashSet::new();
    collect_spawn_task_ids_in_value_with_visited(value, output, &mut visited_handle_targets);
}

fn collect_spawn_task_ids_in_value_with_visited(
    value: &Value,
    output: &mut HashSet<u64>,
    visited_handle_targets: &mut HashSet<usize>,
) {
    if let Some(task_id) = spawn_task_id_from_value(value) {
        output.insert(task_id);
    }
    match value {
        Value::Cell(cell) => {
            for entry in &cell.data {
                collect_spawn_task_ids_in_value_with_visited(entry, output, visited_handle_targets);
            }
        }
        Value::Struct(struct_value) => {
            for entry in struct_value.fields.values() {
                collect_spawn_task_ids_in_value_with_visited(entry, output, visited_handle_targets);
            }
        }
        Value::Object(object_value) => {
            for entry in object_value.properties.values() {
                collect_spawn_task_ids_in_value_with_visited(entry, output, visited_handle_targets);
            }
        }
        Value::Closure(closure) => {
            for entry in &closure.captures {
                collect_spawn_task_ids_in_value_with_visited(entry, output, visited_handle_targets);
            }
        }
        Value::OutputList(values) => {
            for entry in values {
                collect_spawn_task_ids_in_value_with_visited(entry, output, visited_handle_targets);
            }
        }
        Value::HandleObject(handle) => {
            let raw_target = unsafe { handle.target.as_raw() } as usize;
            if visited_handle_targets.insert(raw_target) {
                let target = unsafe { &*handle.target.as_raw() };
                collect_spawn_task_ids_in_value_with_visited(
                    target,
                    output,
                    visited_handle_targets,
                );
            }
        }
        Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::ComplexTensor(_)
        | Value::GpuTensor(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::ClassRef(_)
        | Value::MException(_) => {}
    }
}

fn value_contains_spawn_task_id(value: &Value, task_id: u64) -> bool {
    let mut visited_handle_targets = HashSet::new();
    value_contains_spawn_task_id_with_visited(value, task_id, &mut visited_handle_targets)
}

fn value_contains_spawn_task_id_with_visited(
    value: &Value,
    task_id: u64,
    visited_handle_targets: &mut HashSet<usize>,
) -> bool {
    if spawn_task_id_from_value(value) == Some(task_id) {
        return true;
    }
    match value {
        Value::Cell(cell) => cell.data.iter().any(|entry| {
            value_contains_spawn_task_id_with_visited(entry, task_id, visited_handle_targets)
        }),
        Value::Struct(struct_value) => struct_value.fields.values().any(|entry| {
            value_contains_spawn_task_id_with_visited(entry, task_id, visited_handle_targets)
        }),
        Value::Object(object_value) => object_value.properties.values().any(|entry| {
            value_contains_spawn_task_id_with_visited(entry, task_id, visited_handle_targets)
        }),
        Value::Closure(closure) => closure.captures.iter().any(|entry| {
            value_contains_spawn_task_id_with_visited(entry, task_id, visited_handle_targets)
        }),
        Value::OutputList(values) => values.iter().any(|entry| {
            value_contains_spawn_task_id_with_visited(entry, task_id, visited_handle_targets)
        }),
        Value::HandleObject(handle) => {
            let raw_target = unsafe { handle.target.as_raw() } as usize;
            if !visited_handle_targets.insert(raw_target) {
                return false;
            }
            let target = unsafe { &*handle.target.as_raw() };
            value_contains_spawn_task_id_with_visited(target, task_id, visited_handle_targets)
        }
        Value::Int(_)
        | Value::Num(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Tensor(_)
        | Value::SparseTensor(_)
        | Value::ComplexTensor(_)
        | Value::GpuTensor(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::ClassRef(_)
        | Value::MException(_) => false,
    }
}

fn spawn_task_id_still_live(
    task_id: u64,
    stack: &[Value],
    vars: &[Value],
    context: &crate::bytecode::program::ExecutionContext,
    excluded_var: Option<usize>,
    excluded_local: Option<usize>,
) -> bool {
    if stack
        .iter()
        .any(|value| value_contains_spawn_task_id(value, task_id))
    {
        return true;
    }
    for (index, value) in vars.iter().enumerate() {
        if excluded_var == Some(index) {
            continue;
        }
        if value_contains_spawn_task_id(value, task_id) {
            return true;
        }
    }
    for (index, value) in context.locals.iter().enumerate() {
        if excluded_local == Some(index) {
            continue;
        }
        if value_contains_spawn_task_id(value, task_id) {
            return true;
        }
    }
    false
}

fn retire_spawn_task_id_if_dropped(
    context: &mut crate::bytecode::program::ExecutionContext,
    value: &Value,
    stack: &[Value],
    vars: &[Value],
    excluded_var: Option<usize>,
    excluded_local: Option<usize>,
) {
    let mut task_ids = HashSet::new();
    collect_spawn_task_ids_in_value(value, &mut task_ids);
    for id in task_ids {
        if !spawn_task_id_still_live(id, stack, vars, context, excluded_var, excluded_local) {
            context.spawned_task_ids.remove(&id);
        }
    }
}

fn retire_spawn_task_id_if_replaced(
    context: &mut crate::bytecode::program::ExecutionContext,
    current: &Value,
    incoming: &Value,
    stack: &[Value],
    vars: &[Value],
    excluded_var: Option<usize>,
    excluded_local: Option<usize>,
) {
    let mut current_ids = HashSet::new();
    collect_spawn_task_ids_in_value(current, &mut current_ids);
    if current_ids.is_empty() {
        return;
    }
    let mut incoming_ids = HashSet::new();
    collect_spawn_task_ids_in_value(incoming, &mut incoming_ids);
    for current_id in current_ids {
        if incoming_ids.contains(&current_id) {
            continue;
        }
        if !spawn_task_id_still_live(
            current_id,
            stack,
            vars,
            context,
            excluded_var,
            excluded_local,
        ) {
            context.spawned_task_ids.remove(&current_id);
        }
    }
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
    crate::accel::residency::clear_value_excluding(popped, &Value::OutputList(live));
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
    crate::accel::residency::clear_value_excluding(dropped_local, &Value::OutputList(live));
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
    crate::accel::residency::clear_value_excluding(overwritten, &Value::OutputList(live));
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
    crate::accel::residency::clear_value_excluding(overwritten, &Value::OutputList(live));
}

pub async fn dispatch_instruction(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
    hooks: DispatchHooks<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let DispatchMeta {
        instr,
        var_names,
        function_registry,
        source_id,
        call_arg_spans,
        call_counts,
        current_function_name,
    } = meta;
    let DispatchState {
        stack,
        vars,
        context,
        try_stack,
        last_exception,
        imports,
        global_aliases,
        persistent_aliases,
        missing_input_slots,
        pc,
    } = state;
    let DispatchHooks {
        clear_value_residency,
        store_var_before_overwrite,
        store_var_after_store,
        store_local_before_local_overwrite,
        store_local_before_var_overwrite,
        store_local_after_store,
        store_local_after_fallback_store,
    } = hooks;
    match instr {
        _ if indexing::dispatch_indexing(
            instr,
            stack,
            vars,
            function_registry,
            *pc,
            &mut *clear_value_residency,
        )
        .await? =>
        {
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ if object::dispatch_object(instr, stack, current_function_name).await? => Ok(Some(
            DispatchHandled::Generic(DispatchDecision::FallThrough),
        )),
        _ if arithmetic::dispatch_arithmetic(instr, stack).await? => Ok(Some(
            DispatchHandled::Generic(DispatchDecision::FallThrough),
        )),
        Instr::EmitStackTop { label } => {
            emit_stack_top(stack, label, var_names).await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::EmitVar { var_index, label } => {
            emit_var(vars, *var_index, label, var_names).await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadConst(value) => {
            load_const(stack, *value);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadComplex(re, im) => {
            load_complex(stack, *re, *im);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadBool(value) => {
            load_bool(stack, *value);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadString(value) => {
            load_string(stack, value.clone());
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadCharRow(value) => {
            load_char_row(stack, value.clone())?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadLocal(offset) => {
            load_local(stack, context, vars, *offset)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadVar(index) => {
            if let Some(alias) = global_aliases.get(index) {
                if let Some(global_value) = crate::runtime::globals::get_global_value(alias) {
                    if *index >= vars.len() {
                        vars.resize(*index + 1, Value::Num(0.0));
                        refresh_workspace_state(vars);
                    }
                    vars[*index] = global_value;
                }
            }
            if missing_input_slots.contains(index) {
                return Err(crate::interpreter::errors::mex(
                    "NotEnoughInputs",
                    "Not enough input arguments.",
                ));
            }
            if let (Some(false), Some(slot_name), Some(var_name)) = (
                workspace_slot_assigned(*index),
                workspace_slot_name(*index),
                var_names.get(index),
            ) {
                if slot_name == *var_name {
                    return Err(crate::interpreter::errors::mex(
                        "UndefinedVariable",
                        &format!("Undefined variable: {slot_name}"),
                    ));
                }
            }
            let value = vars[*index].clone();
            debug::trace_load_var(*pc, *index, &value);
            load_var(stack, vars, *index);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadVarForIndexAssignment(index) => {
            if let Some(alias) = global_aliases.get(index) {
                if let Some(global_value) = crate::runtime::globals::get_global_value(alias) {
                    if *index >= vars.len() {
                        vars.resize(*index + 1, Value::Num(0.0));
                        refresh_workspace_state(vars);
                    }
                    vars[*index] = global_value;
                }
            }
            if missing_input_slots.contains(index) {
                return Err(crate::interpreter::errors::mex(
                    "NotEnoughInputs",
                    "Not enough input arguments.",
                ));
            }
            if let (Some(false), Some(slot_name), Some(var_name)) = (
                workspace_slot_assigned(*index),
                workspace_slot_name(*index),
                var_names.get(index),
            ) {
                if slot_name == *var_name {
                    let empty = Tensor::new(Vec::new(), vec![0, 0]).map_err(|err| {
                        crate::interpreter::errors::mex(
                            "IndexAssignmentBaseInit",
                            &format!(
                                "failed to initialize undefined indexed-assignment base '{slot_name}': {err}"
                            ),
                        )
                    })?;
                    debug::trace_load_var(*pc, *index, &Value::Tensor(empty.clone()));
                    stack.push(Value::Tensor(empty));
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::FallThrough,
                    )));
                }
            }
            let value = vars[*index].clone();
            debug::trace_load_var(*pc, *index, &value);
            load_var(stack, vars, *index);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::StoreVar(index) => {
            let preview = stack
                .last()
                .cloned()
                .ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?;
            debug::trace_store_var(*pc, *index, &preview);
            if *index < vars.len() {
                retire_spawn_task_id_if_replaced(
                    context,
                    &vars[*index],
                    &preview,
                    stack,
                    vars,
                    Some(*index),
                    None,
                );
                #[cfg(feature = "native-accel")]
                clear_overwritten_var_residency_excluding_live_values(
                    &vars[*index],
                    *index,
                    stack,
                    vars,
                    context,
                );
            }
            store_var(
                stack,
                vars,
                *index,
                var_names,
                store_var_before_overwrite,
                store_var_after_store,
            )?;
            missing_input_slots.remove(index);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::StoreLocal(offset) => {
            if let Some(incoming) = stack.last().cloned() {
                if let Some(current_frame) = context.call_stack.last() {
                    let local_index = current_frame.locals_start + *offset;
                    if local_index < context.locals.len() {
                        let current_value = context.locals[local_index].clone();
                        retire_spawn_task_id_if_replaced(
                            context,
                            &current_value,
                            &incoming,
                            stack,
                            vars,
                            None,
                            Some(local_index),
                        );
                        #[cfg(feature = "native-accel")]
                        clear_overwritten_local_residency_excluding_live_values(
                            &current_value,
                            local_index,
                            stack,
                            vars,
                            context,
                        );
                    }
                } else if *offset < vars.len() {
                    retire_spawn_task_id_if_replaced(
                        context,
                        &vars[*offset],
                        &incoming,
                        stack,
                        vars,
                        Some(*offset),
                        None,
                    );
                    #[cfg(feature = "native-accel")]
                    clear_overwritten_var_residency_excluding_live_values(
                        &vars[*offset],
                        *offset,
                        stack,
                        vars,
                        context,
                    );
                }
            }
            if context.call_stack.last().is_none() {
                store_var(
                    stack,
                    vars,
                    *offset,
                    var_names,
                    store_local_before_var_overwrite,
                    |stored_index, stored_value| {
                        store_local_after_fallback_store("<main>", stored_index, stored_value);
                    },
                )?;
                missing_input_slots.remove(offset);
            } else {
                store_local(
                    stack,
                    context,
                    vars,
                    *offset,
                    store_local_before_local_overwrite,
                    store_local_before_var_overwrite,
                    store_local_after_store,
                    store_local_after_fallback_store,
                )?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Swap => {
            crate::ops::stack::swap(stack)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Pop => {
            if let Some(value) = stack.pop() {
                retire_spawn_task_id_if_dropped(context, &value, stack, vars, None, None);
                #[cfg(feature = "native-accel")]
                clear_popped_value_residency_excluding_live_values(&value, stack, vars, context);
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::AndAnd(target) => Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
            crate::ops::control_flow::and_and(
                logical_truth_from_value(
                    &crate::interpreter::stack::pop_value(stack)?,
                    "short-circuit && condition",
                )
                .await?,
                *target,
            ),
            pc,
        )))),
        Instr::OrOr(target) => Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
            crate::ops::control_flow::or_or(
                logical_truth_from_value(
                    &crate::interpreter::stack::pop_value(stack)?,
                    "short-circuit || condition",
                )
                .await?,
                *target,
            ),
            pc,
        )))),
        Instr::JumpIfFalse(target) => {
            let cond = crate::interpreter::stack::pop_value(stack)?;
            let truth = logical_truth_from_value(&cond, "if condition").await?;
            Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
                crate::ops::control_flow::jump_if_false(truth, *target),
                pc,
            ))))
        }
        Instr::Jump(target) => Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
            crate::ops::control_flow::jump(*target),
            pc,
        )))),
        Instr::EnterTry(catch_pc, catch_var) => {
            crate::ops::control_flow::enter_try(try_stack, *catch_pc, *catch_var);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::PopTry => {
            crate::ops::control_flow::pop_try(try_stack);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::ReturnValue => Ok(Some(DispatchHandled::ReturnValue(
            apply_control_flow_action(crate::ops::control_flow::return_value(stack)?, pc),
        ))),
        Instr::Return => Ok(Some(DispatchHandled::Return(apply_control_flow_action(
            crate::ops::control_flow::return_void(),
            pc,
        )))),
        Instr::EnterScope(local_count) => {
            crate::ops::control_flow::enter_scope(&mut context.locals, *local_count);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::ExitScope(local_count) => {
            for _ in 0..*local_count {
                if let Some(value) = context.locals.pop() {
                    if let Some(id) = spawn_task_id_from_value(&value) {
                        if !spawn_task_id_still_live(id, stack, vars, context, None, None) {
                            context.spawned_task_ids.remove(&id);
                        }
                    }
                    #[cfg(feature = "native-accel")]
                    clear_scope_value_residency_excluding_live_values(&value, stack, vars, context);
                    #[cfg(not(feature = "native-accel"))]
                    clear_value_residency(&value);
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::RegisterImport { path, wildcard } => {
            imports.push((path.clone(), *wildcard));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclareGlobal(indices) => {
            crate::runtime::globals::declare_global(indices.clone(), vars);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclareGlobalNamed(indices, names) => {
            crate::runtime::globals::declare_global_named(
                indices.clone(),
                names.clone(),
                vars,
                global_aliases,
            );
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclarePersistent(indices) => {
            crate::runtime::globals::declare_persistent(
                current_function_name,
                indices.clone(),
                vars,
            );
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::DeclarePersistentNamed(indices, names) => {
            crate::runtime::globals::declare_persistent_named(
                current_function_name,
                indices.clone(),
                names.clone(),
                vars,
                persistent_aliases,
            );
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::PackToRow(count) => {
            pack_to_row(stack, *count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::PackToCol(count) => {
            pack_to_col(stack, *count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Unpack(out_count) => {
            if *out_count > 0 {
                unpack(stack, *out_count)?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateMatrix(rows, cols) => {
            create_matrix(stack, *rows, *cols)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateMatrixDynamic(num_rows) => {
            create_matrix_dynamic(stack, *num_rows, |rows_data| async move {
                runmat_runtime::create_matrix_from_values(&rows_data).await
            })
            .await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateRange(has_step) => {
            create_range(stack, *has_step, |args| async move {
                runmat_runtime::call_builtin_async("colon", &args).await
            })
            .await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateCell2D(rows, cols) => {
            let mut elems = Vec::with_capacity(*rows * *cols);
            for _ in 0..(*rows * *cols) {
                elems.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            elems.reverse();
            stack.push(crate::ops::cells::create_cell_2d(elems, *rows, *cols)?);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateStructLiteral(fields) => {
            let values = pop_aggregate_literal_values(stack, fields.len())?;
            let mut struct_value = StructValue::new();
            for (field, value) in fields.iter().zip(values) {
                struct_value.fields.insert(field.clone(), value);
            }
            stack.push(Value::Struct(struct_value));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateObjectLiteral { class_name, fields } => {
            let values = pop_aggregate_literal_values(stack, fields.len())?;
            let mut object = initialize_object_with_defaults(class_name);
            for (field, value) in fields.iter().zip(values) {
                object.properties.insert(field.clone(), value);
            }
            stack.push(Value::Object(object));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallBuiltinMulti(name, arg_count, out_count) => {
            match handle_builtin_call_multi(
                calls::BuiltinCallContext {
                    stack,
                    name,
                    arg_count: *arg_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    imports: imports.as_slice(),
                    call_counts,
                    function_registry,
                    current_function_name,
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                *out_count,
                refresh_workspace_state,
            )
            .await?
            {
                BuiltinHandling::Completed => {}
                BuiltinHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                BuiltinHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallBuiltinMultiUsingOutputSlot(name, arg_count, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            match handle_builtin_call_multi(
                calls::BuiltinCallContext {
                    stack,
                    name,
                    arg_count: *arg_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    imports: imports.as_slice(),
                    call_counts,
                    function_registry,
                    current_function_name,
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                out_count,
                refresh_workspace_state,
            )
            .await?
            {
                BuiltinHandling::Completed => {}
                BuiltinHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                BuiltinHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSuperConstructorMulti {
            current_class,
            super_class,
            arg_count,
            out_count,
        } => {
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let result = runmat_runtime::call_super_constructor(
                current_class.clone(),
                super_class.clone(),
                args,
            )
            .await?;
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSuperMethodMulti {
            current_class,
            super_class,
            method,
            arg_count,
            out_count,
        } => {
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let result = runmat_runtime::call_super_method(
                current_class.clone(),
                super_class.clone(),
                method.clone(),
                args,
            )
            .await?;
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFevalMulti(argc, out_count) => {
            let args = crate::call::builtins::collect_call_args(stack, *argc)?;
            let func_val = crate::interpreter::stack::pop_value(stack)?;
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            match crate::call::feval::execute_feval(func_val, args, *out_count, function_registry)
                .await?
            {
                crate::call::feval::FevalDispatch::Completed(result) => {
                    stack.push(calls::normalize_requested_outputs(result, *out_count));
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFevalMultiUsingOutputSlot(argc, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            let args = crate::call::builtins::collect_call_args(stack, *argc)?;
            let func_val = crate::interpreter::stack::pop_value(stack)?;
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            match crate::call::feval::execute_feval(func_val, args, out_count, function_registry)
                .await?
            {
                crate::call::feval::FevalDispatch::Completed(result) => {
                    stack.push(calls::normalize_requested_outputs(result, out_count));
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFevalExpandMultiOutput(specs, out_count) => {
            let args = build_feval_expand_multi_args(stack, specs).await?;
            let func_val = crate::interpreter::stack::pop_value(stack)?;
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            match crate::call::feval::execute_feval(func_val, args, *out_count, function_registry)
                .await?
            {
                crate::call::feval::FevalDispatch::Completed(result) => {
                    stack.push(calls::normalize_requested_outputs(result, *out_count));
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFevalExpandMultiOutputUsingOutputSlot(specs, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            let args = build_feval_expand_multi_args(stack, specs).await?;
            let func_val = crate::interpreter::stack::pop_value(stack)?;
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            match crate::call::feval::execute_feval(func_val, args, out_count, function_registry)
                .await?
            {
                crate::call::feval::FevalDispatch::Completed(result) => {
                    stack.push(calls::normalize_requested_outputs(result, out_count));
                }
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateSemanticFuture(function, arg_count, out_count) => {
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            stack.push(create_async_future_value(*function, *out_count, args));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateSemanticFutureExpandMultiOutput(function, specs, out_count) => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            stack.push(create_async_future_value(*function, *out_count, args));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Spawn => {
            let value = stack.pop().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "spawn instruction expected a value on the stack",
                )
            })?;
            let value = resolve_semantic_future_value(value).await?;
            enforce_spawn_value_concurrency_policy(&value)?;
            stack.push(wrap_spawned_value(context, value));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Await => {
            let value = stack.pop().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "await instruction expected a value on the stack",
                )
            })?;
            let value = unwrap_spawned_value(context, value)?;
            let value = resolve_semantic_future_value(value).await?;
            stack.push(value);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticFunctionMulti(function, arg_count, out_count) => {
            match handle_user_function_call(
                calls::UserCallContext {
                    stack,
                    identity: runmat_hir::CallableIdentity::BoundFunction(*function),
                    fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                    out_count: *out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    current_function_name,
                    imports: imports.as_slice(),
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                *arg_count,
                refresh_workspace_state,
            )
            .await?
            {
                UserCallHandling::Completed => {}
                UserCallHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                UserCallHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticFunctionMultiUsingOutputSlot(function, arg_count, out_count_slot) => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            match handle_user_function_call(
                calls::UserCallContext {
                    stack,
                    identity: runmat_hir::CallableIdentity::BoundFunction(*function),
                    fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                    out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    current_function_name,
                    imports: imports.as_slice(),
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                *arg_count,
                refresh_workspace_state,
            )
            .await?
            {
                UserCallHandling::Completed => {}
                UserCallHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                UserCallHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticNestedFunctionMulti {
            function,
            capture_slots,
            arg_count,
            out_count,
        } => {
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let mut call_args = Vec::with_capacity(capture_slots.len() + args.len());
            for slot in capture_slots {
                call_args.push(vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)));
            }
            call_args.extend(args);
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            let (result, updated_captures) =
                crate::interpreter::runner::invoke_semantic_function_value_with_capture_updates(
                    function.0,
                    &call_args,
                    *out_count,
                    function_registry,
                )
                .await?;
            for (slot, value) in capture_slots.iter().zip(updated_captures.into_iter()) {
                if *slot < vars.len() {
                    vars[*slot] = value;
                }
            }
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticNestedFunctionMultiUsingOutputSlot {
            function,
            capture_slots,
            arg_count,
            out_count_slot,
        } => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            let args = crate::call::builtins::collect_call_args(stack, *arg_count)?;
            let mut call_args = Vec::with_capacity(capture_slots.len() + args.len());
            for slot in capture_slots {
                call_args.push(vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)));
            }
            call_args.extend(args);
            let _output_guard = runmat_runtime::output_context::push_output_count(out_count);
            let _function_input_callsite_guard =
                runmat_runtime::callsite::push_function_input_callsite(
                    source_id,
                    call_arg_spans.clone(),
                );
            let (result, updated_captures) =
                crate::interpreter::runner::invoke_semantic_function_value_with_capture_updates(
                    function.0,
                    &call_args,
                    out_count,
                    function_registry,
                )
                .await?;
            for (slot, value) in capture_slots.iter().zip(updated_captures.into_iter()) {
                if *slot < vars.len() {
                    vars[*slot] = value;
                }
            }
            stack.push(calls::normalize_requested_outputs(result, out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFunctionMulti {
            identity,
            fallback_policy,
            arg_count,
            out_count,
        } => {
            match handle_user_function_call(
                calls::UserCallContext {
                    stack,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    out_count: *out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    current_function_name,
                    imports: imports.as_slice(),
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                *arg_count,
                refresh_workspace_state,
            )
            .await?
            {
                UserCallHandling::Completed => {}
                UserCallHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                UserCallHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFunctionMultiUsingOutputSlot {
            identity,
            fallback_policy,
            arg_count,
            out_count_slot,
        } => {
            let out_count = requested_outputs_from_slot(vars.as_slice(), *out_count_slot)?;
            match handle_user_function_call(
                calls::UserCallContext {
                    stack,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    current_function_name,
                    imports: imports.as_slice(),
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                *arg_count,
                refresh_workspace_state,
            )
            .await?
            {
                UserCallHandling::Completed => {}
                UserCallHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                UserCallHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallBuiltinExpandMultiOutput(name, specs, out_count) => {
            let args = build_builtin_expand_multi_args(stack, specs).await?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let result =
                runmat_runtime::call_builtin_async_with_outputs(name, &args, *out_count).await?;
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSuperConstructorExpandMultiOutput {
            current_class,
            super_class,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let result = runmat_runtime::call_super_constructor(
                current_class.clone(),
                super_class.clone(),
                args,
            )
            .await?;
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSuperMethodExpandMultiOutput {
            current_class,
            super_class,
            method,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let result = runmat_runtime::call_super_method(
                current_class.clone(),
                super_class.clone(),
                method.clone(),
                args,
            )
            .await?;
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallFunctionExpandMultiOutput {
            identity,
            fallback_policy,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            match handle_prepared_user_function_call(
                calls::UserCallContext {
                    stack,
                    identity: identity.clone(),
                    fallback_policy: *fallback_policy,
                    out_count: *out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    current_function_name,
                    imports: imports.as_slice(),
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                args,
                refresh_workspace_state,
            )
            .await?
            {
                UserCallHandling::Completed => {}
                UserCallHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                UserCallHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticFunctionExpandMultiOutput(function, specs, out_count) => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            match handle_prepared_user_function_call(
                calls::UserCallContext {
                    stack,
                    identity: runmat_hir::CallableIdentity::BoundFunction(*function),
                    fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                    out_count: *out_count,
                    source_id,
                    call_arg_spans: call_arg_spans.clone(),
                    current_function_name,
                    imports: imports.as_slice(),
                    exception: calls::ExceptionRouteContext {
                        try_stack,
                        vars,
                        last_exception,
                        pc,
                    },
                },
                args,
                refresh_workspace_state,
            )
            .await?
            {
                UserCallHandling::Completed => {}
                UserCallHandling::Caught => {
                    return Ok(Some(DispatchHandled::Generic(
                        DispatchDecision::ContinueLoop,
                    )))
                }
                UserCallHandling::Uncaught(err) => return Err(*err),
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticNestedFunctionExpandMultiOutput {
            function,
            capture_slots,
            specs,
            out_count,
        } => {
            let args = build_user_function_expand_multi_args(stack, specs).await?;
            let mut call_args = Vec::with_capacity(capture_slots.len() + args.len());
            for slot in capture_slots {
                call_args.push(vars.get(*slot).cloned().unwrap_or(Value::Num(0.0)));
            }
            call_args.extend(args);
            let _output_guard = runmat_runtime::output_context::push_output_count(*out_count);
            let (result, updated_captures) =
                crate::interpreter::runner::invoke_semantic_function_value_with_capture_updates(
                    function.0,
                    &call_args,
                    *out_count,
                    function_registry,
                )
                .await?;
            for (slot, value) in capture_slots.iter().zip(updated_captures.into_iter()) {
                if *slot < vars.len() {
                    vars[*slot] = value;
                }
            }
            stack.push(calls::normalize_requested_outputs(result, *out_count));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallMethodOrMemberIndexMulti {
            identity,
            fallback_policy,
            arg_count,
            out_count,
        } => {
            handle_method_or_member_index_multi_call(
                stack,
                identity.clone(),
                *fallback_policy,
                *arg_count,
                *out_count,
                current_function_name,
            )
            .await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallMethodOrMemberIndexExpandMultiOutput {
            identity,
            fallback_policy,
            specs,
            out_count,
        } => {
            handle_method_or_member_index_expand_multi_call(
                stack,
                identity.clone(),
                *fallback_policy,
                specs,
                *out_count,
                current_function_name,
            )
            .await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadMethod(name) => {
            handle_load_method(stack, name.clone(), current_function_name)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateClosure(func_name, capture_count) => {
            handle_create_closure(stack, func_name.clone(), *capture_count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateSemanticClosure(function, display_name, capture_count) => {
            handle_create_semantic_closure(stack, *function, display_name.clone(), *capture_count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateFunctionHandle(name) => {
            stack.push(Value::FunctionHandle(name.clone()));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateExternalFunctionHandle(name) => {
            stack.push(Value::ExternalFunctionHandle(name.clone()));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateMethodFunctionHandle(name) => {
            stack.push(Value::MethodFunctionHandle(name.clone()));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateBoundFunctionHandle(function, name) => {
            stack.push(Value::BoundFunctionHandle {
                name: name.clone(),
                function: function.0,
            });
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadStaticProperty(class_name, prop) => {
            handle_load_static_property(stack, class_name, prop)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::RegisterClass {
            name,
            super_class,
            is_sealed,
            is_abstract,
            properties,
            methods,
            enumerations,
        } => {
            handle_register_class(
                name.clone(),
                super_class.clone(),
                *is_sealed,
                *is_abstract,
                properties.clone(),
                methods.clone(),
                enumerations.clone(),
            )?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        enforce_spawn_value_concurrency_policy, unwrap_spawned_value, wrap_spawned_value,
        SPAWN_TASK_ID_FIELD, SPAWN_TASK_KIND_FIELD, SPAWN_TASK_KIND_VALUE,
        SPAWN_TASK_PAYLOAD_FIELD,
    };
    use crate::bytecode::program::ExecutionContext;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorView,
        SpawnHandleConcurrency, ThreadProviderGuard,
    };
    use runmat_builtins::{CellArray, HandleRef, IntValue, StructValue, Value};

    struct RejectSpawnProvider;
    static REJECT_PROVIDER: RejectSpawnProvider = RejectSpawnProvider;

    impl AccelProvider for RejectSpawnProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(anyhow::anyhow!("unsupported"))
        }

        fn download<'a>(&'a self, _h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { Err(anyhow::anyhow!("unsupported")) })
        }

        fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "reject-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            41
        }
    }

    struct ShareSpawnProvider;
    static SHARE_PROVIDER: ShareSpawnProvider = ShareSpawnProvider;

    impl AccelProvider for ShareSpawnProvider {
        fn upload(&self, _host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            Err(anyhow::anyhow!("unsupported"))
        }

        fn download<'a>(&'a self, _h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { Err(anyhow::anyhow!("unsupported")) })
        }

        fn free(&self, _h: &GpuTensorHandle) -> anyhow::Result<()> {
            Ok(())
        }

        fn device_info(&self) -> String {
            "share-provider".to_string()
        }

        fn device_id(&self) -> u32 {
            42
        }

        fn spawn_handle_concurrency(&self) -> SpawnHandleConcurrency {
            SpawnHandleConcurrency::ImmutableShare
        }
    }

    #[test]
    fn spawn_policy_rejects_gpu_handles_when_provider_disallows_sharing() {
        let _provider_guard = ThreadProviderGuard::set(Some(&REJECT_PROVIDER));
        let value = Value::GpuTensor(GpuTensorHandle {
            shape: vec![1],
            device_id: 41,
            buffer_id: 7,
        });
        let err = enforce_spawn_value_concurrency_policy(&value)
            .expect_err("reject policy should block spawn capture");
        assert_eq!(
            err.identifier(),
            Some("RunMat:SpawnGpuHandleUnsupported"),
            "expected explicit spawn GPU-handle policy error identifier"
        );
    }

    #[test]
    fn spawn_policy_allows_gpu_handles_when_provider_declares_immutable_share() {
        let _provider_guard = ThreadProviderGuard::set(Some(&SHARE_PROVIDER));
        let value = Value::GpuTensor(GpuTensorHandle {
            shape: vec![1],
            device_id: 42,
            buffer_id: 9,
        });
        enforce_spawn_value_concurrency_policy(&value)
            .expect("immutable sharing policy should allow spawn capture");
    }

    #[test]
    fn spawn_policy_rejects_nested_gpu_handles_in_cell_capture() {
        let _provider_guard = ThreadProviderGuard::set(Some(&REJECT_PROVIDER));
        let nested_cell = CellArray::new(
            vec![
                Value::Num(1.0),
                Value::GpuTensor(GpuTensorHandle {
                    shape: vec![1],
                    device_id: 41,
                    buffer_id: 11,
                }),
            ],
            1,
            2,
        )
        .expect("construct test cell");
        let value = Value::Cell(nested_cell);
        let err = enforce_spawn_value_concurrency_policy(&value)
            .expect_err("reject policy should block nested GPU handle capture");
        assert_eq!(
            err.identifier(),
            Some("RunMat:SpawnGpuHandleUnsupported"),
            "expected nested capture rejection identifier"
        );
    }

    #[test]
    fn spawn_policy_reports_provider_unavailable_for_gpu_handles() {
        let _provider_guard = ThreadProviderGuard::set(None);
        let value = Value::GpuTensor(GpuTensorHandle {
            shape: vec![1],
            device_id: 99,
            buffer_id: 13,
        });
        let err = enforce_spawn_value_concurrency_policy(&value)
            .expect_err("missing provider should reject spawn GPU handle capture");
        assert_eq!(
            err.identifier(),
            Some("RunMat:SpawnProviderUnavailable"),
            "expected missing-provider spawn capture identifier"
        );
    }

    #[test]
    fn spawn_policy_rejects_gpu_handles_captured_by_closure_values() {
        let _provider_guard = ThreadProviderGuard::set(Some(&REJECT_PROVIDER));
        let value = Value::Closure(runmat_builtins::Closure {
            function_name: "worker".to_string(),
            bound_function: None,
            captures: vec![
                Value::Num(2.0),
                Value::GpuTensor(GpuTensorHandle {
                    shape: vec![1],
                    device_id: 41,
                    buffer_id: 21,
                }),
            ],
        });
        let err = enforce_spawn_value_concurrency_policy(&value)
            .expect_err("reject policy should block closure-captured GPU handles");
        assert_eq!(
            err.identifier(),
            Some("RunMat:SpawnGpuHandleUnsupported"),
            "expected closure-capture spawn policy identifier"
        );
    }

    #[test]
    fn spawn_policy_rejects_gpu_handles_nested_in_handle_object_target() {
        let _provider_guard = ThreadProviderGuard::set(Some(&REJECT_PROVIDER));
        let mut payload = StructValue::new();
        payload.fields.insert(
            "nested".to_string(),
            Value::GpuTensor(GpuTensorHandle {
                shape: vec![1],
                device_id: 41,
                buffer_id: 31,
            }),
        );
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let value = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        let err = enforce_spawn_value_concurrency_policy(&value)
            .expect_err("reject policy should block handle-object nested GPU handle capture");
        assert_eq!(
            err.identifier(),
            Some("RunMat:SpawnGpuHandleUnsupported"),
            "expected handle-object nested capture rejection identifier"
        );
    }

    #[test]
    fn spawn_value_wrap_roundtrips_through_await_unwrap() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let wrapped = wrap_spawned_value(&mut context, Value::Num(3.0));
        let unwrapped =
            unwrap_spawned_value(&mut context, wrapped).expect("await should unwrap spawn task");
        assert_eq!(unwrapped, Value::Num(3.0));
    }

    #[test]
    fn await_unwrap_passes_through_non_spawn_value() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let value = unwrap_spawned_value(&mut context, Value::Num(3.0))
            .expect("await should pass through non-task value");
        assert_eq!(value, Value::Num(3.0));
    }

    #[test]
    fn spawn_wrapper_uses_explicit_task_fields() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let wrapped = wrap_spawned_value(&mut context, Value::Num(5.0));
        let Value::Struct(task) = wrapped else {
            panic!("spawn should produce a struct-backed task handle");
        };
        assert_eq!(
            task.fields.get(SPAWN_TASK_KIND_FIELD),
            Some(&Value::String(SPAWN_TASK_KIND_VALUE.to_string()))
        );
        assert_eq!(
            task.fields.get(SPAWN_TASK_PAYLOAD_FIELD),
            Some(&Value::Num(5.0))
        );
        assert_eq!(
            task.fields.get(SPAWN_TASK_ID_FIELD),
            Some(&Value::Int(IntValue::U64(0)))
        );
    }

    #[test]
    fn await_unwrap_rejects_stale_spawn_task_id() {
        let mut wrap_context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let wrapped = wrap_spawned_value(&mut wrap_context, Value::Num(9.0));
        let mut await_context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let err = unwrap_spawned_value(&mut await_context, wrapped)
            .expect_err("await should reject stale/unregistered task ids");
        assert_eq!(err.identifier(), Some("RunMat:AwaitOperandInvalid"));
    }

    #[test]
    fn dropped_spawn_task_handle_retires_task_id() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let wrapped = wrap_spawned_value(&mut context, Value::Num(7.0));
        assert!(
            context.spawned_task_ids.contains(&0),
            "spawn should register task id before drop"
        );
        super::retire_spawn_task_id_if_dropped(&mut context, &wrapped, &[], &[], None, None);
        assert!(
            !context.spawned_task_ids.contains(&0),
            "dropping a spawn task handle should retire its task id"
        );
    }

    #[test]
    fn spawn_task_id_extraction_ignores_non_task_structs() {
        let mut non_task = StructValue::new();
        non_task
            .fields
            .insert("x".to_string(), Value::Int(IntValue::U64(9)));
        assert!(
            super::spawn_task_id_from_value(&Value::Struct(non_task)).is_none(),
            "only spawn task structs should expose task ids"
        );
    }

    #[test]
    fn replaced_spawn_task_id_is_retired_when_incoming_differs() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let current = wrap_spawned_value(&mut context, Value::Num(1.0));
        assert!(
            context.spawned_task_ids.contains(&0),
            "spawn should register task id before replacement"
        );
        super::retire_spawn_task_id_if_replaced(
            &mut context,
            &current,
            &Value::Num(2.0),
            &[],
            &[],
            None,
            None,
        );
        assert!(
            !context.spawned_task_ids.contains(&0),
            "replacing task handle with a non-task value should retire its task id"
        );
    }

    #[test]
    fn replacing_with_same_spawn_task_keeps_id_registered() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let current = wrap_spawned_value(&mut context, Value::Num(3.0));
        assert!(
            context.spawned_task_ids.contains(&0),
            "spawn should register task id before self-replacement"
        );
        super::retire_spawn_task_id_if_replaced(
            &mut context,
            &current,
            &current,
            &[],
            &[],
            None,
            None,
        );
        assert!(
            context.spawned_task_ids.contains(&0),
            "replacing a task handle with itself should keep the task id live"
        );
    }

    #[test]
    fn dropped_spawn_task_handle_keeps_id_when_alias_still_live() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let wrapped = wrap_spawned_value(&mut context, Value::Num(7.0));
        assert!(
            context.spawned_task_ids.contains(&0),
            "spawn should register task id before alias drop"
        );
        let vars = vec![wrapped.clone()];
        super::retire_spawn_task_id_if_dropped(&mut context, &wrapped, &[], &vars, None, None);
        assert!(
            context.spawned_task_ids.contains(&0),
            "dropping one alias should keep task id when another alias remains live"
        );
    }

    #[test]
    fn dropped_nested_spawn_task_handle_in_handle_object_retires_task_id() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let wrapped = wrap_spawned_value(&mut context, Value::Num(7.0));
        let mut payload = StructValue::new();
        payload.fields.insert("task".to_string(), wrapped.clone());
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let nested = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        assert!(
            context.spawned_task_ids.contains(&0),
            "spawn should register task id before nested drop"
        );
        super::retire_spawn_task_id_if_dropped(&mut context, &nested, &[], &[], None, None);
        assert!(
            !context.spawned_task_ids.contains(&0),
            "dropping nested spawn task handle should retire its task id"
        );
    }

    #[test]
    fn dropped_nested_spawn_task_handle_in_handle_object_keeps_id_when_alias_live() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let wrapped = wrap_spawned_value(&mut context, Value::Num(7.0));
        let mut payload = StructValue::new();
        payload.fields.insert("task".to_string(), wrapped.clone());
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let nested = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        let vars = vec![wrapped.clone()];
        super::retire_spawn_task_id_if_dropped(&mut context, &nested, &[], &vars, None, None);
        assert!(
            context.spawned_task_ids.contains(&0),
            "dropping nested alias should keep task id when direct alias remains live"
        );
    }

    #[test]
    fn replaced_nested_spawn_task_handle_in_handle_object_retires_task_id_when_unaliased() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let wrapped = wrap_spawned_value(&mut context, Value::Num(7.0));
        let mut payload = StructValue::new();
        payload.fields.insert("task".to_string(), wrapped);
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let current = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        assert!(
            context.spawned_task_ids.contains(&0),
            "spawn should register task id before nested replacement"
        );
        super::retire_spawn_task_id_if_replaced(
            &mut context,
            &current,
            &Value::Num(0.0),
            &[],
            &[],
            None,
            None,
        );
        assert!(
            !context.spawned_task_ids.contains(&0),
            "replacing nested task handle with non-task value should retire its task id"
        );
    }

    #[test]
    fn replaced_nested_spawn_task_handle_in_handle_object_keeps_id_when_alias_live() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let wrapped = wrap_spawned_value(&mut context, Value::Num(7.0));
        let mut payload = StructValue::new();
        payload.fields.insert("task".to_string(), wrapped.clone());
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let current = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        let vars = vec![wrapped];
        super::retire_spawn_task_id_if_replaced(
            &mut context,
            &current,
            &Value::Num(0.0),
            &[],
            &vars,
            None,
            None,
        );
        assert!(
            context.spawned_task_ids.contains(&0),
            "replacing nested alias should keep task id when a live alias remains"
        );
    }

    #[test]
    fn replaced_nested_spawn_task_handle_in_local_slot_retires_with_excluded_local() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let wrapped = wrap_spawned_value(&mut context, Value::Num(7.0));
        let mut payload = StructValue::new();
        payload.fields.insert("task".to_string(), wrapped);
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let current = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        context.locals.push(current.clone());
        super::retire_spawn_task_id_if_replaced(
            &mut context,
            &current,
            &Value::Num(0.0),
            &[],
            &[],
            None,
            Some(0),
        );
        assert!(
            !context.spawned_task_ids.contains(&0),
            "local-slot replacement should retire nested task id when excluded local is the only alias"
        );
    }

    #[test]
    fn replaced_nested_spawn_task_handle_in_local_slot_keeps_id_when_other_local_alias_live() {
        let mut context = ExecutionContext {
            call_stack: Vec::new(),
            locals: Vec::new(),
            instruction_pointer: 0,
            spawned_task_ids: std::collections::HashSet::new(),
            next_spawn_task_id: 0,
        };
        let wrapped = wrap_spawned_value(&mut context, Value::Num(7.0));
        let mut payload = StructValue::new();
        payload.fields.insert("task".to_string(), wrapped.clone());
        let target = runmat_gc::gc_allocate(Value::Struct(payload)).expect("gc allocate payload");
        let current = Value::HandleObject(HandleRef {
            class_name: "Payload".to_string(),
            target,
            valid: true,
        });
        context.locals.push(current.clone());
        context.locals.push(wrapped);
        super::retire_spawn_task_id_if_replaced(
            &mut context,
            &current,
            &Value::Num(0.0),
            &[],
            &[],
            None,
            Some(0),
        );
        assert!(
            context.spawned_task_ids.contains(&0),
            "local-slot replacement should keep nested task id when another local alias remains live"
        );
    }
}
