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
use crate::runtime::workspace::refresh_workspace_state;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::Value;
use runmat_runtime::dispatcher::gather_if_needed_async;
use runmat_runtime::RuntimeError;
use std::collections::HashMap;

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
    pub semantic_registry: &'a crate::bytecode::SemanticFunctionRegistry,
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
    pub pc: &'a mut usize,
}

pub struct DispatchHooks<'a> {
    pub clear_value_residency: &'a mut dyn FnMut(&Value),
    pub store_var_before_overwrite: &'a mut dyn FnMut(&Value, &Value),
    pub store_var_after_store: &'a mut dyn FnMut(usize, &Value),
    pub store_local_before_local_overwrite: &'a mut dyn FnMut(&Value, &Value),
    pub store_local_before_var_overwrite: &'a mut dyn FnMut(&Value, &Value),
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

fn for_each_gpu_handle_in_value(
    value: &Value,
    f: &mut impl FnMut(&GpuTensorHandle) -> Result<(), RuntimeError>,
) -> Result<(), RuntimeError> {
    match value {
        Value::GpuTensor(handle) => f(handle),
        Value::Cell(cell) => {
            for elem in &cell.data {
                for_each_gpu_handle_in_value(elem, f)?;
            }
            Ok(())
        }
        Value::Struct(struct_value) => {
            for elem in struct_value.fields.values() {
                for_each_gpu_handle_in_value(elem, f)?;
            }
            Ok(())
        }
        Value::Object(object_value) => {
            for elem in object_value.properties.values() {
                for_each_gpu_handle_in_value(elem, f)?;
            }
            Ok(())
        }
        Value::Closure(closure) => {
            for capture in &closure.captures {
                for_each_gpu_handle_in_value(capture, f)?;
            }
            Ok(())
        }
        Value::OutputList(values) => {
            for elem in values {
                for_each_gpu_handle_in_value(elem, f)?;
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
        | Value::ComplexTensor(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::SemanticFunctionHandle { .. }
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

pub async fn dispatch_instruction(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
    hooks: DispatchHooks<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let DispatchMeta {
        instr,
        var_names,
        semantic_registry,
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
        pc,
    } = state;
    let DispatchHooks {
        clear_value_residency,
        store_var_before_overwrite,
        store_var_after_store,
        store_local_before_local_overwrite,
        store_local_before_var_overwrite,
        store_local_after_fallback_store,
    } = hooks;
    match instr {
        _ if indexing::dispatch_indexing(
            instr,
            stack,
            vars,
            semantic_registry,
            *pc,
            &mut *clear_value_residency,
        )
        .await? =>
        {
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ if object::dispatch_object(instr, stack).await? => Ok(Some(DispatchHandled::Generic(
            DispatchDecision::FallThrough,
        ))),
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
            store_var(
                stack,
                vars,
                *index,
                var_names,
                store_var_before_overwrite,
                store_var_after_store,
            )?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::StoreLocal(offset) => {
            store_local(
                stack,
                context,
                vars,
                *offset,
                store_local_before_local_overwrite,
                store_local_before_var_overwrite,
                store_local_after_fallback_store,
            )?;
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
            crate::ops::stack::pop(stack);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::AndAnd(target) => Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
            crate::ops::control_flow::and_and(stack, *target)?,
            pc,
        )))),
        Instr::OrOr(target) => Ok(Some(DispatchHandled::Generic(apply_control_flow_action(
            crate::ops::control_flow::or_or(stack, *target)?,
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
            crate::ops::control_flow::exit_scope(&mut context.locals, *local_count, |val| {
                clear_value_residency(val);
            });
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
        Instr::CallFevalMulti(argc, out_count) => {
            let args = crate::call::builtins::collect_call_args(stack, *argc)?;
            let func_val = crate::interpreter::stack::pop_value(stack)?;
            match crate::call::feval::execute_feval(func_val, args, *out_count, semantic_registry)
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
        Instr::CallFevalExpandMultiOutput(specs, out_count) => {
            let args = build_feval_expand_multi_args(stack, specs).await?;
            let func_val = crate::interpreter::stack::pop_value(stack)?;
            match crate::call::feval::execute_feval(func_val, args, *out_count, semantic_registry)
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
        Instr::Spawn => {
            if stack.is_empty() {
                return Err(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "spawn instruction expected a value on the stack",
                ));
            }
            if let Some(top) = stack.last() {
                enforce_spawn_value_concurrency_policy(top)?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Await => {
            if stack.is_empty() {
                return Err(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "await instruction expected a value on the stack",
                ));
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CallSemanticFunctionMulti(function, arg_count, out_count) => {
            match handle_user_function_call(
                calls::UserCallContext {
                    stack,
                    identity: runmat_hir::CallableIdentity::SemanticFunction(*function),
                    fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                    out_count: *out_count,
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
                    identity: runmat_hir::CallableIdentity::SemanticFunction(*function),
                    fallback_policy: runmat_hir::CallableFallbackPolicy::None,
                    out_count: *out_count,
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
            )
            .await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::LoadMethod(name) => {
            handle_load_method(stack, name.clone())?;
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
        Instr::CreateSemanticFunctionHandle(function, name) => {
            stack.push(Value::SemanticFunctionHandle {
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
            properties,
            methods,
        } => {
            handle_register_class(
                name.clone(),
                super_class.clone(),
                properties.clone(),
                methods.clone(),
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
    use super::enforce_spawn_value_concurrency_policy;
    use runmat_accelerate_api::{
        AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorView,
        SpawnHandleConcurrency, ThreadProviderGuard,
    };
    use runmat_builtins::Value;

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
}
