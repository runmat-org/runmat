mod calls;
mod control_flow;
mod arithmetic;
mod arrays;
mod exceptions;
mod indexing;
mod object;
mod stack;

use crate::bytecode::Instr;
use crate::interpreter::debug;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;
use runmat_runtime::dispatcher::gather_if_needed_async;
use std::collections::HashMap;

pub use calls::{
    build_builtin_expand_at_args, build_builtin_expand_last_args, build_builtin_expand_multi_args,
    build_feval_expand_multi_args, handle_builtin_outcome, handle_feval_dispatch,
    handle_create_closure, handle_load_method, handle_load_static_property, handle_method_call,
    handle_method_or_member_index_call, handle_register_class, handle_static_method_call,
    output_list_for_user_call, push_single_result, prepare_named_user_dispatch,
    push_user_call_outputs, unpack_prepared_user_call, BuiltinHandling, FevalHandling,
    MethodHandling, PreparedUserDispatch,
};
pub use arrays::{create_matrix, create_matrix_dynamic, create_range, pack_to_col, pack_to_row, unpack};
pub use control_flow::{apply_control_flow_action, DispatchDecision};
pub use exceptions::{
    parse_exception, prepare_vm_error, redirect_exception_to_catch, ExceptionHandling,
};
pub use stack::{
    emit_stack_top, emit_var, load_bool, load_char_row, load_complex, load_const, load_local,
    load_string, load_var, store_local, store_var,
};

pub enum DispatchHandled {
    Generic(DispatchDecision),
    ReturnValue(DispatchDecision),
    Return(DispatchDecision),
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

pub async fn dispatch_instruction(
    instr: &Instr,
    stack: &mut Vec<Value>,
    vars: &mut Vec<Value>,
    var_names: &HashMap<usize, String>,
    context: &mut crate::bytecode::ExecutionContext,
    try_stack: &mut Vec<(usize, Option<usize>)>,
    imports: &mut Vec<(Vec<String>, bool)>,
    current_function_name: &str,
    global_aliases: &mut HashMap<usize, String>,
    persistent_aliases: &mut HashMap<usize, String>,
    pc: &mut usize,
    mut clear_value_residency: impl FnMut(&Value),
    mut store_var_before_overwrite: impl FnMut(&Value, &Value),
    mut store_var_after_store: impl FnMut(usize, &Value),
    mut store_local_before_local_overwrite: impl FnMut(&Value, &Value),
    mut store_local_before_var_overwrite: impl FnMut(&Value, &Value),
    mut store_local_after_fallback_store: impl FnMut(&str, usize, &Value),
) -> Result<Option<DispatchHandled>, RuntimeError> {
    match instr {
        _ if indexing::dispatch_indexing(instr, stack, *pc, &mut clear_value_residency).await? => {
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        _ if object::dispatch_object(instr, stack).await? => {
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        _ if arithmetic::dispatch_arithmetic(instr, stack).await? => {
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::EmitStackTop { label } => {
            emit_stack_top(stack, label, var_names).await?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::EmitVar { var_index, label } => {
            emit_var(vars, *var_index, label, var_names).await?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::LoadConst(value) => {
            load_const(stack, *value);
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::LoadComplex(re, im) => {
            load_complex(stack, *re, *im);
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::LoadBool(value) => {
            load_bool(stack, *value);
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::LoadString(value) => {
            load_string(stack, value.clone());
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::LoadCharRow(value) => {
            load_char_row(stack, value.clone())?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::LoadLocal(offset) => {
            load_local(stack, context, vars, *offset)?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::LoadVar(index) => {
            let value = vars[*index].clone();
            debug::trace_load_var(*pc, *index, &value);
            load_var(stack, vars, *index);
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::StoreVar(index) => {
            let preview = stack
                .last()
                .cloned()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
            debug::trace_store_var(*pc, *index, &preview);
            store_var(
                stack,
                vars,
                *index,
                var_names,
                &mut store_var_before_overwrite,
                &mut store_var_after_store,
            )?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::StoreLocal(offset) => {
            store_local(
                stack,
                context,
                vars,
                *offset,
                &mut store_local_before_local_overwrite,
                &mut store_local_before_var_overwrite,
                &mut store_local_after_fallback_store,
            )?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::Swap => {
            crate::ops::stack::swap(stack)?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::Pop => {
            crate::ops::stack::pop(stack);
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
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
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::PopTry => {
            crate::ops::control_flow::pop_try(try_stack);
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::ReturnValue => Ok(Some(DispatchHandled::ReturnValue(
            apply_control_flow_action(crate::ops::control_flow::return_value(stack)?, pc),
        ))),
        Instr::Return => Ok(Some(DispatchHandled::Return(
            apply_control_flow_action(crate::ops::control_flow::return_void(), pc),
        ))),
        Instr::EnterScope(local_count) => {
            crate::ops::control_flow::enter_scope(&mut context.locals, *local_count);
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::ExitScope(local_count) => {
            crate::ops::control_flow::exit_scope(&mut context.locals, *local_count, |_val| {});
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::RegisterImport { path, wildcard } => {
            imports.push((path.clone(), *wildcard));
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::DeclareGlobal(indices) => {
            crate::runtime::globals::declare_global(indices.clone(), vars);
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::DeclareGlobalNamed(indices, names) => {
            crate::runtime::globals::declare_global_named(
                indices.clone(),
                names.clone(),
                vars,
                global_aliases,
            );
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::DeclarePersistent(indices) => {
            crate::runtime::globals::declare_persistent(current_function_name, indices.clone(), vars);
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::DeclarePersistentNamed(indices, names) => {
            crate::runtime::globals::declare_persistent_named(
                current_function_name,
                indices.clone(),
                names.clone(),
                vars,
                persistent_aliases,
            );
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::PackToRow(count) => {
            pack_to_row(stack, *count)?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::PackToCol(count) => {
            pack_to_col(stack, *count)?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::Unpack(out_count) => {
            if *out_count > 0 {
                unpack(stack, *out_count)?;
            }
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::CreateMatrix(rows, cols) => {
            create_matrix(stack, *rows, *cols)?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::CreateMatrixDynamic(num_rows) => {
            create_matrix_dynamic(stack, *num_rows, |rows_data| async move {
                runmat_runtime::create_matrix_from_values(&rows_data).await
            })
            .await?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        Instr::CreateRange(has_step) => {
            create_range(stack, *has_step, |args| async move {
                runmat_runtime::call_builtin_async("colon", &args).await
            })
            .await?;
            Ok(Some(DispatchHandled::Generic(DispatchDecision::FallThrough)))
        }
        _ => Ok(None),
    }
}
