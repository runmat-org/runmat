use crate::bytecode::EmitLabel;
use crate::bytecode::ExecutionContext;
use crate::interpreter::errors::mex;
use crate::interpreter::stack::{pop2, pop_value};
use crate::runtime::workspace::{
    ensure_workspace_slot_name, mark_workspace_assigned, refresh_workspace_state,
};
use runmat_builtins::{CharArray, Value};
use runmat_runtime::{dispatcher::gather_if_needed_async, RuntimeError};
use std::collections::HashMap;

fn resolve_emit_label_text(
    label: &EmitLabel,
    var_names: &HashMap<usize, String>,
) -> Option<String> {
    match label {
        EmitLabel::Ans => Some("ans".to_string()),
        EmitLabel::Var(idx) => var_names
            .get(idx)
            .cloned()
            .or_else(|| Some(format!("var{idx}"))),
    }
}

pub async fn emit_stack_top(
    stack: &[Value],
    label: &EmitLabel,
    var_names: &HashMap<usize, String>,
) -> Result<(), RuntimeError> {
    if let Some(value) = stack.last() {
        let label_text = resolve_emit_label_text(label, var_names);
        let host_value = gather_if_needed_async(value).await?;
        runmat_runtime::console::record_value_output(label_text.as_deref(), &host_value);
    }
    Ok(())
}

pub async fn emit_var(
    vars: &[Value],
    var_index: usize,
    label: &EmitLabel,
    var_names: &HashMap<usize, String>,
) -> Result<(), RuntimeError> {
    if let Some(value) = vars.get(var_index) {
        let label_text = resolve_emit_label_text(label, var_names);
        let host_value = gather_if_needed_async(value).await?;
        runmat_runtime::console::record_value_output(label_text.as_deref(), &host_value);
    }
    Ok(())
}

#[inline]
pub fn load_const(stack: &mut Vec<Value>, value: f64) {
    stack.push(Value::Num(value));
}

#[inline]
pub fn load_complex(stack: &mut Vec<Value>, re: f64, im: f64) {
    stack.push(Value::Complex(re, im));
}

#[inline]
pub fn load_bool(stack: &mut Vec<Value>, value: bool) {
    stack.push(Value::Bool(value));
}

#[inline]
pub fn load_string(stack: &mut Vec<Value>, value: String) {
    stack.push(Value::String(value));
}

pub fn load_char_row(stack: &mut Vec<Value>, value: String) -> Result<(), RuntimeError> {
    let ca = CharArray::new(value.chars().collect(), 1, value.chars().count())
        .map_err(|e| mex("CharError", &e))?;
    stack.push(Value::CharArray(ca));
    Ok(())
}

pub fn load_local(
    stack: &mut Vec<Value>,
    context: &ExecutionContext,
    vars: &[Value],
    offset: usize,
) -> Result<(), RuntimeError> {
    if let Some(current_frame) = context.call_stack.last() {
        let local_index = current_frame.locals_start + offset;
        if local_index >= context.locals.len() {
            return Err("Local variable index out of bounds".to_string().into());
        }
        stack.push(context.locals[local_index].clone());
    } else if offset < vars.len() {
        stack.push(vars[offset].clone());
    } else {
        stack.push(Value::Num(0.0));
    }
    Ok(())
}

#[inline]
pub fn load_var(stack: &mut Vec<Value>, vars: &[Value], index: usize) {
    stack.push(vars[index].clone());
}

pub fn store_var<BeforeOverwrite, AfterStore>(
    stack: &mut Vec<Value>,
    vars: &mut Vec<Value>,
    index: usize,
    var_names: &HashMap<usize, String>,
    mut before_overwrite: BeforeOverwrite,
    mut after_store: AfterStore,
) -> Result<(), RuntimeError>
where
    BeforeOverwrite: FnMut(&Value, &Value),
    AfterStore: FnMut(usize, &Value),
{
    let value = pop_value(stack)?;
    if index < vars.len() {
        before_overwrite(&vars[index], &value);
    }
    if index >= vars.len() {
        vars.resize(index + 1, Value::Num(0.0));
        refresh_workspace_state(vars);
    }
    vars[index] = value;
    if let Some(name) = var_names.get(&index) {
        ensure_workspace_slot_name(index, name);
    }
    mark_workspace_assigned(index);
    after_store(index, &vars[index]);
    Ok(())
}

pub fn store_local<BeforeLocalOverwrite, BeforeVarOverwrite, AfterFallbackStore>(
    stack: &mut Vec<Value>,
    context: &mut ExecutionContext,
    vars: &mut Vec<Value>,
    offset: usize,
    mut before_local_overwrite: BeforeLocalOverwrite,
    mut before_var_overwrite: BeforeVarOverwrite,
    mut after_fallback_store: AfterFallbackStore,
) -> Result<(), RuntimeError>
where
    BeforeLocalOverwrite: FnMut(&Value, &Value),
    BeforeVarOverwrite: FnMut(&Value, &Value),
    AfterFallbackStore: FnMut(&str, usize, &Value),
{
    let value = pop_value(stack)?;
    if let Some(current_frame) = context.call_stack.last() {
        let local_index = current_frame.locals_start + offset;
        while context.locals.len() <= local_index {
            context.locals.push(Value::Num(0.0));
        }
        before_local_overwrite(&context.locals[local_index], &value);
        context.locals[local_index] = value;
    } else {
        if offset >= vars.len() {
            vars.resize(offset + 1, Value::Num(0.0));
            refresh_workspace_state(vars);
        }
        before_var_overwrite(&vars[offset], &value);
        vars[offset] = value;
        let func_name = context
            .call_stack
            .last()
            .map(|f| f.function_name.as_str())
            .unwrap_or("<main>");
        after_fallback_store(func_name, offset, &vars[offset]);
    }
    Ok(())
}

#[inline]
pub fn pop(stack: &mut Vec<Value>) {
    let _ = stack.pop();
}

#[inline]
pub fn swap(stack: &mut Vec<Value>) -> Result<(), RuntimeError> {
    let (b, a) = pop2(stack)?;
    stack.push(a);
    stack.push(b);
    Ok(())
}
