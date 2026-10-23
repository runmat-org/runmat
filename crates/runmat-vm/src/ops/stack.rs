use crate::bytecode::program::ExecutionContext;
use crate::bytecode::EmitLabel;
use crate::interpreter::stack::{pop2, pop_value};
use crate::interpreter::vars::InterpreterVars;
use crate::runtime::workspace::{admit_declared_global_slot_write, workspace_slot_value};
use runmat_runtime::RuntimeError;
use runmat_value::{CharArray, Value};
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
        EmitLabel::InternalCapture => None,
    }
}

pub async fn emit_stack_top(
    runtime: &runmat_runtime::context::RuntimeContext,
    stack: &[Value],
    label: &EmitLabel,
    var_names: &HashMap<usize, String>,
) -> Result<(), RuntimeError> {
    if matches!(label, EmitLabel::InternalCapture) {
        return Ok(());
    }
    if let Some(value) = stack.last() {
        let label_text = resolve_emit_label_text(label, var_names);
        runmat_runtime::console::emit_statement_value(runtime, label_text.as_deref(), value)
            .await?;
    }
    Ok(())
}

pub async fn emit_var(
    runtime: &runmat_runtime::context::RuntimeContext,
    vars: &InterpreterVars,
    var_index: usize,
    label: &EmitLabel,
    var_names: &HashMap<usize, String>,
) -> Result<(), RuntimeError> {
    if matches!(label, EmitLabel::InternalCapture) {
        return Ok(());
    }
    let value = vars.get(var_index).map(|value| value.clone());
    if let Some(value) = value {
        let label_text = resolve_emit_label_text(label, var_names);
        runmat_runtime::console::emit_statement_value(runtime, label_text.as_deref(), &value)
            .await?;
    }
    Ok(())
}

#[inline]
pub fn load_const(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    value: f64,
) -> Result<(), RuntimeError> {
    stack.try_push(Value::Num(value))
}

#[inline]
pub fn load_int(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    value: runmat_value::IntValue,
) -> Result<(), RuntimeError> {
    stack.try_push(Value::Int(value))
}

#[inline]
pub fn load_complex(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    re: f64,
    im: f64,
) -> Result<(), RuntimeError> {
    stack.try_push(Value::Complex(re, im))
}

#[inline]
pub fn load_bool(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    value: bool,
) -> Result<(), RuntimeError> {
    stack.try_push(Value::Bool(value))
}

#[inline]
pub fn load_string(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    value: String,
) -> Result<(), RuntimeError> {
    stack.try_push(Value::String(value.into()))
}

pub fn load_char_row(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    value: String,
) -> Result<(), RuntimeError> {
    let ca = CharArray::new_row(&value);
    stack.try_push(Value::CharArray(ca))
}

pub fn load_local(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    context: &ExecutionContext,
    vars: &InterpreterVars,
    offset: usize,
) -> Result<(), RuntimeError> {
    if let Some(current_frame) = context.call_stack.last() {
        let local_index = current_frame.locals_start + offset;
        if local_index >= context.locals.len() {
            return Err("Local variable index out of bounds".to_string().into());
        }
        stack.try_push(context.locals[local_index].clone())?;
    } else if offset < vars.len() {
        stack.try_push(
            vars.get(offset)
                .expect("local fallback slot lies within frame extent")
                .clone(),
        )?;
    } else {
        stack.try_push(Value::Num(0.0))?;
    }
    Ok(())
}

#[inline]
pub fn load_var(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    vars: &InterpreterVars,
    index: usize,
) -> Result<(), RuntimeError> {
    stack.try_push(workspace_slot_value(index).unwrap_or_else(|| {
        vars.get(index)
            .expect("loaded variable slot lies within frame extent")
            .clone()
    }))
}

pub fn store_var<BeforeOverwrite, AfterStore>(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    vars: &mut InterpreterVars,
    index: usize,
    var_names: &HashMap<usize, String>,
    before_overwrite: BeforeOverwrite,
    after_store: AfterStore,
) -> Result<(), RuntimeError>
where
    BeforeOverwrite: FnMut(usize),
    AfterStore: FnMut(usize),
{
    let value = pop_value(stack)?;
    store_var_value(value, vars, index, var_names, before_overwrite, after_store)
}

pub fn store_var_value<BeforeOverwrite, AfterStore>(
    value: Value,
    vars: &mut InterpreterVars,
    index: usize,
    var_names: &HashMap<usize, String>,
    mut before_overwrite: BeforeOverwrite,
    mut after_store: AfterStore,
) -> Result<(), RuntimeError>
where
    BeforeOverwrite: FnMut(usize),
    AfterStore: FnMut(usize),
{
    if index < vars.len() {
        before_overwrite(index);
    }
    crate::runtime::workspace::commit_workspace_slot_value(
        vars,
        index,
        var_names.get(&index).map(String::as_str),
        value,
    )?;
    after_store(index);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn store_local<BeforeLocalOverwrite, BeforeVarOverwrite, AfterLocalStore, AfterFallbackStore>(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    context: &mut ExecutionContext,
    vars: &mut InterpreterVars,
    offset: usize,
    before_local_overwrite: BeforeLocalOverwrite,
    before_var_overwrite: BeforeVarOverwrite,
    after_local_store: AfterLocalStore,
    after_fallback_store: AfterFallbackStore,
) -> Result<(), RuntimeError>
where
    BeforeLocalOverwrite: FnMut(&Value, &Value),
    BeforeVarOverwrite: FnMut(usize),
    AfterLocalStore: FnMut(usize, &Value),
    AfterFallbackStore: FnMut(&str, usize),
{
    let value = pop_value(stack)?;
    store_local_value(
        value,
        context,
        vars,
        offset,
        StoreLocalHooks {
            before_local_overwrite,
            before_var_overwrite,
            after_local_store,
            after_fallback_store,
        },
    )
}

pub struct StoreLocalHooks<
    BeforeLocalOverwrite,
    BeforeVarOverwrite,
    AfterLocalStore,
    AfterFallbackStore,
> {
    pub before_local_overwrite: BeforeLocalOverwrite,
    pub before_var_overwrite: BeforeVarOverwrite,
    pub after_local_store: AfterLocalStore,
    pub after_fallback_store: AfterFallbackStore,
}

pub fn store_local_value<
    BeforeLocalOverwrite,
    BeforeVarOverwrite,
    AfterLocalStore,
    AfterFallbackStore,
>(
    value: Value,
    context: &mut ExecutionContext,
    vars: &mut InterpreterVars,
    offset: usize,
    mut hooks: StoreLocalHooks<
        BeforeLocalOverwrite,
        BeforeVarOverwrite,
        AfterLocalStore,
        AfterFallbackStore,
    >,
) -> Result<(), RuntimeError>
where
    BeforeLocalOverwrite: FnMut(&Value, &Value),
    BeforeVarOverwrite: FnMut(usize),
    AfterLocalStore: FnMut(usize, &Value),
    AfterFallbackStore: FnMut(&str, usize),
{
    if let Some(current_frame) = context.call_stack.last() {
        let local_index = current_frame.locals_start + offset;
        admit_declared_global_slot_write(local_index, &value)?;
        while context.locals.len() <= local_index {
            context.locals.push(Value::Num(0.0));
        }
        (hooks.before_local_overwrite)(&context.locals[local_index], &value);
        context.locals[local_index] = value;
        (hooks.after_local_store)(local_index, &context.locals[local_index]);
    } else {
        if offset >= vars.len() {
            vars.resize_owned(offset + 1, Value::Num(0.0));
        }
        (hooks.before_var_overwrite)(offset);
        crate::runtime::workspace::commit_workspace_slot_value(vars, offset, None, value)?;
        let func_name = context
            .call_stack
            .last()
            .map(|f| f.function_name.as_str())
            .unwrap_or("<main>");
        (hooks.after_fallback_store)(func_name, offset);
    }
    Ok(())
}

#[inline]
pub fn swap(stack: &mut crate::interpreter::stack::InterpreterStack) -> Result<(), RuntimeError> {
    let (b, a) = pop2(stack)?;
    stack.try_push(a)?;
    stack.try_push(b)?;
    Ok(())
}
