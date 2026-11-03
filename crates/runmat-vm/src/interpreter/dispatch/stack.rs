use crate::bytecode::program::ExecutionContext;
use crate::bytecode::EmitLabel;
use crate::interpreter::vars::InterpreterVars;
use crate::ops::stack as stack_ops;
use runmat_runtime::RuntimeError;
use runmat_value::Value;
use std::collections::{HashMap, HashSet};

/// One ordinary binding-store owner, shared by bytecode StoreVar and a
/// selected source-loop checkpoint. The supplied value is already rooted by
/// the caller until this synchronous publication completes.
pub(crate) struct BindingPublicationFrame<'a> {
    pub vars: &'a mut InterpreterVars,
    pub context: &'a mut ExecutionContext,
    pub var_names: &'a HashMap<usize, String>,
    pub persistent_aliases: &'a mut crate::interpreter::persistent::PersistentAliases,
    pub missing_input_slots: &'a mut HashSet<usize>,
    pub current_function_name: &'a str,
}

pub(crate) fn publish_var_value<BeforeOverwrite, AfterStore>(
    value: Value,
    index: usize,
    frame: BindingPublicationFrame<'_>,
    before_overwrite: BeforeOverwrite,
    after_store: AfterStore,
) -> Result<(), RuntimeError>
where
    BeforeOverwrite: FnMut(usize),
    AfterStore: FnMut(usize),
{
    let BindingPublicationFrame {
        vars,
        context,
        var_names,
        persistent_aliases,
        missing_input_slots,
        current_function_name,
    } = frame;
    stack_ops::store_var_value(value, vars, index, var_names, before_overwrite, after_store)?;
    missing_input_slots.remove(&index);
    crate::interpreter::persistent::publish_store(
        persistent_aliases,
        vars,
        &context.runtime.session_variable_authority(),
        current_function_name,
        index,
    )?;
    super::lexical::publish(context, vars, index)
}

pub async fn emit_stack_top(
    context: &ExecutionContext,
    stack: &[Value],
    label: &EmitLabel,
    var_names: &HashMap<usize, String>,
) -> Result<(), RuntimeError> {
    stack_ops::emit_stack_top(&context.runtime, stack, label, var_names).await
}

pub async fn emit_var(
    context: &ExecutionContext,
    vars: &InterpreterVars,
    var_index: usize,
    label: &EmitLabel,
    var_names: &HashMap<usize, String>,
) -> Result<(), RuntimeError> {
    stack_ops::emit_var(&context.runtime, vars, var_index, label, var_names).await
}

pub fn load_const(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    value: f64,
) -> Result<(), RuntimeError> {
    stack_ops::load_const(stack, value)
}

pub fn load_int(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    value: runmat_value::IntValue,
) -> Result<(), RuntimeError> {
    stack_ops::load_int(stack, value)
}

pub fn load_complex(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    re: f64,
    im: f64,
) -> Result<(), RuntimeError> {
    stack_ops::load_complex(stack, re, im)
}

pub fn load_bool(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    value: bool,
) -> Result<(), RuntimeError> {
    stack_ops::load_bool(stack, value)
}

pub fn load_string(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    value: String,
) -> Result<(), RuntimeError> {
    stack_ops::load_string(stack, value)
}

pub fn load_char_row(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    value: String,
) -> Result<(), RuntimeError> {
    stack_ops::load_char_row(stack, value)
}

pub fn load_var(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    vars: &InterpreterVars,
    index: usize,
) -> Result<(), RuntimeError> {
    stack_ops::load_var(stack, vars, index)
}

pub fn load_local(
    stack: &mut crate::interpreter::stack::InterpreterStack,
    context: &ExecutionContext,
    vars: &InterpreterVars,
    offset: usize,
) -> Result<(), RuntimeError> {
    stack_ops::load_local(stack, context, vars, offset)
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
    stack_ops::store_var(stack, vars, index, var_names, before_overwrite, after_store)
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
    stack_ops::store_local(
        stack,
        context,
        vars,
        offset,
        before_local_overwrite,
        before_var_overwrite,
        after_local_store,
        after_fallback_store,
    )
}
