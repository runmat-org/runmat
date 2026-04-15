use crate::bytecode::EmitLabel;
use crate::bytecode::ExecutionContext;
use crate::ops::stack as stack_ops;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;
use std::collections::HashMap;

pub async fn emit_stack_top(
    stack: &[Value],
    label: &EmitLabel,
    var_names: &HashMap<usize, String>,
) -> Result<(), RuntimeError> {
    stack_ops::emit_stack_top(stack, label, var_names).await
}

pub async fn emit_var(
    vars: &[Value],
    var_index: usize,
    label: &EmitLabel,
    var_names: &HashMap<usize, String>,
) -> Result<(), RuntimeError> {
    stack_ops::emit_var(vars, var_index, label, var_names).await
}

pub fn load_const(stack: &mut Vec<Value>, value: f64) {
    stack_ops::load_const(stack, value);
}

pub fn load_complex(stack: &mut Vec<Value>, re: f64, im: f64) {
    stack_ops::load_complex(stack, re, im);
}

pub fn load_bool(stack: &mut Vec<Value>, value: bool) {
    stack_ops::load_bool(stack, value);
}

pub fn load_string(stack: &mut Vec<Value>, value: String) {
    stack_ops::load_string(stack, value);
}

pub fn load_char_row(stack: &mut Vec<Value>, value: String) -> Result<(), RuntimeError> {
    stack_ops::load_char_row(stack, value)
}

pub fn load_var(stack: &mut Vec<Value>, vars: &[Value], index: usize) {
    stack_ops::load_var(stack, vars, index);
}

pub fn load_local(
    stack: &mut Vec<Value>,
    context: &ExecutionContext,
    vars: &[Value],
    offset: usize,
) -> Result<(), RuntimeError> {
    stack_ops::load_local(stack, context, vars, offset)
}

pub fn store_var<BeforeOverwrite, AfterStore>(
    stack: &mut Vec<Value>,
    vars: &mut Vec<Value>,
    index: usize,
    var_names: &HashMap<usize, String>,
    before_overwrite: BeforeOverwrite,
    after_store: AfterStore,
) -> Result<(), RuntimeError>
where
    BeforeOverwrite: FnMut(&Value, &Value),
    AfterStore: FnMut(usize, &Value),
{
    stack_ops::store_var(stack, vars, index, var_names, before_overwrite, after_store)
}

pub fn store_local<BeforeLocalOverwrite, BeforeVarOverwrite, AfterFallbackStore>(
    stack: &mut Vec<Value>,
    context: &mut ExecutionContext,
    vars: &mut Vec<Value>,
    offset: usize,
    before_local_overwrite: BeforeLocalOverwrite,
    before_var_overwrite: BeforeVarOverwrite,
    after_fallback_store: AfterFallbackStore,
) -> Result<(), RuntimeError>
where
    BeforeLocalOverwrite: FnMut(&Value, &Value),
    BeforeVarOverwrite: FnMut(&Value, &Value),
    AfterFallbackStore: FnMut(&str, usize, &Value),
{
    stack_ops::store_local(
        stack,
        context,
        vars,
        offset,
        before_local_overwrite,
        before_var_overwrite,
        after_fallback_store,
    )
}
