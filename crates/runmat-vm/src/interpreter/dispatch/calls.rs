use crate::call::builtins::ImportedBuiltinResolution;
use crate::call::feval::FevalDispatch;
use crate::call::shared::{
    build_expanded_args_from_specs, collect_multi_outputs, expand_cell_indices,
    lookup_user_function, prepare_user_call, subsref_empty_brace_cell,
    subsref_brace_numeric_index_values, validate_user_function_arity, PreparedUserCall,
};
use crate::bytecode::ArgSpec;
use crate::functions::UserFunction;
use crate::interpreter::dispatch::exceptions::{redirect_exception_to_catch, ExceptionHandling};
use runmat_builtins::{MException, Value};
use runmat_runtime::RuntimeError;
use std::future::Future;

pub enum FevalHandling {
    Completed,
    InvokeUser {
        name: String,
        args: Vec<Value>,
        functions: std::collections::HashMap<String, crate::functions::UserFunction>,
    },
}

pub struct PreparedUserDispatch {
    pub func: UserFunction,
    pub var_map: std::collections::HashMap<runmat_hir::VarId, runmat_hir::VarId>,
    pub func_program: runmat_hir::HirProgram,
    pub func_vars: Vec<Value>,
}

pub enum BuiltinHandling {
    Completed,
    Caught,
    Uncaught(RuntimeError),
}

pub fn handle_feval_dispatch(
    dispatch: Result<FevalDispatch, RuntimeError>,
    stack: &mut Vec<Value>,
) -> Result<FevalHandling, RuntimeError> {
    match dispatch? {
        FevalDispatch::Completed(result) => {
            stack.push(result);
            Ok(FevalHandling::Completed)
        }
        FevalDispatch::InvokeUser {
            name,
            args,
            functions,
        } => Ok(FevalHandling::InvokeUser {
            name,
            args,
            functions,
        }),
    }
}

pub fn unpack_prepared_user_call(prepared: PreparedUserCall) -> PreparedUserDispatch {
    let PreparedUserCall {
        func,
        var_map,
        func_program,
        func_vars,
    } = prepared;
    PreparedUserDispatch {
        func,
        var_map,
        func_program,
        func_vars,
    }
}

pub fn prepare_named_user_dispatch(
    name: &str,
    functions: &std::collections::HashMap<String, UserFunction>,
    args: &[Value],
    vars: &[Value],
) -> Result<PreparedUserDispatch, RuntimeError> {
    let func = lookup_user_function(name, functions)?;
    validate_user_function_arity(name, &func, args.len()).map_err(RuntimeError::from)?;
    let prepared = prepare_user_call(func, args, vars).map_err(RuntimeError::from)?;
    Ok(unpack_prepared_user_call(prepared))
}

pub fn push_user_call_outputs(
    stack: &mut Vec<Value>,
    name: &str,
    func: &UserFunction,
    var_map: &std::collections::HashMap<runmat_hir::VarId, runmat_hir::VarId>,
    func_result_vars: &[Value],
    out_count: usize,
) -> Result<(), RuntimeError> {
    let outputs = collect_multi_outputs(name, func, var_map, func_result_vars, out_count)?;
    for value in outputs {
        stack.push(value);
    }
    Ok(())
}

pub fn output_list_for_user_call(
    name: &str,
    func: &UserFunction,
    var_map: &std::collections::HashMap<runmat_hir::VarId, runmat_hir::VarId>,
    func_result_vars: &[Value],
    out_count: usize,
) -> Result<Value, RuntimeError> {
    let outputs = collect_multi_outputs(name, func, var_map, func_result_vars, out_count)?;
    Ok(Value::OutputList(outputs))
}

pub fn push_single_result(stack: &mut Vec<Value>, result: Value) {
    stack.push(result);
}

pub async fn build_builtin_expand_last_args<F, Fut>(
    stack: &mut Vec<Value>,
    fixed_argc: usize,
    num_indices: usize,
    invalid_expand_msg: &'static str,
    mut expand_object_indices: F,
) -> Result<Vec<Value>, RuntimeError>
where
    F: FnMut(Value, Vec<Value>) -> Fut,
    Fut: Future<Output = Result<Vec<Value>, RuntimeError>>,
{
    let mut indices = Vec::with_capacity(num_indices);
    for _ in 0..num_indices {
        indices.push(
            stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?,
        );
    }
    indices.reverse();
    let base = stack
        .pop()
        .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;
    let mut fixed = Vec::with_capacity(fixed_argc);
    for _ in 0..fixed_argc {
        fixed.push(
            stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?,
        );
    }
    fixed.reverse();

    let expanded = match (base, indices.len()) {
        (Value::Cell(ca), 1) | (Value::Cell(ca), 2) => expand_cell_indices(&ca, &indices)?,
        (other, _) => match other {
            Value::Object(obj) => expand_object_indices(Value::Object(obj), indices).await?,
            _ => return Err(crate::interpreter::errors::mex("ExpandError", invalid_expand_msg)),
        },
    };

    let mut args = fixed;
    args.extend(expanded);
    Ok(args)
}

pub async fn build_builtin_expand_at_args<F, Fut>(
    stack: &mut Vec<Value>,
    before_count: usize,
    num_indices: usize,
    after_count: usize,
    invalid_expand_msg: &'static str,
    mut expand_object_indices: F,
) -> Result<Vec<Value>, RuntimeError>
where
    F: FnMut(Value, Vec<Value>) -> Fut,
    Fut: Future<Output = Result<Vec<Value>, RuntimeError>>,
{
    let mut after = Vec::with_capacity(after_count);
    for _ in 0..after_count {
        after.push(
            stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?,
        );
    }
    after.reverse();

    let mut indices = Vec::with_capacity(num_indices);
    for _ in 0..num_indices {
        indices.push(
            stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?,
        );
    }
    indices.reverse();

    let base = stack
        .pop()
        .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?;

    let mut before = Vec::with_capacity(before_count);
    for _ in 0..before_count {
        before.push(
            stack
                .pop()
                .ok_or(crate::interpreter::errors::mex("StackUnderflow", "stack underflow"))?,
        );
    }
    before.reverse();

    let expanded = match (base, indices.len()) {
        (Value::Cell(ca), 1) | (Value::Cell(ca), 2) => expand_cell_indices(&ca, &indices)?,
        (Value::Object(obj), _) => expand_object_indices(Value::Object(obj), indices).await?,
        _ => return Err(crate::interpreter::errors::mex("ExpandError", invalid_expand_msg)),
    };

    let mut args = before;
    args.extend(expanded);
    args.extend(after);
    Ok(args)
}

pub async fn build_builtin_expand_multi_args(
    stack: &mut Vec<Value>,
    specs: &[ArgSpec],
) -> Result<Vec<Value>, RuntimeError> {
    build_expanded_args_from_specs(
        stack,
        specs,
        "CallBuiltinExpandMulti requires cell or object for expand_all",
        "CallBuiltinExpandMulti requires cell or object cell access",
        |base| async move {
            match base {
                Value::Object(obj) => {
                    let empty = subsref_empty_brace_cell()?;
                    let args = vec![
                        Value::Object(obj),
                        Value::String("subsref".to_string()),
                        Value::String("{}".to_string()),
                        empty,
                    ];
                    let v = runmat_runtime::call_builtin_async("call_method", &args).await?;
                    Ok(match v {
                        Value::Cell(ca) => crate::call::shared::expand_all_cell(&ca),
                        other => vec![other],
                    })
                }
                _ => Err(crate::interpreter::errors::mex(
                    "ExpandError",
                    "CallBuiltinExpandMulti requires cell or object for expand_all",
                )),
            }
        },
        |base, indices| async move {
            match base {
                Value::Object(obj) => {
                    let idx_vals = subsref_brace_numeric_index_values(&indices);
                    let cell = runmat_runtime::call_builtin_async("__make_cell", &idx_vals).await?;
                    let args = vec![
                        Value::Object(obj),
                        Value::String("subsref".to_string()),
                        Value::String("{}".to_string()),
                        cell,
                    ];
                    let v = runmat_runtime::call_builtin_async("call_method", &args).await?;
                    Ok(vec![v])
                }
                _ => Err(crate::interpreter::errors::mex(
                    "ExpandError",
                    "CallBuiltinExpandMulti requires cell or object cell access",
                )),
            }
        },
    )
    .await
}

pub async fn build_feval_expand_multi_args(
    stack: &mut Vec<Value>,
    specs: &[ArgSpec],
) -> Result<Vec<Value>, RuntimeError> {
    build_expanded_args_from_specs(
        stack,
        specs,
        "CallFevalExpandMulti requires cell or object for expand_all",
        "CallFevalExpandMulti requires cell or object cell access",
        |base| async move {
            match base {
                Value::Object(obj) => {
                    let empty = subsref_empty_brace_cell()?;
                    let args = vec![
                        Value::Object(obj),
                        Value::String("subsref".to_string()),
                        Value::String("{}".to_string()),
                        empty,
                    ];
                    let v = runmat_runtime::call_builtin_async("call_method", &args).await?;
                    Ok(match v {
                        Value::Cell(ca) => crate::call::shared::expand_all_cell(&ca),
                        other => vec![other],
                    })
                }
                _ => Err(crate::interpreter::errors::mex(
                    "InvalidExpandAllTarget",
                    "CallFevalExpandMulti requires cell or object for expand_all",
                )),
            }
        },
        |base, indices| async move {
            match base {
                Value::Object(obj) => {
                    let cell = crate::call::shared::subsref_brace_index_cell_raw(&indices)?;
                    let args = vec![
                        Value::Object(obj),
                        Value::String("subsref".to_string()),
                        Value::String("{}".to_string()),
                        cell,
                    ];
                    let v = runmat_runtime::call_builtin_async("call_method", &args).await?;
                    Ok(vec![v])
                }
                _ => Err(crate::interpreter::errors::mex(
                    "ExpandError",
                    "CallFevalExpandMulti requires cell or object cell access",
                )),
            }
        },
    )
    .await
}

pub fn handle_builtin_outcome(
    result: Result<Value, RuntimeError>,
    imported: ImportedBuiltinResolution,
    stack: &mut Vec<Value>,
    try_stack: &mut Vec<(usize, Option<usize>)>,
    vars: &mut Vec<Value>,
    last_exception: &mut Option<MException>,
    pc: &mut usize,
    refresh_vars: impl Fn(&[Value]),
) -> Result<BuiltinHandling, RuntimeError> {
    match result {
        Ok(result) => {
            stack.push(result);
            Ok(BuiltinHandling::Completed)
        }
        Err(err) => match imported {
            ImportedBuiltinResolution::Resolved(value) => {
                stack.push(value);
                Ok(BuiltinHandling::Completed)
            }
            ImportedBuiltinResolution::Ambiguous(message) => Err(message.into()),
            ImportedBuiltinResolution::NotFound => Ok(
                match redirect_exception_to_catch(
                    err,
                    try_stack,
                    vars,
                    last_exception,
                    pc,
                    refresh_vars,
                ) {
                    ExceptionHandling::Caught => BuiltinHandling::Caught,
                    ExceptionHandling::Uncaught(err) => BuiltinHandling::Uncaught(err),
                },
            ),
        },
    }
}
