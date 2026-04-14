use crate::call::builtins::ImportedBuiltinResolution;
use crate::call::closures as call_closures;
use crate::call::feval::FevalDispatch;
use crate::call::shared::{
    build_expanded_args_from_specs, collect_multi_outputs, expand_cell_indices,
    lookup_user_function, prepare_user_call, subsref_empty_brace_cell,
    subsref_brace_numeric_index_values, validate_user_function_arity, PreparedUserCall,
};
use crate::bytecode::ArgSpec;
use crate::bytecode::Instr;
use crate::functions::UserFunction;
use crate::interpreter::dispatch::exceptions::{redirect_exception_to_catch, ExceptionHandling};
use crate::object::class_def as obj_class_def;
use crate::object::resolve as obj_resolve;
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

pub enum MethodHandling {
    Completed,
}

pub enum UserCallHandling {
    Completed,
    Caught,
    Uncaught(RuntimeError),
}

#[cfg(feature = "native-accel")]
async fn accel_prepare_args(name: &str, args: &[Value]) -> Result<Vec<Value>, RuntimeError> {
    Ok(runmat_accelerate::prepare_builtin_args(name, args)
        .await
        .map_err(|e| e.to_string())?)
}

#[cfg(not(feature = "native-accel"))]
async fn accel_prepare_args(_name: &str, args: &[Value]) -> Result<Vec<Value>, RuntimeError> {
    Ok(args.to_vec())
}

async fn call_builtin_auto(name: &str, args: &[Value]) -> Result<Value, RuntimeError> {
    let prepared = accel_prepare_args(name, args).await?;
    runmat_runtime::call_builtin_async(name, &prepared).await
}

fn output_hint_for_next(next_instr: Option<&Instr>) -> usize {
    match next_instr {
        Some(Instr::Pop) | Some(Instr::EmitStackTop { .. }) => 0,
        _ => 1,
    }
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

pub async fn build_user_function_expand_multi_args(
    stack: &mut Vec<Value>,
    specs: &[ArgSpec],
) -> Result<Vec<Value>, RuntimeError> {
    build_expanded_args_from_specs(
        stack,
        specs,
        "CallFunctionExpandMulti requires cell or object for expand_all",
        "CallFunctionExpandMulti requires cell or object cell access",
        |base| async move {
            match base {
                Value::Cell(ca) => Ok(crate::call::shared::expand_all_cell(&ca)),
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
                    "CallFunctionExpandMulti requires cell or object for expand_all",
                )),
            }
        },
        |base, indices| async move {
            match (base, indices.len()) {
                (Value::Cell(ca), 1) | (Value::Cell(ca), 2) => expand_cell_indices(&ca, &indices),
                (Value::Object(obj), _) => {
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
                    "CallFunctionExpandMulti requires cell or object cell access",
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

pub async fn handle_method_call(
    stack: &mut Vec<Value>,
    name: &str,
    arg_count: usize,
) -> Result<MethodHandling, RuntimeError> {
    let (base, args) = call_closures::collect_method_args(stack, arg_count)?;
    let value = call_closures::call_method(base, name, args).await?;
    stack.push(value);
    Ok(MethodHandling::Completed)
}

pub async fn handle_prepared_user_function_call<BF, BFFut, IF, IFFut>(
    stack: &mut Vec<Value>,
    name: &str,
    args: Vec<Value>,
    out_count: usize,
    bytecode_functions: &std::collections::HashMap<String, UserFunction>,
    vars: &mut Vec<Value>,
    try_stack: &mut Vec<(usize, Option<usize>)>,
    last_exception: &mut Option<MException>,
    pc: &mut usize,
    refresh_vars: impl Fn(&[Value]),
    builtin_fallback: BF,
    interpret_counts: IF,
) -> Result<UserCallHandling, RuntimeError>
where
    BF: FnOnce(String, Vec<Value>, usize) -> BFFut,
    BFFut: Future<Output = Result<Option<Value>, RuntimeError>>,
    IF: FnOnce(crate::bytecode::Bytecode, Vec<Value>, String, usize, usize) -> IFFut,
    IFFut: Future<Output = Result<Vec<Value>, RuntimeError>>,
{
    let arg_count = args.len();
    if let Some(result) = builtin_fallback(name.to_string(), args.clone(), out_count).await? {
        stack.push(result);
        return Ok(UserCallHandling::Completed);
    }

    let prepared = prepare_named_user_dispatch(name, bytecode_functions, &args, vars)?;
    let PreparedUserDispatch {
        func,
        var_map,
        func_program,
        func_vars,
    } = prepared;
    let mut func_bytecode = crate::compile(&func_program, bytecode_functions)?;
    func_bytecode.source_id = func.source_id;

    let func_result_vars = match interpret_counts(
        func_bytecode,
        func_vars,
        name.to_string(),
        out_count,
        arg_count,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            return Ok(match redirect_exception_to_catch(
                e,
                try_stack,
                vars,
                last_exception,
                pc,
                refresh_vars,
            ) {
                ExceptionHandling::Caught => UserCallHandling::Caught,
                ExceptionHandling::Uncaught(err) => UserCallHandling::Uncaught(err),
            })
        }
    };

    if out_count == 1 {
        push_user_call_outputs(stack, name, &func, &var_map, &func_result_vars, 1)?;
    } else {
        let output_list = output_list_for_user_call(name, &func, &var_map, &func_result_vars, out_count)?;
        push_single_result(stack, output_list);
    }
    Ok(UserCallHandling::Completed)
}

pub async fn handle_user_function_call<BF, BFFut, IF, IFFut>(
    stack: &mut Vec<Value>,
    name: &str,
    arg_count: usize,
    out_count: usize,
    bytecode_functions: &std::collections::HashMap<String, UserFunction>,
    vars: &mut Vec<Value>,
    try_stack: &mut Vec<(usize, Option<usize>)>,
    last_exception: &mut Option<MException>,
    pc: &mut usize,
    refresh_vars: impl Fn(&[Value]),
    builtin_fallback: BF,
    interpret_counts: IF,
) -> Result<UserCallHandling, RuntimeError>
where
    BF: FnOnce(String, Vec<Value>, usize) -> BFFut,
    BFFut: Future<Output = Result<Option<Value>, RuntimeError>>,
    IF: FnOnce(crate::bytecode::Bytecode, Vec<Value>, String, usize, usize) -> IFFut,
    IFFut: Future<Output = Result<Vec<Value>, RuntimeError>>,
{
    let args = crate::call::builtins::collect_call_args(stack, arg_count)?;
    handle_prepared_user_function_call(
        stack,
        name,
        args,
        out_count,
        bytecode_functions,
        vars,
        try_stack,
        last_exception,
        pc,
        refresh_vars,
        builtin_fallback,
        interpret_counts,
    )
    .await
}

pub async fn handle_builtin_expand_last_call<F, Fut>(
    stack: &mut Vec<Value>,
    name: &str,
    fixed_argc: usize,
    num_indices: usize,
    next_instr: Option<&Instr>,
    expand_object_indices: F,
) -> Result<BuiltinHandling, RuntimeError>
where
    F: FnMut(Value, Vec<Value>) -> Fut,
    Fut: Future<Output = Result<Vec<Value>, RuntimeError>>,
{
    let args = build_builtin_expand_last_args(
        stack,
        fixed_argc,
        num_indices,
        "CallBuiltinExpandLast requires cell or object cell access",
        expand_object_indices,
    )
    .await?;
    let output_hint = output_hint_for_next(next_instr);
    let _output_guard = runmat_runtime::output_context::push_output_count(output_hint);
    push_single_result(stack, call_builtin_auto(name, &args).await?);
    Ok(BuiltinHandling::Completed)
}

pub async fn handle_builtin_expand_at_call<F, Fut>(
    stack: &mut Vec<Value>,
    name: &str,
    before_count: usize,
    num_indices: usize,
    after_count: usize,
    next_instr: Option<&Instr>,
    expand_object_indices: F,
) -> Result<BuiltinHandling, RuntimeError>
where
    F: FnMut(Value, Vec<Value>) -> Fut,
    Fut: Future<Output = Result<Vec<Value>, RuntimeError>>,
{
    let args = build_builtin_expand_at_args(
        stack,
        before_count,
        num_indices,
        after_count,
        "CallBuiltinExpandAt requires cell or object cell access",
        expand_object_indices,
    )
    .await?;
    let output_hint = output_hint_for_next(next_instr);
    let _output_guard = runmat_runtime::output_context::push_output_count(output_hint);
    push_single_result(stack, call_builtin_auto(name, &args).await?);
    Ok(BuiltinHandling::Completed)
}

pub async fn handle_builtin_expand_multi_call(
    stack: &mut Vec<Value>,
    name: &str,
    specs: &[ArgSpec],
    next_instr: Option<&Instr>,
) -> Result<BuiltinHandling, RuntimeError> {
    let args = build_builtin_expand_multi_args(stack, specs).await?;
    let output_hint = output_hint_for_next(next_instr);
    let _output_guard = runmat_runtime::output_context::push_output_count(output_hint);
    push_single_result(stack, call_builtin_auto(name, &args).await?);
    Ok(BuiltinHandling::Completed)
}

pub async fn handle_method_or_member_index_call(
    stack: &mut Vec<Value>,
    name: String,
    arg_count: usize,
) -> Result<MethodHandling, RuntimeError> {
    let (base, args) = call_closures::collect_method_args(stack, arg_count)?;
    let value = call_closures::call_method_or_member_index(base, name, args).await?;
    stack.push(value);
    Ok(MethodHandling::Completed)
}

pub async fn handle_static_method_call(
    stack: &mut Vec<Value>,
    class_name: &str,
    method: &str,
    arg_count: usize,
) -> Result<MethodHandling, RuntimeError> {
    let mut args = crate::call::builtins::collect_call_args(stack, arg_count)?;
    match call_closures::call_static_method(class_name, method, args.clone()).await {
        Ok(v) => {
            stack.push(v);
            Ok(MethodHandling::Completed)
        }
        Err(_) => {
            let is_type_class = matches!(
                class_name,
                "gpuArray"
                    | "logical"
                    | "double"
                    | "single"
                    | "int8"
                    | "int16"
                    | "int32"
                    | "int64"
                    | "uint8"
                    | "uint16"
                    | "uint32"
                    | "uint64"
                    | "char"
                    | "string"
                    | "cell"
                    | "struct"
            );
            if is_type_class {
                args.push(Value::from(class_name));
                let v = runmat_runtime::call_builtin_async(method, &args).await?;
                stack.push(v);
                Ok(MethodHandling::Completed)
            } else {
                Err(format!("Unknown static method '{}' on class {}", method, class_name).into())
            }
        }
    }
}

pub fn handle_load_method(
    stack: &mut Vec<Value>,
    name: String,
) -> Result<MethodHandling, RuntimeError> {
    let base = crate::interpreter::stack::pop_value(stack)?;
    let value = call_closures::load_method_closure(base, name)?;
    stack.push(value);
    Ok(MethodHandling::Completed)
}

pub fn handle_create_closure(
    stack: &mut Vec<Value>,
    func_name: String,
    capture_count: usize,
) -> Result<MethodHandling, RuntimeError> {
    call_closures::create_closure(stack, func_name, capture_count)?;
    Ok(MethodHandling::Completed)
}

pub fn handle_load_static_property(
    stack: &mut Vec<Value>,
    class_name: &str,
    prop: &str,
) -> Result<MethodHandling, RuntimeError> {
    let value = obj_resolve::load_static_member(class_name, prop)?;
    stack.push(value);
    Ok(MethodHandling::Completed)
}

pub fn handle_register_class(
    name: String,
    super_class: Option<String>,
    properties: Vec<(String, bool, String, String)>,
    methods: Vec<(String, String, bool, String)>,
) -> Result<MethodHandling, RuntimeError> {
    obj_class_def::register_class(name, super_class, properties, methods)?;
    Ok(MethodHandling::Completed)
}
