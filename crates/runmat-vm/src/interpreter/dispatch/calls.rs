use crate::call::builtins::ImportedBuiltinResolution;
use crate::call::feval::FevalDispatch;
use crate::call::shared::{collect_multi_outputs, PreparedUserCall};
use crate::functions::UserFunction;
use crate::interpreter::dispatch::exceptions::{redirect_exception_to_catch, ExceptionHandling};
use runmat_builtins::{MException, Value};
use runmat_runtime::RuntimeError;

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
