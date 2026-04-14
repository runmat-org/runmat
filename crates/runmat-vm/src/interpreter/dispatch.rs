use crate::bytecode::program::Bytecode;
use crate::call::builtins::ImportedBuiltinResolution;
use crate::call::feval::FevalDispatch;
use crate::call::shared::{collect_multi_outputs, PreparedUserCall};
use crate::functions::UserFunction;
use crate::interpreter::errors::{attach_span_at, ensure_runtime_error_identifier};
use crate::ops::control_flow::ControlFlowAction;
use crate::runtime::call_stack::error_namespace;
use runmat_builtins::{MException, Value};
use runmat_runtime::RuntimeError;

pub enum ExceptionHandling {
    Caught,
    Uncaught(RuntimeError),
}

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

pub enum DispatchDecision {
    ContinueLoop,
    FallThrough,
    Return,
}

#[inline]
pub fn apply_control_flow_action(action: ControlFlowAction, pc: &mut usize) -> DispatchDecision {
    match action {
        ControlFlowAction::Jump(target) => {
            *pc = target;
            DispatchDecision::ContinueLoop
        }
        ControlFlowAction::Next => DispatchDecision::FallThrough,
        ControlFlowAction::Return => DispatchDecision::Return,
    }
}

pub fn parse_exception(err: &RuntimeError) -> MException {
    if let Some(identifier) = err.identifier() {
        return MException::new(identifier.to_string(), err.message().to_string());
    }
    let message = err.message();
    if let Some(idx) = message.rfind(": ") {
        let (id, msg) = message.split_at(idx);
        let message = msg.trim_start_matches(':').trim().to_string();
        let ident = if id.trim().is_empty() {
            format!("{}:error", error_namespace())
        } else {
            id.trim().to_string()
        };
        return MException::new(ident, message);
    }
    if let Some(idx) = message.rfind(':') {
        let (id, msg) = message.split_at(idx);
        let message = msg.trim_start_matches(':').trim().to_string();
        let ident = if id.trim().is_empty() {
            format!("{}:error", error_namespace())
        } else {
            id.trim().to_string()
        };
        return MException::new(ident, message);
    }
    MException::new(format!("{}:error", error_namespace()), message.to_string())
}

pub fn redirect_exception_to_catch(
    err: RuntimeError,
    try_stack: &mut Vec<(usize, Option<usize>)>,
    vars: &mut Vec<Value>,
    last_exception: &mut Option<MException>,
    pc: &mut usize,
    refresh_vars: impl Fn(&[Value]),
) -> ExceptionHandling {
    if let Some((catch_pc, catch_var)) = try_stack.pop() {
        if let Some(var_idx) = catch_var {
            if var_idx >= vars.len() {
                vars.resize(var_idx + 1, Value::Num(0.0));
                refresh_vars(vars);
            }
            let mex = parse_exception(&err);
            *last_exception = Some(mex.clone());
            vars[var_idx] = Value::MException(mex);
        }
        *pc = catch_pc;
        ExceptionHandling::Caught
    } else {
        ExceptionHandling::Uncaught(err)
    }
}

pub fn prepare_vm_error(
    bytecode: &Bytecode,
    pc: usize,
    err: impl Into<RuntimeError>,
) -> RuntimeError {
    let err: RuntimeError = ensure_runtime_error_identifier(err.into());
    attach_span_at(bytecode, pc, err)
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
