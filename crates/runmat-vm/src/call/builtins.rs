use crate::bytecode::Instr;
use crate::interpreter::errors::mex;
use crate::interpreter::stack::pop_args;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

#[cfg(feature = "native-accel")]
pub async fn prepare_builtin_args(name: &str, args: &[Value]) -> Result<Vec<Value>, RuntimeError> {
    Ok(runmat_accelerate::prepare_builtin_args(name, args)
        .await
        .map_err(|e| e.to_string())?)
}

#[cfg(not(feature = "native-accel"))]
pub async fn prepare_builtin_args(
    _name: &str,
    args: &[Value],
) -> Result<Vec<Value>, RuntimeError> {
    Ok(args.to_vec())
}

pub fn collect_call_args(stack: &mut Vec<Value>, arg_count: usize) -> Result<Vec<Value>, RuntimeError> {
    pop_args(stack, arg_count)
}

pub fn special_counter_builtin(
    name: &str,
    arg_count: usize,
    call_counts: &[(usize, usize)],
) -> Result<Option<Value>, RuntimeError> {
    if name == "nargin" {
        if arg_count != 0 {
            return Err(mex("TooManyInputs", "nargin takes no arguments"));
        }
        let (nin, _) = call_counts.last().cloned().unwrap_or((0, 0));
        return Ok(Some(Value::Num(nin as f64)));
    }
    if name == "nargout" {
        if arg_count != 0 {
            return Err(mex("TooManyInputs", "nargout takes no arguments"));
        }
        let (_, nout) = call_counts.last().cloned().unwrap_or((0, 0));
        return Ok(Some(Value::Num(nout as f64)));
    }
    Ok(None)
}

pub fn requested_output_count(instructions: &[Instr], pc: usize) -> Option<usize> {
    match instructions.get(pc + 1) {
        Some(Instr::Unpack(count)) => Some(*count),
        _ => None,
    }
}

pub fn single_result_output_list(result: Value, out_count: usize) -> Value {
    let mut outputs = Vec::with_capacity(out_count);
    if out_count > 0 {
        outputs.push(result);
        for _ in 1..out_count {
            outputs.push(Value::Num(0.0));
        }
    }
    Value::OutputList(outputs)
}
