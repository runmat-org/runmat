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

pub enum ImportedBuiltinResolution {
    Resolved(Value),
    Ambiguous(String),
    NotFound,
}

pub async fn resolve_imported_builtin(
    name: &str,
    imports: &[(Vec<String>, bool)],
    prepared_primary: &[Value],
    requested_outputs: Option<usize>,
) -> Result<ImportedBuiltinResolution, RuntimeError> {
    let mut specific_matches: Vec<(String, Value)> = Vec::new();
    for (path, wildcard) in imports {
        if *wildcard {
            continue;
        }
        if path.last().map(|s| s.as_str()) == Some(name) {
            let qual = path.join(".");
            let qual_args = prepare_builtin_args(&qual, prepared_primary).await?;
            let result = match requested_outputs {
                Some(count) => runmat_runtime::call_builtin_async_with_outputs(&qual, &qual_args, count).await,
                None => runmat_runtime::call_builtin_async(&qual, &qual_args).await,
            };
            if let Ok(value) = result {
                specific_matches.push((qual, value));
            }
        }
    }
    if specific_matches.len() > 1 {
        let msg = specific_matches
            .iter()
            .map(|(q, _)| q.clone())
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(ImportedBuiltinResolution::Ambiguous(format!(
            "ambiguous builtin '{}' via imports: {}",
            name, msg
        )));
    }
    if let Some((_, value)) = specific_matches.pop() {
        return Ok(ImportedBuiltinResolution::Resolved(value));
    }

    let mut wildcard_matches: Vec<(String, Value)> = Vec::new();
    for (path, wildcard) in imports {
        if !*wildcard || path.is_empty() {
            continue;
        }
        let qual = format!("{}.{}", path.join("."), name);
        let qual_args = prepare_builtin_args(&qual, prepared_primary).await?;
        let result = match requested_outputs {
            Some(count) => runmat_runtime::call_builtin_async_with_outputs(&qual, &qual_args, count).await,
            None => runmat_runtime::call_builtin_async(&qual, &qual_args).await,
        };
        if let Ok(value) = result {
            wildcard_matches.push((qual, value));
        }
    }
    if wildcard_matches.len() > 1 {
        let msg = wildcard_matches
            .iter()
            .map(|(q, _)| q.clone())
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(ImportedBuiltinResolution::Ambiguous(format!(
            "ambiguous builtin '{}' via wildcard imports: {}",
            name, msg
        )));
    }
    if let Some((_, value)) = wildcard_matches.pop() {
        return Ok(ImportedBuiltinResolution::Resolved(value));
    }

    Ok(ImportedBuiltinResolution::NotFound)
}

pub fn rethrow_without_explicit_exception(
    name: &str,
    args: &[Value],
    last_identifier: Option<&str>,
    last_message: Option<&str>,
) -> Option<RuntimeError> {
    if name == "rethrow" && args.is_empty() {
        if let (Some(identifier), Some(message)) = (last_identifier, last_message) {
            return Some(format!("{}: {}", identifier, message).to_string().into());
        }
    }
    None
}
