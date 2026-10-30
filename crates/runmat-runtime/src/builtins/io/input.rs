//! MATLAB-compatible `input` builtin for line-oriented console interaction.

use runmat_builtins::{
    BuiltinErrorDescriptor, InputStringMode, INPUT_ERROR_EVAL_FAILED,
    INPUT_ERROR_INTERACTION_FAILED, INPUT_ERROR_INVALID_NUMERIC_EXPRESSION,
    INPUT_ERROR_INVALID_PROMPT_TYPE, INPUT_ERROR_INVALID_STRING_FLAG,
    INPUT_ERROR_PROMPT_ROW_VECTOR, INPUT_ERROR_PROMPT_SCALAR_STRING, INPUT_ERROR_TOO_MANY_INPUTS,
    INPUT_ERROR_TOO_MANY_OUTPUTS, INPUT_NO_PROMPT_EXTENSION, INPUT_SWAPPED_ARGUMENTS_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{
    BuiltinIntegerAuditKind, INPUT_DESCRIPTOR, INPUT_EXTENSIONS, INPUT_INTEGER_AUDIT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, LogicalArray, Tensor, Value};

use crate::execution::RootedValue;
use crate::interaction;
use crate::{
    build_runtime_error, call_builtin_async, gather_if_needed_async, BuiltinResult, RuntimeError,
};

const DEFAULT_PROMPT: &str = "Input: ";
const BUILTIN_NAME: &str = "input";

fn input_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    input_error_with(error, error.message)
}

fn input_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier.to_string());
    }
    builder.build()
}

fn input_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier.to_string());
    }
    builder.build()
}

#[runtime_builtin(
    name = "input",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::input"
)]
async fn input_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<RootedValue> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(input_error(&INPUT_ERROR_TOO_MANY_OUTPUTS));
    }
    if args.len() > 2 {
        return Err(input_error(&INPUT_ERROR_TOO_MANY_INPUTS));
    }

    if args.is_empty() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INPUT_NO_PROMPT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.iter().any(crate::dispatcher::value_contains_gpu) {
        return Err(input_error(&INPUT_ERROR_INVALID_PROMPT_TYPE));
    }

    let mut prompt_index = if args.is_empty() { None } else { Some(0usize) };
    let mut parsed_flag: Option<InputStringMode> = None;

    if let Some(idx) = if args.len() == 2 { Some(1usize) } else { None } {
        match parse_string_flag(&args[idx]).await {
            Ok(flag) => parsed_flag = Some(flag),
            Err(original_err) => {
                if let Some(prompt_idx) = prompt_index {
                    match parse_string_flag(&args[prompt_idx]).await {
                        Ok(swapped_flag) => {
                            crate::compatibility::ensure_builtin_extension_enabled(
                                &INPUT_SWAPPED_ARGUMENTS_EXTENSION,
                                BUILTIN_NAME,
                            )?;
                            parsed_flag = Some(swapped_flag);
                            prompt_index = Some(idx);
                        }
                        Err(_) => {
                            return Err(original_err);
                        }
                    }
                } else {
                    return Err(original_err);
                }
            }
        }
    }

    let prompt = if let Some(idx) = prompt_index {
        parse_prompt(&args[idx]).await?
    } else {
        DEFAULT_PROMPT.to_string()
    };
    let return_string = parsed_flag.is_some();
    loop {
        let line = interaction::request_line_async(&prompt, true)
            .await
            .map_err(|err| {
                let message = err.message().to_string();
                input_error_with_source(
                    &INPUT_ERROR_INTERACTION_FAILED,
                    format!("input: {message}"),
                    err,
                )
            })?;
        if return_string {
            return RootedValue::new(Value::CharArray(CharArray::new_row(&line)))
                .map_err(|error| error.into_runtime_error());
        }
        match parse_numeric_response(&line).await {
            Ok(value) => return Ok(value),
            Err(error)
                if error.failure_origin()
                    == crate::runtime_error::RuntimeFailureOrigin::LanguageEvaluation =>
            {
                crate::console::record_console_line(
                    crate::console::ConsoleStream::Stderr,
                    error.format_diagnostic(),
                )?;
            }
            Err(error) => return Err(error),
        }
    }
}

async fn parse_prompt(value: &Value) -> Result<String, RuntimeError> {
    let gathered = gather_if_needed_async(value).await?;
    match &gathered {
        Value::CharArray(_) => {
            crate::builtins::common::validation::value_text_scalar_string(&gathered)
                .ok_or_else(|| input_error(&INPUT_ERROR_PROMPT_ROW_VECTOR))
        }
        Value::String(runmat_value::StringElement::Text(_)) => {
            crate::builtins::common::validation::value_text_scalar_string(&gathered)
                .ok_or_else(|| input_error(&INPUT_ERROR_INVALID_PROMPT_TYPE))
        }
        Value::StringArray(_) => {
            crate::builtins::common::validation::value_text_scalar_string(&gathered)
                .ok_or_else(|| input_error(&INPUT_ERROR_PROMPT_SCALAR_STRING))
        }
        other => Err(input_error_with(
            &INPUT_ERROR_INVALID_PROMPT_TYPE,
            format!("input: invalid prompt type ({other:?})"),
        )),
    }
}

async fn parse_string_flag(value: &Value) -> Result<InputStringMode, RuntimeError> {
    let gathered = gather_if_needed_async(value).await?;
    let text = crate::builtins::common::validation::value_text_scalar_string(&gathered)
        .ok_or_else(|| {
            input_error_with(
                &INPUT_ERROR_INVALID_STRING_FLAG,
                format!("input: invalid string flag ({gathered:?})"),
            )
        })?;
    InputStringMode::decode(&text).ok_or_else(|| {
        input_error_with(
            &INPUT_ERROR_INVALID_STRING_FLAG,
            format!("input: invalid string flag ({text})"),
        )
    })
}

async fn parse_numeric_response(line: &str) -> Result<RootedValue, RuntimeError> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed == "[]" {
        return RootedValue::new(Value::Tensor(Tensor::zeros(vec![0, 0])))
            .map_err(|error| error.into_runtime_error());
    }

    // Fast path 1: scalar literals, named constants, and logical keywords.
    // Handles the vast majority of input() use cases without touching the VM.
    if let Some(v) = parse_scalar_value(trimmed) {
        return RootedValue::new(v).map_err(|error| error.into_runtime_error());
    }

    // Fast path 2: matrix/vector literals like `[1 2 3]`, `[1;2;3]`, `[true false]`.
    // Avoids recursive interpret() calls for this common case.
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        if let Some(v) = parse_matrix_literal(trimmed) {
            return RootedValue::new(v).map_err(|error| error.into_runtime_error());
        }
    }

    // Full eval path for complex expressions (`sqrt(2)`, `pi/2`, `ones(3)`, etc.).
    // The eval hook is only safe to call when the executor can handle re-entrant
    // polls (e.g. the WASM async runtime). On native the fast paths above cover
    // the common cases; truly complex expressions fall back to str2double here.
    if let Some(hook) = interaction::current_eval_hook() {
        let outcome = hook(trimmed.to_string()).await.map_err(|err| {
            let message = err.message().to_string();
            input_error_with_source(
                &INPUT_ERROR_EVAL_FAILED,
                format!("input: expression evaluation failed ({message})"),
                err,
            )
        })?;
        return match outcome {
            interaction::EvalHookOutcome::Value(value) => Ok(value),
            interaction::EvalHookOutcome::InvalidExpression(error) => Err(error),
        };
    }

    // Fallback when no eval hook is installed (unit tests, native REPL).
    let parsed = call_builtin_async("str2double", &[Value::string(trimmed.to_string())])
        .await
        .map_err(|err| {
            let message = err.message().to_string();
            input_error_with_source(
                &INPUT_ERROR_INVALID_NUMERIC_EXPRESSION,
                format!("input: invalid numeric expression ({message})"),
                err,
            )
        })?;
    if matches!(parsed.value(), Value::Num(value) if value.is_nan()) {
        let mut error = input_error_with(
            &INPUT_ERROR_INVALID_NUMERIC_EXPRESSION,
            INPUT_ERROR_INVALID_NUMERIC_EXPRESSION.message,
        );
        error.failure_origin = crate::runtime_error::RuntimeFailureOrigin::LanguageEvaluation;
        return Err(error);
    }
    Ok(parsed)
}

/// Parse a single MATLAB scalar token into a [`Value`].
///
/// Returns [`Value::Bool`] for `true`/`false` (case-insensitive), [`Value::Num`]
/// for numeric literals and named constants (`pi`, `inf`, `nan`), and
/// `None` for anything that looks like a matrix, function call, or unknown
/// identifier.
///
/// Note: `e` is intentionally **not** handled here. It is not a MATLAB built-in
/// constant; typing `e` at an `input()` prompt would perform a variable lookup in
/// MATLAB and error if `e` is undefined. Unknown identifiers fall through to the
/// eval hook or `str2double`, which produce the correct error.
fn parse_scalar_value(s: &str) -> Option<Value> {
    match s.to_ascii_lowercase().as_str() {
        "true" => return Some(Value::Bool(true)),
        "false" => return Some(Value::Bool(false)),
        "pi" => return Some(Value::Num(std::f64::consts::PI)),
        "inf" | "+inf" | "infinity" | "+infinity" => return Some(Value::Num(f64::INFINITY)),
        "-inf" | "-infinity" => return Some(Value::Num(f64::NEG_INFINITY)),
        "nan" => return Some(Value::Num(f64::NAN)),
        _ => {}
    }
    // Plain numeric literals: integers, decimals, scientific notation, optional sign.
    // We reject anything containing brackets, commas, spaces (which would indicate a
    // matrix or an expression), or letters other than 'e'/'E' for exponent notation.
    let has_non_numeric = s.chars().any(|c| {
        matches!(c, '[' | ']' | ',' | ';' | '(' | ')' | ' ' | '\t')
            || (c.is_ascii_alphabetic() && c != 'e' && c != 'E' && c != 'i' && c != 'j')
    });
    if has_non_numeric {
        return None;
    }
    s.parse::<f64>().ok().map(Value::Num)
}

/// Parse a MATLAB matrix literal of the form `[elements]`.
///
/// Rows are separated by `;` and elements within a row by whitespace and/or `,`.
/// Every element must be a token accepted by [`parse_scalar_value`].
/// Returns `None` if the literal is malformed or contains non-scalar elements.
///
/// Output type mirrors MATLAB semantics:
/// - All-logical elements → [`Value::LogicalArray`]
/// - Any numeric element  → [`Value::Tensor`] (logical elements coerced to `f64`)
fn parse_matrix_literal(s: &str) -> Option<Value> {
    let inner = s.strip_prefix('[')?.strip_suffix(']')?;
    let inner = inner.trim();
    if inner.is_empty() {
        return Some(Value::Tensor(Tensor::zeros(vec![0, 0])));
    }

    let row_strs: Vec<&str> = inner.split(';').collect();
    let mut values: Vec<Value> = Vec::new();
    let mut nrows = 0usize;
    let mut ncols: Option<usize> = None;

    for row_str in &row_strs {
        let tokens: Vec<&str> = row_str
            .split(|c: char| c == ',' || c.is_ascii_whitespace())
            .filter(|t| !t.is_empty())
            .collect();
        if tokens.is_empty() {
            continue;
        }
        match ncols {
            None => ncols = Some(tokens.len()),
            Some(expected) if tokens.len() != expected => return None,
            _ => {}
        }
        for token in &tokens {
            values.push(parse_scalar_value(token)?);
        }
        nrows += 1;
    }

    let ncols = ncols.unwrap_or(0);
    if nrows == 0 || ncols == 0 {
        return Some(Value::Tensor(Tensor::zeros(vec![0, 0])));
    }
    // Scalar: preserve the exact type (Bool or Num) rather than always wrapping in Tensor.
    if nrows == 1 && ncols == 1 {
        return Some(values.remove(0));
    }

    // All-logical → LogicalArray; any numeric element → Tensor (bools coerced to f64).
    // `values` is in row-major order (row 0 left-to-right, then row 1, …), but both
    // Tensor and LogicalArray store data in column-major order (data[r + c*rows]).
    // Reorder so that column-major index maps to the correct element.
    let all_logical = values.iter().all(|v| matches!(v, Value::Bool(_)));
    if all_logical {
        let mut data: Vec<u8> = vec![0u8; nrows * ncols];
        for r in 0..nrows {
            for c in 0..ncols {
                let row_major_idx = r * ncols + c;
                let col_major_idx = r + c * nrows;
                data[col_major_idx] = match &values[row_major_idx] {
                    Value::Bool(b) => u8::from(*b),
                    _ => unreachable!(),
                };
            }
        }
        LogicalArray::new(data, vec![nrows, ncols])
            .ok()
            .map(Value::LogicalArray)
    } else {
        let mut data: Vec<f64> = vec![0f64; nrows * ncols];
        for r in 0..nrows {
            for c in 0..ncols {
                let row_major_idx = r * ncols + c;
                let col_major_idx = r + c * nrows;
                data[col_major_idx] = match &values[row_major_idx] {
                    Value::Num(f) => *f,
                    Value::Bool(b) => f64::from(u8::from(*b)),
                    _ => unreachable!(),
                };
            }
        }
        Tensor::new_2d(data, nrows, ncols).ok().map(Value::Tensor)
    }
}

#[cfg(test)]
#[path = "input/tests.rs"]
pub(crate) mod tests;
