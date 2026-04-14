//! MATLAB-compatible `input` builtin for line-oriented console interaction.

use runmat_builtins::{CharArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::interaction;
use crate::{
    build_runtime_error, call_builtin_async, gather_if_needed_async, BuiltinResult, RuntimeError,
};

const DEFAULT_PROMPT: &str = "Input: ";

fn input_error(identifier: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(identifier.to_string())
        .with_builtin("input")
        .build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::input")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "input",
    op_kind: GpuOpKind::Custom("interaction"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Prompts execute on the host. Input text is always delivered via the host handler; GPU tensors are only gathered when used as prompt strings.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::input")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "input",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Side-effecting builtin; excluded from fusion plans.",
};

#[runtime_builtin(
    name = "input",
    type_resolver(crate::builtins::io::type_resolvers::input_type),
    builtin_path = "crate::builtins::io::input"
)]
async fn input_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() > 2 {
        return Err(input_error(
            "RunMat:input:TooManyInputs",
            "input: too many inputs",
        ));
    }

    let mut prompt_index = if args.is_empty() { None } else { Some(0usize) };
    let mut parsed_flag: Option<bool> = None;

    if let Some(idx) = if args.len() == 2 { Some(1usize) } else { None } {
        match parse_string_flag(&args[idx]).await {
            Ok(flag) => parsed_flag = Some(flag),
            Err(original_err) => {
                if let Some(prompt_idx) = prompt_index {
                    match parse_string_flag(&args[prompt_idx]).await {
                        Ok(swapped_flag) => {
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
    let return_string = parsed_flag.unwrap_or(false);
    let line = interaction::request_line_async(&prompt, true)
        .await
        .map_err(|err| {
            let message = err.message().to_string();
            build_runtime_error(format!("input: {message}"))
                .with_identifier("RunMat:input:InteractionFailed")
                .with_source(err)
                .with_builtin("input")
                .build()
        })?;
    if return_string {
        return Ok(Value::CharArray(CharArray::new_row(&line)));
    }
    parse_numeric_response(&line).await
}

async fn parse_prompt(value: &Value) -> Result<String, RuntimeError> {
    let gathered = gather_if_needed_async(value).await?;
    match gathered {
        Value::CharArray(ca) => {
            if ca.rows != 1 {
                Err(input_error(
                    "RunMat:input:PromptMustBeRowVector",
                    "input: prompt must be a row vector",
                ))
            } else {
                Ok(ca.data.iter().collect())
            }
        }
        Value::String(text) => Ok(text),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err(input_error(
                    "RunMat:input:PromptMustBeScalarString",
                    "input: prompt must be a scalar string",
                ))
            }
        }
        other => Err(input_error(
            "RunMat:input:InvalidPromptType",
            format!("input: invalid prompt type ({other:?})"),
        )),
    }
}

async fn parse_string_flag(value: &Value) -> Result<bool, RuntimeError> {
    let gathered = gather_if_needed_async(value).await?;
    let text = match gathered {
        Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect::<String>(),
        Value::String(s) => s,
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
        other => {
            return Err(input_error(
                "RunMat:input:InvalidStringFlag",
                format!("input: invalid string flag ({other:?})"),
            ))
        }
    };
    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("s") {
        Ok(true)
    } else {
        Err(input_error(
            "RunMat:input:InvalidStringFlag",
            format!("input: invalid string flag ({trimmed})"),
        ))
    }
}

async fn parse_numeric_response(line: &str) -> Result<Value, RuntimeError> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed == "[]" {
        return Ok(Value::Tensor(Tensor::zeros(vec![0, 0])));
    }

    // Fast path 1: scalar literals and named constants.
    // Handles the vast majority of input() use cases without touching the VM.
    if let Some(f) = parse_scalar_token(trimmed) {
        return Ok(Value::Num(f));
    }

    // Fast path 2: matrix/vector literals like `[1 2 3]`, `[1;2;3]`, `[1 2;3 4]`.
    // Avoids recursive interpret() calls for this common case.
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        if let Some(v) = parse_matrix_literal(trimmed) {
            return Ok(v);
        }
    }

    // Full eval path for complex expressions (`sqrt(2)`, `pi/2`, `ones(3)`, etc.).
    // The eval hook is only safe to call when the executor can handle re-entrant
    // polls (e.g. the WASM async runtime). On native the fast paths above cover
    // the common cases; truly complex expressions fall back to str2double here.
    if let Some(hook) = interaction::current_eval_hook() {
        return hook(trimmed.to_string()).await.map_err(|err| {
            let message = err.message().to_string();
            build_runtime_error(format!("input: invalid expression ({message})"))
                .with_identifier("RunMat:input:EvalFailed")
                .with_source(err)
                .with_builtin("input")
                .build()
        });
    }

    // Fallback when no eval hook is installed (unit tests, native REPL).
    call_builtin_async("str2double", &[Value::String(trimmed.to_string())])
        .await
        .map_err(|err| {
            let message = err.message().to_string();
            build_runtime_error(format!("input: invalid numeric expression ({message})"))
                .with_identifier("RunMat:input:InvalidNumericExpression")
                .with_source(err)
                .with_builtin("input")
                .build()
        })
}

/// Parse a single token that represents a MATLAB numeric scalar or named constant.
/// Returns `None` if the token is not a recognisable scalar (e.g. it contains
/// brackets, commas, or looks like a function call).
fn parse_scalar_token(s: &str) -> Option<f64> {
    // Named constants (case-insensitive to match MATLAB behaviour).
    match s.to_ascii_lowercase().as_str() {
        "pi" => return Some(std::f64::consts::PI),
        "inf" | "+inf" | "infinity" | "+infinity" => return Some(f64::INFINITY),
        "-inf" | "-infinity" => return Some(f64::NEG_INFINITY),
        "nan" => return Some(f64::NAN),
        "e" => return Some(std::f64::consts::E),
        "true" => return Some(1.0),
        "false" => return Some(0.0),
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
    s.parse::<f64>().ok()
}

/// Parse a MATLAB matrix literal of the form `[elements]`.
///
/// Rows are separated by `;` and elements within a row by whitespace and/or `,`.
/// Every element must be a token accepted by [`parse_scalar_token`].
/// Returns `None` if the literal is malformed or contains non-scalar elements.
fn parse_matrix_literal(s: &str) -> Option<Value> {
    let inner = s.strip_prefix('[')?.strip_suffix(']')?;
    let inner = inner.trim();
    if inner.is_empty() {
        return Some(Value::Tensor(Tensor::zeros(vec![0, 0])));
    }

    let row_strs: Vec<&str> = inner.split(';').collect();
    let mut data: Vec<f64> = Vec::new();
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
            data.push(parse_scalar_token(token)?);
        }
        nrows += 1;
    }

    let ncols = ncols.unwrap_or(0);
    if nrows == 0 || ncols == 0 {
        return Some(Value::Tensor(Tensor::zeros(vec![0, 0])));
    }
    if nrows == 1 && ncols == 1 {
        return Some(Value::Num(data[0]));
    }
    Tensor::new_2d(data, nrows, ncols).ok().map(Value::Tensor)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::interaction::{push_queued_response, InteractionResponse};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_input_parses_scalar() {
        push_queued_response(Ok(InteractionResponse::Line("41".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        assert_eq!(value, Value::Num(41.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_mode_returns_char_row() {
        push_queued_response(Ok(InteractionResponse::Line("RunMat".into())));
        let prompt = Value::CharArray(CharArray::new_row("Name: "));
        let mode = Value::String("s".to_string());
        let value = futures::executor::block_on(input_builtin(vec![prompt, mode])).expect("input");
        assert_eq!(value, Value::CharArray(CharArray::new_row("RunMat")));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_response_returns_empty_tensor() {
        push_queued_response(Ok(InteractionResponse::Line("   ".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        match value {
            Value::Tensor(t) => assert!(t.data.is_empty()),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn matrix_literal_parses_without_eval_hook() {
        // The fast-path parser handles `[1 2 3]` directly, so no eval hook (and
        // therefore no recursive interpret() call) is needed.
        push_queued_response(Ok(InteractionResponse::Line("[1 2 3]".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.rows, 1);
                assert_eq!(t.cols, 3);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected 1×3 tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn named_constants_parse_without_eval_hook() {
        push_queued_response(Ok(InteractionResponse::Line("pi".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        assert_eq!(value, Value::Num(std::f64::consts::PI));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn column_vector_parses_without_eval_hook() {
        push_queued_response(Ok(InteractionResponse::Line("[1;2;3]".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.rows, 3);
                assert_eq!(t.cols, 1);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected 3×1 tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_string_flag_errors_before_prompt() {
        push_queued_response(Ok(InteractionResponse::Line("ignored".into())));
        let prompt = Value::String("Ready?".to_string());
        let bad_flag = Value::String("not-string-mode".to_string());
        let err = futures::executor::block_on(input_builtin(vec![prompt, bad_flag])).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:input:InvalidStringFlag"));
    }
}
