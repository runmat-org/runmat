//! MATLAB-compatible `input` builtin for line-oriented console interaction.

use runmat_builtins::{CharArray, LogicalArray, Tensor, Value};
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

    // Fast path 1: scalar literals, named constants, and logical keywords.
    // Handles the vast majority of input() use cases without touching the VM.
    if let Some(v) = parse_scalar_value(trimmed) {
        return Ok(v);
    }

    // Fast path 2: matrix/vector literals like `[1 2 3]`, `[1;2;3]`, `[true false]`.
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

    /// `e` is not a MATLAB built-in constant. The fast-path parser must not map
    /// it to Euler's number; it should fall through so the eval hook or
    /// `str2double` can handle it (which will NaN or error on an unknown identifier).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bare_e_is_not_eulers_number() {
        assert_eq!(parse_scalar_value("e"), None);
        assert_eq!(parse_scalar_value("E"), None);
    }

    /// `[1 e 3]` must not silently produce `[1.0, 2.718…, 3.0]`.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn matrix_with_bare_e_does_not_parse() {
        assert_eq!(parse_matrix_literal("[1 e 3]"), None);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn true_input_returns_logical_not_double() {
        push_queued_response(Ok(InteractionResponse::Line("true".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        assert_eq!(value, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn false_input_returns_logical_not_double() {
        push_queued_response(Ok(InteractionResponse::Line("false".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        assert_eq!(value, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bool_input_is_case_insensitive() {
        push_queued_response(Ok(InteractionResponse::Line("TRUE".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        assert_eq!(value, Value::Bool(true));
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
    fn logical_row_vector_parses_as_logical_array() {
        push_queued_response(Ok(InteractionResponse::Line("[true false]".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        match value {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![1, 2]);
                assert_eq!(la.data, vec![1, 0]);
            }
            other => panic!("expected LogicalArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_column_vector_parses_as_logical_array() {
        push_queued_response(Ok(InteractionResponse::Line("[true; false]".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        match value {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![2, 1]);
                assert_eq!(la.data, vec![1, 0]);
            }
            other => panic!("expected LogicalArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mixed_logical_and_numeric_coerces_to_double_tensor() {
        push_queued_response(Ok(InteractionResponse::Line("[true 2.0]".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.rows, 1);
                assert_eq!(t.cols, 2);
                assert_eq!(t.data, vec![1.0, 2.0]);
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn matrix_2x2_column_major_layout() {
        // [1 2; 3 4] → get2(r,c) must return element at row r, col c, not the transpose.
        // Column-major storage: data = [1, 3, 2, 4] (not the row-major [1, 2, 3, 4]).
        push_queued_response(Ok(InteractionResponse::Line("[1 2; 3 4]".into())));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        match value {
            Value::Tensor(t) => {
                assert_eq!(t.rows, 2);
                assert_eq!(t.cols, 2);
                assert_eq!(t.get2(0, 0).unwrap(), 1.0, "(0,0) should be 1");
                assert_eq!(t.get2(0, 1).unwrap(), 2.0, "(0,1) should be 2");
                assert_eq!(t.get2(1, 0).unwrap(), 3.0, "(1,0) should be 3");
                assert_eq!(t.get2(1, 1).unwrap(), 4.0, "(1,1) should be 4");
            }
            other => panic!("expected 2×2 tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_matrix_2x2_column_major_layout() {
        // [true false; false true] → column-major data = [1, 0, 0, 1].
        push_queued_response(Ok(InteractionResponse::Line(
            "[true false; false true]".into(),
        )));
        let value = futures::executor::block_on(input_builtin(vec![])).expect("input");
        match value {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![2, 2]);
                // column-major: col 0 first ([true, false]), then col 1 ([false, true])
                assert_eq!(la.data, vec![1, 0, 0, 1]);
            }
            other => panic!("expected 2×2 LogicalArray, got {other:?}"),
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
