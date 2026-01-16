//! MATLAB-compatible `input` builtin for line-oriented console interaction.

use runmat_builtins::{CharArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::interaction;
use crate::{call_builtin, gather_if_needed, runtime_error, BuiltinResult, RuntimeControlFlow};

const DEFAULT_PROMPT: &str = "Input: ";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "input", builtin_path = "crate::builtins::io::input")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "input"
category: "io"
keywords: ["input", "prompt", "stdin", "interactive", "pause", "gather"]
summary: "Prompt the user for input. Return either the typed text or a parsed numeric value."
references:
  - https://www.mathworks.com/help/matlab/ref/input.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Prompts are handled on the host. When `input` is called inside GPU-enabled workflows, only the textual arguments are gathered."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::input::tests"
  integration: null
---

# What does the `input` function do in MATLAB / RunMat?
`input(prompt)` displays `prompt`, waits for the user to type a line of text, and then returns the
evaluated value. When you pass `'s'` as the second argument, `input` skips numeric parsing and
returns the raw text as a character row vector. RunMat mirrors MATLAB’s single-line prompt/response
flow so scripts that rely on interactive questions continue to work.

## Behaviour in RunMat
- Prompts accept character row vectors and string scalars. When omitted, RunMat uses `"Input: "`.
- The function always blocks until the host supplies a line of text (via the REPL, CLI pipe,
  or wasm bindings). `stdinRequested` events surface the outstanding prompt to JS hosts so they can
  resume later.
- When `'s'` (or `"s"`) is provided, the return value is a character array containing exactly what
  the user typed (without the trailing newline).
- Without `'s'`, RunMat forwards the response through the same numeric parser that backs `str2double`.
  This covers scalar numbers, MATLAB-style vector/matrix literals, `Inf`, `NaN`, and complex tokens.
  Arbitrary expressions (for example `1+rand()`) are not evaluated automatically—read the text with
  `'s'` and pass it to `eval` once that builtin lands.
- Empty responses (just pressing Enter) return an empty `[]` double, consistent with MATLAB.
- When the numeric parser rejects the text, RunMat raises `MATLAB:input:InvalidNumericExpression`
  so callers can present a friendly retry loop.

## Examples
```matlab
value = input("Enter a scalar: ");
```

```matlab
name = input("Your name? ", "s");
fprintf("Hello, %s!\n", name);
```

```matlab
vec = input("Enter a row vector like [1 2 3]: ");
```

```matlab
raw = input("Type anything and I will echo it back: ", "s");
disp("You typed: " + raw)
```

## FAQ

### Does RunMat evaluate arbitrary MATLAB expressions?
Not yet. The numeric branch reuses the `str2double` parser, which supports MATLAB numeric literals
(scalars, vectors, matrices, `Inf`, `NaN`, complex pairs) but does not run arbitrary code. When you
need to evaluate expressions such as `1+sin(pi/4)`, capture the text via `input(..., 's')` and pass
it to `eval` in a follow-up release.

### What happens if the user just presses Enter?
RunMat returns the empty double `[]`, matching MATLAB’s `input` behaviour.

### Can I call `input` inside GPU workloads?
Yes. Prompts always flow through the host console/UI. GPU-resident prompt arguments are gathered
once before display, and the response itself is always a host value.
"#;

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

#[runtime_builtin(name = "input", builtin_path = "crate::builtins::io::input")]
fn input_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.len() > 2 {
        return Err(runtime_error("MATLAB:input:TooManyInputs").build().into());
    }

    let mut prompt_index = if args.is_empty() { None } else { Some(0usize) };
    let mut parsed_flag: Option<bool> = None;

    if let Some(idx) = if args.len() == 2 { Some(1usize) } else { None } {
        match parse_string_flag(&args[idx]) {
            Ok(flag) => parsed_flag = Some(flag),
            Err(original_err) => {
                if let Some(prompt_idx) = prompt_index {
                    match parse_string_flag(&args[prompt_idx]) {
                        Ok(swapped_flag) => {
                            parsed_flag = Some(swapped_flag);
                            prompt_index = Some(idx);
                        }
                        Err(_) => {
                            return Err(original_err.into());
                        }
                    }
                } else {
                    return Err(original_err.into());
                }
            }
        }
    }

    let prompt = if let Some(idx) = prompt_index {
        parse_prompt(&args[idx])?
    } else {
        DEFAULT_PROMPT.to_string()
    };
    let return_string = parsed_flag.unwrap_or(false);
    let line = interaction::request_line(&prompt, true).map_err(|flow| match flow {
        RuntimeControlFlow::Suspend(pending) => RuntimeControlFlow::Suspend(pending),
        RuntimeControlFlow::Error(err) => RuntimeControlFlow::Error(
            runtime_error(format!("input: {}", err.message()))
                .with_source(err)
                .build(),
        ),
    })?;
    if return_string {
        return Ok(Value::CharArray(CharArray::new_row(&line)));
    }
    parse_numeric_response(&line)
}

fn parse_prompt(value: &Value) -> Result<String, RuntimeControlFlow> {
    let gathered = gather_if_needed(value)?;
    match gathered {
        Value::CharArray(ca) => {
            if ca.rows != 1 {
                Err(runtime_error("MATLAB:input:PromptMustBeRowVector")
                    .build()
                    .into())
            } else {
                Ok(ca.data.iter().collect())
            }
        }
        Value::String(text) => Ok(text),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err(runtime_error("MATLAB:input:PromptMustBeScalarString")
                    .build()
                    .into())
            }
        }
        other => Err(runtime_error(format!("MATLAB:input:InvalidPromptType ({other:?})"))
            .build()
            .into()),
    }
}

fn parse_string_flag(value: &Value) -> Result<bool, RuntimeControlFlow> {
    let gathered = gather_if_needed(value)?;
    let text = match gathered {
        Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect::<String>(),
        Value::String(s) => s,
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
        other => {
            return Err(runtime_error(format!("MATLAB:input:InvalidStringFlag ({other:?})"))
                .build()
                .into())
        }
    };
    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("s") {
        Ok(true)
    } else {
        Err(runtime_error(format!("MATLAB:input:InvalidStringFlag ({trimmed})"))
            .build()
            .into())
    }
}

fn parse_numeric_response(line: &str) -> Result<Value, RuntimeControlFlow> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed == "[]" {
        return Ok(Value::Tensor(Tensor::zeros(vec![0, 0])));
    }
    call_builtin("str2double", &[Value::String(trimmed.to_string())]).map_err(|err| {
        runtime_error(format!("MATLAB:input:InvalidNumericExpression ({})", err.message()))
            .with_source(err)
            .build()
            .into()
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::interaction::{push_queued_response, InteractionResponse};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_input_parses_scalar() {
        push_queued_response(Ok(InteractionResponse::Line("41".into())));
        let value = input_builtin(vec![]).expect("input");
        assert_eq!(value, Value::Num(41.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_mode_returns_char_row() {
        push_queued_response(Ok(InteractionResponse::Line("RunMat".into())));
        let prompt = Value::CharArray(CharArray::new_row("Name: "));
        let mode = Value::String("s".to_string());
        let value = input_builtin(vec![prompt, mode]).expect("input");
        assert_eq!(value, Value::CharArray(CharArray::new_row("RunMat")));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_response_returns_empty_tensor() {
        push_queued_response(Ok(InteractionResponse::Line("   ".into())));
        let value = input_builtin(vec![]).expect("input");
        match value {
            Value::Tensor(t) => assert!(t.data.is_empty()),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn invalid_string_flag_errors_before_prompt() {
        push_queued_response(Ok(InteractionResponse::Line("ignored".into())));
        let prompt = Value::String("Ready?".to_string());
        let bad_flag = Value::String("not-string-mode".to_string());
        let err = input_builtin(vec![prompt, bad_flag]).unwrap_err();
        assert!(err.contains("InvalidStringFlag"));
    }
}
