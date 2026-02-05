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
        return Err(build_runtime_error("MATLAB:input:TooManyInputs").build());
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
                .with_source(err)
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
                Err(build_runtime_error("MATLAB:input:PromptMustBeRowVector").build())
            } else {
                Ok(ca.data.iter().collect())
            }
        }
        Value::String(text) => Ok(text),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err(build_runtime_error("MATLAB:input:PromptMustBeScalarString").build())
            }
        }
        other => {
            Err(build_runtime_error(format!("MATLAB:input:InvalidPromptType ({other:?})")).build())
        }
    }
}

async fn parse_string_flag(value: &Value) -> Result<bool, RuntimeError> {
    let gathered = gather_if_needed_async(value).await?;
    let text = match gathered {
        Value::CharArray(ca) if ca.rows == 1 => ca.data.iter().collect::<String>(),
        Value::String(s) => s,
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].clone(),
        other => {
            return Err(
                build_runtime_error(format!("MATLAB:input:InvalidStringFlag ({other:?})")).build(),
            )
        }
    };
    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("s") {
        Ok(true)
    } else {
        Err(build_runtime_error(format!("MATLAB:input:InvalidStringFlag ({trimmed})")).build())
    }
}

async fn parse_numeric_response(line: &str) -> Result<Value, RuntimeError> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed == "[]" {
        return Ok(Value::Tensor(Tensor::zeros(vec![0, 0])));
    }
    call_builtin_async("str2double", &[Value::String(trimmed.to_string())])
        .await
        .map_err(|err| {
            let message = err.message().to_string();
            build_runtime_error(format!("MATLAB:input:InvalidNumericExpression ({message})"))
                .with_source(err)
                .build()
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
    fn invalid_string_flag_errors_before_prompt() {
        push_queued_response(Ok(InteractionResponse::Line("ignored".into())));
        let prompt = Value::String("Ready?".to_string());
        let bad_flag = Value::String("not-string-mode".to_string());
        let err = futures::executor::block_on(input_builtin(vec![prompt, bad_flag])).unwrap_err();
        assert!(err.message().contains("InvalidStringFlag"));
    }
}
