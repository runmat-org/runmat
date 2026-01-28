//! MATLAB-compatible `regexpi` builtin for RunMat.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::regex::regexp::{self, RegexpEvaluation};
use crate::{build_runtime_error, make_cell, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::regex::regexpi")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "regexpi",
    op_kind: GpuOpKind::Custom("regex"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes on the CPU; GPU inputs are gathered before evaluation and results stay on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::regex::regexpi")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "regexpi",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Control-flow-heavy regex evaluation is not eligible for fusion.",
};

const BUILTIN_NAME: &str = "regexpi";

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

/// Evaluate `regexpi` with MATLAB-compatible defaults and return the shared regex evaluation handle.
pub async fn evaluate(
    subject: Value,
    pattern: Value,
    rest: &[Value],
) -> BuiltinResult<RegexpEvaluation> {
    let options = build_options(rest);
    regexp::evaluate_with(BUILTIN_NAME, subject, pattern, &options).await
}

#[runtime_builtin(
    name = "regexpi",
    category = "strings/regex",
    summary = "Case-insensitive regular expression matching with MATLAB-compatible outputs.",
    keywords = "regexpi,regex,pattern,ignorecase,match",
    accel = "sink",
    builtin_path = "crate::builtins::strings::regex::regexpi"
)]
async fn regexpi_builtin(
    subject: Value,
    pattern: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let evaluation = evaluate(subject, pattern, &rest).await?;
    let mut outputs = evaluation.outputs_for_single()?;
    if outputs.is_empty() {
        return Ok(Value::Num(0.0));
    }
    if outputs.len() == 1 {
        Ok(outputs.remove(0))
    } else {
        let len = outputs.len();
        make_cell(outputs, 1, len)
            .map_err(|err| runtime_error_for(format!("{BUILTIN_NAME}: {err}")))
    }
}

fn build_options(rest: &[Value]) -> Vec<Value> {
    let mut options: Vec<Value> = rest.to_vec();
    if !has_case_directive(rest) {
        options.push(Value::String("ignorecase".into()));
    }
    options
}

fn has_case_directive(values: &[Value]) -> bool {
    values.iter().any(|value| {
        matches!(
            option_name(value).as_deref(),
            Some("ignorecase") | Some("matchcase")
        )
    })
}

fn option_name(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            Some(ca.data.iter().collect::<String>().to_ascii_lowercase())
        }
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{CellArray, StringArray};

    use crate::builtins::common::test_support;

    fn evaluate(subject: Value, pattern: Value, rest: &[Value]) -> BuiltinResult<RegexpEvaluation> {
        futures::executor::block_on(super::evaluate(subject, pattern, rest))
    }

    fn run_regexpi_builtin(
        subject: Value,
        pattern: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        futures::executor::block_on(regexpi_builtin(subject, pattern, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_default_is_case_insensitive() {
        let eval = evaluate(
            Value::String("Abracadabra".into()),
            Value::String("a".into()),
            &[],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 4.0, 6.0, 8.0, 11.0]);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_match_output_ignores_case() {
        let eval = evaluate(
            Value::String("abcXYZ123".into()),
            Value::String("[a-z]+".into()),
            &[Value::String("match".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 1);
                let first = unsafe { &*ca.data[0].as_raw() };
                assert_eq!(first, &Value::String("abcXYZ".into()));
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_matchcase_overrides_default() {
        let eval = evaluate(
            Value::String("CaseTest".into()),
            Value::String("case".into()),
            &[Value::String("matchcase".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Tensor(t) => assert!(t.data.is_empty()),
            Value::Num(n) => assert_eq!(*n, 0.0),
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_builtin_match_output() {
        let result = run_regexpi_builtin(
            Value::String("FooBarBaz".into()),
            Value::String("bar".into()),
            vec![Value::String("match".into())],
        )
        .unwrap();
        match result {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 1);
                let entry = unsafe { &*ca.data[0].as_raw() };
                assert_eq!(entry, &Value::String("Bar".into()));
            }
            other => panic!("unexpected builtin output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_tokens_once_returns_structured_cells() {
        let eval = evaluate(
            Value::String("ID:AB12".into()),
            Value::String("(?<prefix>[a-z]+)(?<digits>\\d+)".into()),
            &[
                Value::String("tokens".into()),
                Value::String("names".into()),
                Value::String("once".into()),
            ],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 2);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 2);
                let first = unsafe { &*ca.data[0].as_raw() };
                let second = unsafe { &*ca.data[1].as_raw() };
                assert_eq!(first, &Value::String("AB".into()));
                assert_eq!(second, &Value::String("12".into()));
            }
            other => panic!("unexpected tokens output {other:?}"),
        }
        match &outputs[1] {
            Value::Struct(st) => {
                assert_eq!(st.fields.len(), 2);
                assert_eq!(st.fields.get("prefix"), Some(&Value::String("AB".into())));
                assert_eq!(st.fields.get("digits"), Some(&Value::String("12".into())));
            }
            other => panic!("unexpected names output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_force_cell_output_for_scalar_subject() {
        let eval = evaluate(
            Value::String("Hello".into()),
            Value::String("l".into()),
            &[
                Value::String("forcecelloutput".into()),
                Value::String("match".into()),
            ],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 1);
                let cell = unsafe { &*ca.data[0].as_raw() };
                match cell {
                    Value::Cell(inner) => {
                        assert_eq!(inner.data.len(), 2);
                        let first = unsafe { &*inner.data[0].as_raw() };
                        let second = unsafe { &*inner.data[1].as_raw() };
                        assert_eq!(first, &Value::String("l".into()));
                        assert_eq!(second, &Value::String("l".into()));
                    }
                    other => panic!("unexpected nested value {other:?}"),
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_token_extents_provide_indices() {
        let eval = evaluate(
            Value::String("ID:AB12".into()),
            Value::String("([A-Z]+)(\\d+)".into()),
            &[Value::String("tokenExtents".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 1);
                let matrix = unsafe { &*ca.data[0].as_raw() };
                match matrix {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![2, 2]);
                        assert_eq!(t.data, vec![4.0, 6.0, 5.0, 7.0]);
                    }
                    other => panic!("expected tensor for token extents, got {other:?}"),
                }
            }
            other => panic!("unexpected token extents output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_split_returns_segments() {
        let eval = evaluate(
            Value::String("Red,Green,BLUE".into()),
            Value::String(",".into()),
            &[Value::String("split".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 3);
                let parts: Vec<String> = ca
                    .data
                    .iter()
                    .map(|ptr| match unsafe { &*ptr.as_raw() } {
                        Value::String(s) => s.clone(),
                        other => panic!("expected string split part, got {other:?}"),
                    })
                    .collect();
                assert_eq!(parts, vec!["Red", "Green", "BLUE"]);
            }
            other => panic!("unexpected split output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_emptymatch_allow_keeps_zero_length_matches() {
        let eval = evaluate(
            Value::String("aba".into()),
            Value::String("b*".into()),
            &[
                Value::String("emptymatch".into()),
                Value::String("allow".into()),
            ],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]),
            other => panic!("expected tensor with match indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_string_array_preserves_shape() {
        let array =
            StringArray::new(vec!["OneTwo".into(), "THREEfour".into()], vec![2, 1]).unwrap();
        let eval = evaluate(
            Value::StringArray(array),
            Value::String("[a-z]+".into()),
            &[Value::String("match".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 1);
                assert_eq!(ca.data.len(), 2);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_cell_array_inputs_require_all_strings() {
        let handles = vec![
            unsafe {
                runmat_gc_api::GcPtr::from_raw(Box::into_raw(Box::new(Value::String("A".into()))))
            },
            unsafe { runmat_gc_api::GcPtr::from_raw(Box::into_raw(Box::new(Value::Num(1.0)))) },
        ];
        let cell = CellArray::new_handles(handles, 2, 1).unwrap();
        let err = evaluate(Value::Cell(cell), Value::String("a".into()), &[])
            .err()
            .expect("expected regexpi to reject non-text cell elements");
        let message = err.message().to_string();
        assert!(
            message.contains("cell array elements"),
            "unexpected error message: {message}"
        );
    }
}
