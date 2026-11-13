//! MATLAB-compatible `regexpi` builtin for RunMat.

#[cfg(test)]
use runmat_builtins::REGEXPI_DESCRIPTOR;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinGpuSpec, ConstantStrategy, GpuOpKind, ReductionNaN, ResidencyPolicy,
};
use crate::builtins::strings::regex::regexp;
#[cfg(test)]
use crate::builtins::strings::regex::regexp::RegexpEvaluation;
#[cfg(test)]
use crate::BuiltinResult;
use crate::{build_runtime_error, make_cell, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::regex::regexpi")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "regexpi",
    op_kind: GpuOpKind::Custom("regex"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The runtime rejects numeric and resident text roles before provider access; matching remains on the host.",
};

const BUILTIN_NAME: &str = "regexpi";

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

/// Evaluate `regexpi` with MATLAB-compatible defaults and return the shared regex evaluation handle.
#[cfg(test)]
pub async fn evaluate(
    subject: Value,
    pattern: Value,
    rest: &[Value],
) -> BuiltinResult<RegexpEvaluation> {
    regexp::evaluate_with_default(
        BUILTIN_NAME,
        subject,
        pattern,
        crate::call::FunctionArgumentView::from_slice(rest),
        true,
    )
    .await
}

#[runtime_builtin(
    name = "regexpi",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::regex::regexpi"
)]
async fn regexpi_builtin(
    subject: Value,
    pattern: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let evaluation =
        regexp::evaluate_with_default(BUILTIN_NAME, subject, pattern, rest.view(), true).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        let mut outputs = evaluation.outputs_for_multi()?;
        if outputs.len() > out_count {
            outputs.truncate(out_count);
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_value::{CellArray, StringArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = REGEXPI_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"out = regexpi(subject, pattern)"));
        assert!(labels.contains(&"out = regexpi(subject, pattern, options...)"));
        assert!(labels.contains(
            &"[start,end_idx,token_extents,match,tokens,names,split] = regexpi(subject, pattern, options...)"
        ));
    }

    fn evaluate(subject: Value, pattern: Value, rest: &[Value]) -> BuiltinResult<RegexpEvaluation> {
        futures::executor::block_on(super::evaluate(subject, pattern, rest))
    }

    fn run_regexpi_builtin(
        subject: Value,
        pattern: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        futures::executor::block_on(regexpi_builtin(
            subject,
            pattern,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
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
                assert_eq!(t.materialize_f64(), vec![1.0, 4.0, 6.0, 8.0, 11.0]);
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
                assert_eq!(ca.data().len(), 1);
                let first = &ca.data()[0];
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
            Value::Tensor(t) => assert!(t.materialize_f64().is_empty()),
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
                assert_eq!(ca.data().len(), 1);
                let entry = &ca.data()[0];
                assert_eq!(entry, &Value::String("Bar".into()));
            }
            other => panic!("unexpected builtin output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_default_third_output_is_token_extents() {
        let _output_count =
            crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let result = run_regexpi_builtin(
            Value::String("AB12".into()),
            Value::String("([a-z]+)(\\d+)".into()),
            Vec::new(),
        )
        .unwrap();
        match result {
            Value::OutputList(outputs) => {
                assert_eq!(outputs.len(), 3);
                match &outputs[2] {
                    Value::Cell(matches) => {
                        assert_eq!(matches.data().len(), 1);
                        match &matches.data()[0] {
                            Value::Tensor(extents) => {
                                assert_eq!(extents.shape(), &[2, 2]);
                                assert_eq!(extents.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
                            }
                            other => panic!("expected token extents, got {other:?}"),
                        }
                    }
                    other => panic!("expected token extent cell, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
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
                let first = &ca.data()[0];
                let second = &ca.data()[1];
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
                let cell = &ca.data()[0];
                match cell {
                    Value::Cell(inner) => {
                        assert_eq!(inner.data().len(), 2);
                        let first = &inner.data()[0];
                        let second = &inner.data()[1];
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
                assert_eq!(ca.data().len(), 1);
                let matrix = &ca.data()[0];
                match matrix {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape(), &[2, 2]);
                        assert_eq!(t.materialize_f64(), vec![4.0, 6.0, 5.0, 7.0]);
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
                assert_eq!(ca.data().len(), 3);
                let parts: Vec<String> = ca
                    .data()
                    .iter()
                    .map(|ptr| match ptr {
                        Value::String(s) => s.as_str().expect("nonmissing split part").to_owned(),
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
            Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]),
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
                assert_eq!(ca.data().len(), 2);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_cell_array_inputs_require_all_strings() {
        let cell = CellArray::new(vec![Value::String("A".into()), Value::Num(1.0)], 2, 1).unwrap();
        let err = evaluate(Value::Cell(cell), Value::String("a".into()), &[])
            .err()
            .expect("expected regexpi to reject non-text cell elements");
        let message = err.message().to_string();
        assert!(
            message.contains("cell array elements"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_case_default_uses_effective_consumed_options() {
        let call = |rest: Vec<Value>| {
            futures::executor::block_on(regexpi_builtin(
                Value::String("A".into()),
                Value::String("a".into()),
                crate::builtins::common::test_support::builtin_arguments(rest),
            ))
            .expect("registered regexpi")
        };
        for rest in [
            vec![],
            vec![Value::String("warnings".into())],
            vec![
                Value::String("warnings".into()),
                Value::String("matchcase".into()),
            ],
        ] {
            assert_eq!(call(rest), Value::Num(1.0));
        }
        let sensitive = call(vec![Value::String("matchcase".into())]);
        assert_ne!(sensitive, Value::Num(1.0));
    }
}
