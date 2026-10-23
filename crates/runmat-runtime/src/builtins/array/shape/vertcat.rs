//! MATLAB-compatible `vertcat` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::{build_runtime_error, RuntimeError};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::vertcat")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "vertcat",
    op_kind: GpuOpKind::Custom("cat"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("cat")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to cat(dim=1); providers without cat fall back to host gather + upload.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::vertcat")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "vertcat",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Concatenation materialises outputs immediately, terminating fusion pipelines.",
};

fn vertcat_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("vertcat").build()
}

#[runtime_builtin(
    name = "vertcat",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::array::shape::vertcat"
)]
async fn vertcat_builtin(
    context: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    if args.is_empty() {
        return crate::execution::RootedValue::in_optional_context(
            empty_double()?,
            context.as_ref(),
        );
    }
    if args.len() == 1 {
        return crate::execution::RootedValue::in_optional_context(
            args[0].clone(),
            context.as_ref(),
        );
    }

    let prefix = crate::execution::RootedValue::in_optional_context(
        Value::Int(IntValue::I32(1)),
        context.as_ref(),
    )?;
    let forwarded = crate::builtins::common::mapped_callable::prepend_rooted_argument(
        &args,
        prefix,
        context.as_ref(),
    )
    .await?;
    match crate::dispatcher::call_retained_builtin_async("cat", forwarded, context.as_ref()).await {
        Ok(value) => Ok(value),
        Err(err) => Err(adapt_cat_error(err)),
    }
}

pub(crate) fn is_canonical_host_binding(binding: crate::builtin::RuntimeBuiltinBinding) -> bool {
    binding.identity.builtin.name == "vertcat"
        && binding.identity.variant == "default"
        && std::ptr::fn_addr_eq(
            binding.implementation,
            __rt_wrap_vertcat_builtin as crate::builtin::RuntimeBuiltinImplementation,
        )
        && binding.continuation_layout.is_some_and(|layout| {
            std::ptr::fn_addr_eq(
                layout,
                __rt_owned_layout_vertcat_builtin
                    as crate::builtin::RuntimeBuiltinContinuationLayout,
            )
        })
}

fn empty_double() -> crate::BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|e| vertcat_error(format!("vertcat: {e}")))
}

fn adapt_cat_error(mut error: RuntimeError) -> RuntimeError {
    let message = error.message.clone();
    let adjusted = if let Some(rest) = message.strip_prefix("cat:") {
        format!("vertcat:{rest}")
    } else if let Some(idx) = message.find("cat:") {
        let rest = &message[idx + 4..];
        format!("vertcat:{rest}")
    } else if message.starts_with("vertcat:") {
        message
    } else {
        format!("vertcat: {message}")
    };
    error.message = adjusted;
    error.context = error.context.with_builtin("vertcat");
    error
}

/// Selected pure-host literal path, using the same `cat` assembly and
/// `vertcat` error translation as the async builtin route.
pub(crate) fn concatenate_host_values(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match args.len() {
        0 => empty_double(),
        1 => Ok(args.into_iter().next().expect("one argument")),
        _ => crate::builtins::array::shape::cat::concatenate_host_values(0, args)
            .map_err(adapt_cat_error),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn vertcat_builtin(args: Vec<Value>) -> crate::BuiltinResult<crate::execution::RootedValue> {
        block_on(super::vertcat_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }
    use crate::builtins::common::test_support;
    use runmat_value::{
        CellArray, CharArray, ComplexTensor, IntegerStorage, LogicalArray, StringArray, Tensor,
    };

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_invocation_returns_zero_by_zero() {
        let result = vertcat_builtin(Vec::new()).expect("vertcat");
        match result.value() {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![0, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_argument_round_trips() {
        let value = Value::Num(42.0);
        let result = vertcat_builtin(vec![value.clone()]).expect("vertcat");
        assert_eq!(result.value(), &value);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn true_empty_operand_is_neutral_for_dynamic_growth() {
        let empty = Tensor::new(Vec::new(), vec![0, 0]).expect("empty");
        let result = vertcat_builtin(vec![Value::Tensor(empty), Value::Num(2.0)]).expect("vertcat");
        assert_eq!(result.value(), &Value::Num(2.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_vertical_concat() {
        let top = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let bottom = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let result =
            vertcat_builtin(vec![Value::Tensor(top), Value::Tensor(bottom)]).expect("vertcat");
        match result.value() {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![4, 2]);
                assert_eq!(
                    t.materialize_f64(),
                    vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn vertcat_preserves_leftmost_integer_class() {
        let top = Tensor::new_integer(IntegerStorage::I8(vec![12]), vec![1, 1]).expect("top");
        let result = vertcat_builtin(vec![
            Value::Tensor(top),
            Value::Int(IntValue::U64(u64::MAX)),
        ])
        .expect("vertcat");

        let Value::Tensor(output) = result.value() else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I8(vec![12, i8::MAX]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_arrays_concatenate_rows() {
        let top = CharArray::from_unicode_scalars("RunMat".chars().collect(), 1, 6).unwrap();
        let bottom = CharArray::from_unicode_scalars("Rocks!".chars().collect(), 1, 6).unwrap();
        let result = vertcat_builtin(vec![Value::CharArray(top), Value::CharArray(bottom)])
            .expect("vertcat");
        match result.value() {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 2);
                assert_eq!(arr.cols, 6);
                let first: String = runmat_value::decode_utf16_code_units(&arr.data()[..6])
                    .expect("valid UTF-16 fixture text");
                let second: String = runmat_value::decode_utf16_code_units(&arr.data()[6..])
                    .expect("valid UTF-16 fixture text");
                assert_eq!(first, "RunMat");
                assert_eq!(second, "Rocks!");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_arrays_concatenate_rows() {
        let header = StringArray::new(vec!["Name".into(), "Score".into()], vec![1, 2]).unwrap();
        let rows = StringArray::new(vec!["Alice".into(), "98".into()], vec![1, 2]).unwrap();
        let result = vertcat_builtin(vec![Value::StringArray(header), Value::StringArray(rows)])
            .expect("vertcat");
        match result.value() {
            Value::StringArray(arr) => {
                assert_eq!(arr.shape, vec![2, 2]);
                assert_eq!(arr.data(), vec!["Name", "Alice", "Score", "98"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_and_char_arrays_vertical_concat() {
        let top = StringArray::new(vec!["alpha".into()], vec![1, 1]).unwrap();
        let bottom = CharArray::new_row("beta");
        let result = vertcat_builtin(vec![Value::StringArray(top), Value::CharArray(bottom)])
            .expect("vertcat mixed");
        match result.value() {
            Value::StringArray(arr) => {
                assert_eq!(arr.shape, vec![2, 1]);
                assert_eq!(arr.data(), vec!["alpha", "beta"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mismatched_columns_error_mentions_vertcat() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]).unwrap();
        let err = vertcat_builtin(vec![Value::Tensor(a), Value::Tensor(b)]).unwrap_err();
        assert!(err.starts_with("vertcat:"));
        assert!(err.contains("dimension 2 mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vertcat_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let top = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let bottom = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let view_top = runmat_accelerate_api::HostTensorView {
                data: &top.materialize_f64(),
                shape: top.shape(),
            };
            let view_bottom = runmat_accelerate_api::HostTensorView {
                data: &bottom.materialize_f64(),
                shape: bottom.shape(),
            };
            let h_top = provider.upload(&view_top).expect("upload top");
            let h_bottom = provider.upload(&view_bottom).expect("upload bottom");
            let result = vertcat_builtin(vec![Value::GpuTensor(h_top), Value::GpuTensor(h_bottom)])
                .expect("vertcat");
            let gathered = result.with_owned(test_support::gather).expect("gather");
            assert_eq!(gathered.shape(), vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_arrays_concatenate_rows() {
        let first = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let second = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        let result = vertcat_builtin(vec![
            Value::LogicalArray(first),
            Value::LogicalArray(second),
        ])
        .expect("vertcat logical");
        match result.value() {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data, vec![1, 0, 0, 1, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_arrays_concatenate_rows() {
        let first = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2, 1]).unwrap();
        let second = ComplexTensor::new(vec![(5.0, 6.0), (7.0, 8.0)], vec![2, 1]).unwrap();
        let result = vertcat_builtin(vec![
            Value::ComplexTensor(first),
            Value::ComplexTensor(second),
        ])
        .expect("vertcat complex");
        match result.value() {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape(), vec![4, 1]);
                assert_eq!(
                    ct.materialize_f64(),
                    vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_arrays_concatenate_rows() {
        let first = CellArray::new(vec![Value::Num(1.0), Value::from("low")], 1, 2).unwrap();
        let second = CellArray::new(vec![Value::Num(2.0), Value::from("high")], 1, 2).unwrap();
        let result =
            vertcat_builtin(vec![Value::Cell(first), Value::Cell(second)]).expect("vertcat cell");
        match result.value() {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert_eq!(cell.cols, 2);
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn vertcat_like_gpu_from_host_inputs() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let prototype = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_view = runmat_accelerate_api::HostTensorView {
                data: &prototype.materialize_f64(),
                shape: prototype.shape(),
            };
            let proto_handle = provider.upload(&proto_view).expect("upload proto");

            let top = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let bottom = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let result = vertcat_builtin(vec![
                Value::Tensor(top),
                Value::Tensor(bottom),
                Value::from("like"),
                Value::GpuTensor(proto_handle),
            ])
            .expect("vertcat like");
            match result.value() {
                Value::GpuTensor(_) => {}
                other => panic!("expected gpu tensor, got {other:?}"),
            }
            let gathered = result.with_owned(test_support::gather).expect("gather");
            assert_eq!(gathered.shape(), vec![4, 1]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn vertcat_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let top = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
        let bottom = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();

        let cpu_value = vertcat_builtin(vec![
            Value::Tensor(top.clone()),
            Value::Tensor(bottom.clone()),
        ])
        .expect("cpu vertcat");
        let expected = match cpu_value.value() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view_top = runmat_accelerate_api::HostTensorView {
            data: &top.materialize_f64(),
            shape: top.shape(),
        };
        let view_bottom = runmat_accelerate_api::HostTensorView {
            data: &bottom.materialize_f64(),
            shape: bottom.shape(),
        };
        let ht = provider.upload(&view_top).expect("upload top");
        let hb = provider.upload(&view_bottom).expect("upload bottom");
        let gpu_value =
            vertcat_builtin(vec![Value::GpuTensor(ht), Value::GpuTensor(hb)]).expect("gpu vertcat");
        let gathered = gpu_value.with_owned(test_support::gather).expect("gather");
        assert_eq!(gathered.shape(), expected.shape());
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    }
}
