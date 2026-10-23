//! MATLAB-compatible `horzcat` builtin with GPU-aware semantics for RunMat.

use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};

use crate::{build_runtime_error, RuntimeError};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::horzcat")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "horzcat",
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
    notes: "Delegates to cat(dim=2); providers without cat fall back to host gather + upload.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::horzcat")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "horzcat",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Concatenation materialises outputs immediately, terminating fusion pipelines.",
};

fn horzcat_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("horzcat").build()
}

#[runtime_builtin(
    name = "horzcat",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::array::shape::horzcat"
)]
async fn horzcat_builtin(
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
        Value::Int(IntValue::I32(2)),
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
    binding.identity.builtin.name == "horzcat"
        && binding.identity.variant == "default"
        && std::ptr::fn_addr_eq(
            binding.implementation,
            __rt_wrap_horzcat_builtin as crate::builtin::RuntimeBuiltinImplementation,
        )
        && binding.continuation_layout.is_some_and(|layout| {
            std::ptr::fn_addr_eq(
                layout,
                __rt_owned_layout_horzcat_builtin
                    as crate::builtin::RuntimeBuiltinContinuationLayout,
            )
        })
}

fn empty_double() -> crate::BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|e| horzcat_error(format!("horzcat: {e}")))
}

fn adapt_cat_error(mut error: RuntimeError) -> RuntimeError {
    let message = error.message.clone();
    let adjusted = if let Some(rest) = message.strip_prefix("cat:") {
        format!("horzcat:{rest}")
    } else if let Some(idx) = message.find("cat:") {
        let rest = &message[idx + 4..];
        format!("horzcat:{rest}")
    } else if message.starts_with("horzcat:") {
        message
    } else {
        format!("horzcat: {message}")
    };
    error.message = adjusted;
    error.context = error.context.with_builtin("horzcat");
    error
}

/// Selected pure-host literal path. It shares `cat`'s exact class/shape
/// assembly, including the public `horzcat` diagnostic adaptation.
pub(crate) fn concatenate_host_values(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    match args.len() {
        0 => empty_double(),
        1 => Ok(args.into_iter().next().expect("one argument")),
        _ => crate::builtins::array::shape::cat::concatenate_host_values(1, args)
            .map_err(adapt_cat_error),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn horzcat_builtin(args: Vec<Value>) -> crate::BuiltinResult<crate::execution::RootedValue> {
        block_on(super::horzcat_builtin(
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
        let result = horzcat_builtin(Vec::new()).expect("horzcat");
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
        let result = horzcat_builtin(vec![Value::Num(3.5)]).expect("horzcat");
        assert_eq!(result.value(), &Value::Num(3.5));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn true_empty_operand_is_neutral_for_dynamic_growth() {
        let empty = Tensor::new(Vec::new(), vec![0, 0]).expect("empty");
        let result = horzcat_builtin(vec![Value::Tensor(empty), Value::Num(1.0)]).expect("horzcat");
        assert_eq!(result.value(), &Value::Num(1.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_horizontal_concat() {
        let left = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let right = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
        let result =
            horzcat_builtin(vec![Value::Tensor(left), Value::Tensor(right)]).expect("horzcat");
        match result.value() {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![2, 3]);
                assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0, 10.0, 20.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn horzcat_preserves_exact_integer_class() {
        let left =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("left");
        let result = horzcat_builtin(vec![Value::Tensor(left), Value::Num(3.5)]).expect("horzcat");

        let Value::Tensor(output) = result.value() else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 4]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_arrays_concatenate_by_columns() {
        let lhs = CharArray::from_unicode_scalars("Run".chars().collect(), 1, 3).unwrap();
        let rhs = CharArray::from_unicode_scalars("Mat".chars().collect(), 1, 3).unwrap();
        let result =
            horzcat_builtin(vec![Value::CharArray(lhs), Value::CharArray(rhs)]).expect("horzcat");
        match result.value() {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 6);
                let text = runmat_value::decode_utf16_code_units(arr.data())
                    .expect("valid UTF-16 fixture text");
                assert_eq!(text, "RunMat");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_arrays_concatenate() {
        let left = StringArray::new(vec!["left".into(), "right".into()], vec![1, 2]).unwrap();
        let right = StringArray::new(vec!["top".into(), "bottom".into()], vec![1, 2]).unwrap();
        let result = horzcat_builtin(vec![Value::StringArray(left), Value::StringArray(right)])
            .expect("horzcat");
        match result.value() {
            Value::StringArray(arr) => {
                assert_eq!(arr.shape, vec![1, 4]);
                assert_eq!(arr.data(), vec!["left", "right", "top", "bottom"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_and_char_arrays_concatenate() {
        let left = StringArray::new(vec!["wn = ".into()], vec![1, 1]).unwrap();
        let mid = CharArray::new_row("6");
        let right = StringArray::new(vec![" rad/s".into()], vec![1, 1]).unwrap();
        let result = horzcat_builtin(vec![
            Value::StringArray(left),
            Value::CharArray(mid),
            Value::StringArray(right),
        ])
        .expect("horzcat mixed");
        match result.value() {
            Value::StringArray(arr) => {
                assert_eq!(arr.shape, vec![1, 3]);
                assert_eq!(arr.data(), vec!["wn = ", "6", " rad/s"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mismatched_rows_error_mentions_horzcat() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3, 1]).unwrap();
        let err = horzcat_builtin(vec![Value::Tensor(a), Value::Tensor(b)]).unwrap_err();
        assert!(err.starts_with("horzcat:"));
        assert!(err.contains("dimension 1 mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn horzcat_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let left = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let right = Tensor::new(vec![10.0, 30.0], vec![2, 1]).unwrap();
            let view_left = runmat_accelerate_api::HostTensorView {
                data: &left.materialize_f64(),
                shape: left.shape(),
            };
            let view_right = runmat_accelerate_api::HostTensorView {
                data: &right.materialize_f64(),
                shape: right.shape(),
            };
            let h_left = provider.upload(&view_left).expect("upload left");
            let h_right = provider.upload(&view_right).expect("upload right");
            let result = horzcat_builtin(vec![Value::GpuTensor(h_left), Value::GpuTensor(h_right)])
                .expect("horzcat");
            let gathered = result.with_owned(test_support::gather).expect("gather");
            assert_eq!(gathered.shape(), vec![2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 10.0, 30.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_arrays_concatenate() {
        let top = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let bottom = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        let result = horzcat_builtin(vec![Value::LogicalArray(top), Value::LogicalArray(bottom)])
            .expect("horzcat logical");
        match result.value() {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 6]);
                assert_eq!(array.data, vec![1, 0, 1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_arrays_concatenate() {
        let left = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
        let right = ComplexTensor::new(vec![(5.0, 6.0), (7.0, 8.0)], vec![1, 2]).unwrap();
        let result = horzcat_builtin(vec![
            Value::ComplexTensor(left),
            Value::ComplexTensor(right),
        ])
        .expect("horzcat complex");
        match result.value() {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape(), vec![1, 4]);
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
    fn cell_arrays_concatenate_columns() {
        let lhs = CellArray::new(
            vec![
                Value::Num(1.0),
                Value::from("a"),
                Value::Num(2.0),
                Value::from("b"),
            ],
            2,
            2,
        )
        .unwrap();
        let rhs = CellArray::new(
            vec![
                Value::Num(3.0),
                Value::from("c"),
                Value::Num(4.0),
                Value::from("d"),
            ],
            2,
            2,
        )
        .unwrap();
        let result =
            horzcat_builtin(vec![Value::Cell(lhs), Value::Cell(rhs)]).expect("horzcat cell");
        match result.value() {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert_eq!(cell.cols, 4);
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn horzcat_like_gpu_from_host_inputs() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let prototype = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_view = runmat_accelerate_api::HostTensorView {
                data: &prototype.materialize_f64(),
                shape: prototype.shape(),
            };
            let proto_handle = provider.upload(&proto_view).expect("upload proto");

            let left = Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap();
            let right = Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap();
            let result = horzcat_builtin(vec![
                Value::Tensor(left),
                Value::Tensor(right),
                Value::from("like"),
                Value::GpuTensor(proto_handle),
            ])
            .expect("horzcat like");
            match result.value() {
                Value::GpuTensor(_) => {}
                other => panic!("expected gpu tensor, got {other:?}"),
            }
            let gathered = result.with_owned(test_support::gather).expect("gather");
            assert_eq!(gathered.shape(), vec![2, 2]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn horzcat_wgpu_matches_cpu() {
        let provider_init = std::panic::catch_unwind(|| {
            runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            )
        });
        if !matches!(provider_init, Ok(Ok(_))) {
            return;
        }
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();

        let cpu_value = horzcat_builtin(vec![Value::Tensor(a.clone()), Value::Tensor(b.clone())])
            .expect("cpu horzcat");
        let expected = match cpu_value.value() {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };

        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let view_a = runmat_accelerate_api::HostTensorView {
            data: &a.materialize_f64(),
            shape: a.shape(),
        };
        let view_b = runmat_accelerate_api::HostTensorView {
            data: &b.materialize_f64(),
            shape: b.shape(),
        };
        let ha = provider.upload(&view_a).expect("upload a");
        let hb = provider.upload(&view_b).expect("upload b");
        let gpu_value =
            horzcat_builtin(vec![Value::GpuTensor(ha), Value::GpuTensor(hb)]).expect("gpu horzcat");
        let gathered = gpu_value.with_owned(test_support::gather).expect("gather");
        assert_eq!(gathered.shape(), expected.shape());
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    }
}
