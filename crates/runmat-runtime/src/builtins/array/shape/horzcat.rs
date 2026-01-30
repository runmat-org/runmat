//! MATLAB-compatible `horzcat` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{IntValue, Tensor, Type, Value};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, RuntimeError};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::type_shapes::scalar_tensor_shape;

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

fn concat_input_shape(ty: &Type) -> Option<Vec<Option<usize>>> {
    match ty {
        Type::Tensor { shape: Some(shape) } => Some(shape.clone()),
        Type::Logical { shape: Some(shape) } => Some(shape.clone()),
        Type::Num | Type::Int | Type::Bool => Some(scalar_tensor_shape()),
        _ => None,
    }
}

fn concat_shape(shapes: &[Vec<Option<usize>>], dim_1based: usize) -> Option<Vec<Option<usize>>> {
    if shapes.is_empty() || dim_1based == 0 {
        return None;
    }
    let rank = shapes
        .iter()
        .map(|shape| shape.len())
        .max()?
        .max(dim_1based);
    let mut padded = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let mut current = shape.clone();
        while current.len() < rank {
            current.push(Some(1));
        }
        padded.push(current);
    }

    let mut output = vec![None; rank];
    let dim_zero = dim_1based - 1;
    for axis in 0..rank {
        if axis == dim_zero {
            let mut total: Option<usize> = Some(0);
            for shape in &padded {
                match (total, shape[axis]) {
                    (Some(acc), Some(value)) => total = acc.checked_add(value),
                    _ => {
                        total = None;
                        break;
                    }
                }
            }
            output[axis] = total;
        } else {
            let mut shared: Option<usize> = None;
            let mut mismatch = false;
            for shape in &padded {
                match (shared, shape[axis]) {
                    (None, value) => shared = value,
                    (Some(current), Some(value)) if current == value => {}
                    (Some(_), Some(_)) => {
                        mismatch = true;
                        break;
                    }
                    _ => {
                        shared = None;
                        break;
                    }
                }
            }
            output[axis] = if mismatch { None } else { shared };
        }
    }

    let min_len = dim_1based.max(2).min(output.len());
    while output.len() > min_len && matches!(output.last(), Some(Some(1))) {
        output.pop();
    }
    Some(output)
}

fn cell_element_type(inputs: &[Type]) -> Option<Box<Type>> {
    let mut element: Option<Type> = None;
    for ty in inputs {
        let Type::Cell { element_type, .. } = ty else {
            return None;
        };
        match (&element, element_type.as_deref()) {
            (None, Some(current)) => element = Some(current.clone()),
            (Some(existing), Some(current)) if existing == current => {}
            (Some(_), Some(_)) => return None,
            _ => {}
        }
    }
    element.map(Box::new)
}

fn concat_type_with_dim(args: &[Type], dim_1based: usize) -> Type {
    if args.is_empty() {
        return Type::tensor();
    }
    if args.len() == 1 {
        return args[0].clone();
    }

    let all_cells = args.iter().all(|arg| matches!(arg, Type::Cell { .. }));
    if all_cells {
        return Type::Cell {
            element_type: cell_element_type(args),
            length: None,
        };
    }

    let all_strings = args.iter().all(|arg| matches!(arg, Type::String));
    if all_strings {
        return Type::cell_of(Type::String);
    }

    let has_numeric =
        args.iter().any(|arg| matches!(arg, Type::Tensor { .. } | Type::Num | Type::Int));
    let has_logical = args
        .iter()
        .any(|arg| matches!(arg, Type::Logical { .. } | Type::Bool));

    if has_numeric {
        let shapes: Option<Vec<Vec<Option<usize>>>> = args.iter().map(concat_input_shape).collect();
        return Type::Tensor {
            shape: shapes.as_ref().and_then(|s| concat_shape(s, dim_1based)),
        };
    }

    if has_logical {
        let shapes: Option<Vec<Vec<Option<usize>>>> = args.iter().map(concat_input_shape).collect();
        return Type::Logical {
            shape: shapes.as_ref().and_then(|s| concat_shape(s, dim_1based)),
        };
    }

    Type::Unknown
}

fn horzcat_type(args: &[Type]) -> Type {
    concat_type_with_dim(args, 2)
}

#[runtime_builtin(
    name = "horzcat",
    category = "array/shape",
    summary = "Concatenate inputs horizontally (dimension 2) just like MATLAB square brackets.",
    keywords = "horzcat,horizontal concatenation,array,gpu",
    accel = "array_construct",
    type_resolver(horzcat_type),
    builtin_path = "crate::builtins::array::shape::horzcat"
)]
async fn horzcat_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return empty_double();
    }
    if args.len() == 1 {
        return Ok(args.into_iter().next().unwrap());
    }

    let mut forwarded = Vec::with_capacity(args.len() + 1);
    forwarded.push(Value::Int(IntValue::I32(2)));
    forwarded.extend(args);
    match crate::call_builtin_async("cat", &forwarded).await {
        Ok(value) => Ok(value),
        Err(err) => Err(adapt_cat_error(err)),
    }
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;

    fn horzcat_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::horzcat_builtin(args))
    }
    use crate::builtins::common::test_support;
    use runmat_builtins::{CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_invocation_returns_zero_by_zero() {
        let result = horzcat_builtin(Vec::new()).expect("horzcat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.data.is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn single_argument_round_trips() {
        let result = horzcat_builtin(vec![Value::Num(3.5)]).expect("horzcat");
        assert_eq!(result, Value::Num(3.5));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_horizontal_concat() {
        let left = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let right = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
        let result =
            horzcat_builtin(vec![Value::Tensor(left), Value::Tensor(right)]).expect("horzcat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                assert_eq!(t.data, vec![1.0, 3.0, 2.0, 4.0, 10.0, 20.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_arrays_concatenate_by_columns() {
        let lhs = CharArray::new("Run".chars().collect(), 1, 3).unwrap();
        let rhs = CharArray::new("Mat".chars().collect(), 1, 3).unwrap();
        let result =
            horzcat_builtin(vec![Value::CharArray(lhs), Value::CharArray(rhs)]).expect("horzcat");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 6);
                let text: String = arr.data.into_iter().collect();
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
        match result {
            Value::StringArray(arr) => {
                assert_eq!(arr.shape, vec![1, 4]);
                assert_eq!(arr.data, vec!["left", "right", "top", "bottom"]);
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
                data: &left.data,
                shape: &left.shape,
            };
            let view_right = runmat_accelerate_api::HostTensorView {
                data: &right.data,
                shape: &right.shape,
            };
            let h_left = provider.upload(&view_left).expect("upload left");
            let h_right = provider.upload(&view_right).expect("upload right");
            let result = horzcat_builtin(vec![Value::GpuTensor(h_left), Value::GpuTensor(h_right)])
                .expect("horzcat");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 3.0, 10.0, 30.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_arrays_concatenate() {
        let top = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let bottom = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        let result = horzcat_builtin(vec![Value::LogicalArray(top), Value::LogicalArray(bottom)])
            .expect("horzcat logical");
        match result {
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
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 4]);
                assert_eq!(
                    ct.data,
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
        match result {
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
        test_support::with_test_provider(|provider| {
            let prototype = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let proto_view = runmat_accelerate_api::HostTensorView {
                data: &prototype.data,
                shape: &prototype.shape,
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
            let handle = match result {
                Value::GpuTensor(h) => h,
                other => panic!("expected gpu tensor, got {other:?}"),
            };
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 3.0, 5.0, 7.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn horzcat_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();

        let cpu_value = horzcat_builtin(vec![Value::Tensor(a.clone()), Value::Tensor(b.clone())])
            .expect("cpu horzcat");
        let expected = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor output, got {other:?}"),
        };

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view_a = runmat_accelerate_api::HostTensorView {
            data: &a.data,
            shape: &a.shape,
        };
        let view_b = runmat_accelerate_api::HostTensorView {
            data: &b.data,
            shape: &b.shape,
        };
        let ha = provider.upload(&view_a).expect("upload a");
        let hb = provider.upload(&view_b).expect("upload b");
        let gpu_value =
            horzcat_builtin(vec![Value::GpuTensor(ha), Value::GpuTensor(hb)]).expect("gpu horzcat");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.data, expected.data);
    }
}
