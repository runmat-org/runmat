//! MATLAB-compatible `strings` builtin that preallocates string arrays filled with empty scalars.

use runmat_builtins::{LogicalArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::random_args::{keyword_of, shape_from_value};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::type_resolvers::string_array_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const FN_NAME: &str = "strings";
const SIZE_INTEGER_ERR: &str = "size inputs must be integers";
const SIZE_NONNEGATIVE_ERR: &str = "size inputs must be nonnegative integers";
const SIZE_FINITE_ERR: &str = "size inputs must be finite";
const SIZE_NUMERIC_ERR: &str = "size arguments must be numeric scalars or vectors";
const SIZE_SCALAR_ERR: &str = "size inputs must be scalar";

fn strings_flow(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(FN_NAME).build()
}

fn remap_strings_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, FN_NAME)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::strings")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: FN_NAME,
    op_kind: GpuOpKind::Custom("array_creation"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs entirely on the host; size arguments pulled from the GPU are gathered before allocation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::strings")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: FN_NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Preallocates host string arrays; no fusion-supported kernels are generated.",
};

struct ParsedStrings {
    shape: Vec<usize>,
    fill: FillKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FillKind {
    Empty,
    Missing,
}

#[runtime_builtin(
    name = "strings",
    category = "strings/core",
    summary = "Preallocate string arrays filled with empty string scalars.",
    keywords = "strings,string array,empty,preallocate",
    accel = "array_construct",
    type_resolver(string_array_type),
    builtin_path = "crate::builtins::strings::core::strings"
)]
async fn strings_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let ParsedStrings { shape, fill } = parse_arguments(rest).await?;
    let total = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            strings_flow(format!("{FN_NAME}: requested size exceeds platform limits"))
        })
    })?;

    let fill_text = match fill {
        FillKind::Empty => String::new(),
        FillKind::Missing => "<missing>".to_string(),
    };

    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(fill_text.clone());
    }

    let array =
        StringArray::new(data, shape).map_err(|e| strings_flow(format!("{FN_NAME}: {e}")))?;
    Ok(Value::StringArray(array))
}

async fn parse_arguments(args: Vec<Value>) -> BuiltinResult<ParsedStrings> {
    let mut size_values: Vec<Value> = Vec::new();
    let mut like_proto: Option<Value> = None;
    let mut fill = FillKind::Empty;

    let mut idx = 0;
    while idx < args.len() {
        let host = gather_if_needed_async(&args[idx])
            .await
            .map_err(remap_strings_flow)?;
        if let Some(keyword) = keyword_of(&host) {
            match keyword.as_str() {
                "like" => {
                    if like_proto.is_some() {
                        return Err(strings_flow(format!(
                            "{FN_NAME}: multiple 'like' specifications are not supported"
                        )));
                    }
                    let Some(proto_raw) = args.get(idx + 1) else {
                        return Err(strings_flow(format!(
                            "{FN_NAME}: expected prototype after 'like'"
                        )));
                    };
                    let proto = gather_if_needed_async(proto_raw)
                        .await
                        .map_err(remap_strings_flow)?;
                    like_proto = Some(proto);
                    idx += 2;
                    continue;
                }
                "missing" => {
                    fill = FillKind::Missing;
                    idx += 1;
                    continue;
                }
                "empty" => {
                    fill = FillKind::Empty;
                    idx += 1;
                    continue;
                }
                _ => {}
            }
        }
        size_values.push(host);
        idx += 1;
    }

    let dims = parse_size_values(size_values)?;
    let mut shape = if let Some(dims) = dims {
        normalize_dims(dims)
    } else if let Some(proto) = like_proto.as_ref() {
        prototype_shape(proto)?
    } else {
        vec![1, 1]
    };

    if shape.is_empty() {
        shape = vec![0, 0];
    }

    Ok(ParsedStrings { shape, fill })
}

fn prototype_shape(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::StringArray(sa) => Ok(sa.shape.clone()),
        _ => shape_from_value(value, FN_NAME).map_err(strings_flow),
    }
}

fn err_integer() -> RuntimeError {
    strings_flow(format!("{FN_NAME}: {SIZE_INTEGER_ERR}"))
}

fn err_nonnegative() -> RuntimeError {
    strings_flow(format!("{FN_NAME}: {SIZE_NONNEGATIVE_ERR}"))
}

fn err_finite() -> RuntimeError {
    strings_flow(format!("{FN_NAME}: {SIZE_FINITE_ERR}"))
}

fn parse_size_values(values: Vec<Value>) -> BuiltinResult<Option<Vec<usize>>> {
    match values.len() {
        0 => Ok(None),
        1 => parse_single_argument(values.into_iter().next().unwrap()).map(Some),
        _ => {
            let mut dims = Vec::with_capacity(values.len());
            for value in &values {
                dims.push(parse_size_scalar(value)?);
            }
            Ok(Some(dims))
        }
    }
}

fn parse_single_argument(value: Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(iv) => Ok(vec![validate_i64_dimension(iv.to_i64())?]),
        Value::Num(n) => Ok(vec![parse_numeric_dimension(n)?]),
        Value::Bool(b) => Ok(vec![if b { 1 } else { 0 }]),
        Value::Tensor(t) => parse_size_tensor(&t),
        Value::LogicalArray(arr) => parse_size_logical_array(&arr),
        other => Err(strings_flow(format!(
            "{FN_NAME}: {SIZE_NUMERIC_ERR}, got {other:?}"
        ))),
    }
}

fn parse_size_scalar(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Int(iv) => {
            let raw = iv.to_i64();
            validate_i64_dimension(raw)
        }
        Value::Num(n) => parse_numeric_dimension(*n),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Tensor(t) => {
            if t.data.len() != 1 {
                return Err(strings_flow(format!("{FN_NAME}: {SIZE_SCALAR_ERR}")));
            }
            parse_numeric_dimension(t.data[0])
        }
        Value::LogicalArray(arr) => {
            if arr.data.len() != 1 {
                return Err(strings_flow(format!("{FN_NAME}: {SIZE_SCALAR_ERR}")));
            }
            Ok(if arr.data[0] != 0 { 1 } else { 0 })
        }
        other => Err(strings_flow(format!(
            "{FN_NAME}: {SIZE_NUMERIC_ERR}, got {other:?}"
        ))),
    }
}

fn parse_size_tensor(tensor: &Tensor) -> BuiltinResult<Vec<usize>> {
    if tensor.data.is_empty() {
        return Ok(vec![0, 0]);
    }
    if !is_vector_shape(&tensor.shape) {
        return Err(strings_flow(format!(
            "{FN_NAME}: size vector must be a row or column vector"
        )));
    }
    tensor
        .data
        .iter()
        .map(|&value| parse_numeric_dimension(value))
        .collect()
}

fn parse_size_logical_array(array: &LogicalArray) -> BuiltinResult<Vec<usize>> {
    if array.data.is_empty() {
        return Ok(vec![0, 0]);
    }
    if !is_vector_shape(&array.shape) {
        return Err(strings_flow(format!(
            "{FN_NAME}: size vector must be a row or column vector"
        )));
    }
    array
        .data
        .iter()
        .map(|&value| Ok(if value != 0 { 1 } else { 0 }))
        .collect()
}

fn parse_numeric_dimension(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(err_finite());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(err_integer());
    }
    if rounded < 0.0 {
        return Err(err_nonnegative());
    }
    if rounded > usize::MAX as f64 {
        return Err(strings_flow(format!(
            "{FN_NAME}: requested dimension exceeds platform limits"
        )));
    }
    Ok(rounded as usize)
}

fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    match dims.len() {
        0 => vec![0, 0],
        1 => {
            let side = dims[0];
            vec![side, side]
        }
        _ => dims,
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 | 1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => shape.iter().filter(|&&d| d > 1).count() <= 1,
    }
}

fn validate_i64_dimension(raw: i64) -> BuiltinResult<usize> {
    if raw < 0 {
        return Err(err_nonnegative());
    }
    if (raw as u128) > (usize::MAX as u128) {
        return Err(strings_flow(format!(
            "{FN_NAME}: requested dimension exceeds platform limits"
        )));
    }
    Ok(raw as usize)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::Type;

    fn strings_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::strings_builtin(rest))
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_default_scalar() {
        let result = strings_builtin(Vec::new()).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 1]);
                assert_eq!(array.data, vec![String::new()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_square_from_single_dimension() {
        let args = vec![Value::Num(4.0)];
        let result = strings_builtin(args).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![4, 4]);
                assert!(array.data.iter().all(|s| s.is_empty()));
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_rectangular_multiple_args() {
        let args = vec![
            Value::Int(runmat_builtins::IntValue::I32(2)),
            Value::Num(3.0),
        ];
        let result = strings_builtin(args).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data.len(), 6);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_from_size_vector_tensor() {
        let dims = Tensor::new(vec![2.0, 3.0, 1.0], vec![1, 3]).unwrap();
        let result = strings_builtin(vec![Value::Tensor(dims)]).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 3, 1]);
                assert_eq!(array.data.len(), 6);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_preserves_trailing_singletons() {
        let args = vec![
            Value::Num(3.0),
            Value::Int(runmat_builtins::IntValue::I32(1)),
            Value::Num(1.0),
            Value::Bool(true),
        ];
        let result = strings_builtin(args).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![3, 1, 1, 1]);
                assert_eq!(array.data.len(), 3);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_bool_dimensions() {
        let result = strings_builtin(vec![Value::Bool(true), Value::Bool(false)]).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 0]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_logical_vector_argument() {
        let logical =
            LogicalArray::new(vec![1u8, 0, 1], vec![1, 3]).expect("logical size construction");
        let result = strings_builtin(vec![Value::LogicalArray(logical)]).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 0, 1]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_negative_dimension_errors() {
        let err =
            error_message(strings_builtin(vec![Value::Num(-5.0)]).expect_err("expected error"));
        assert!(err.contains(super::SIZE_NONNEGATIVE_ERR));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_rejects_non_integer_dimension() {
        let err =
            error_message(strings_builtin(vec![Value::Num(2.5)]).expect_err("expected error"));
        assert!(err.contains(super::SIZE_INTEGER_ERR));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_rejects_non_numeric_dimension() {
        let err = error_message(
            strings_builtin(vec![Value::String("size".into())]).expect_err("expected error"),
        );
        assert!(err.contains("size arguments must be numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_empty_vector_returns_empty_array() {
        let dims = Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let result = strings_builtin(vec![Value::Tensor(dims)]).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![0, 0]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_missing_option_fills_with_missing() {
        let result = strings_builtin(vec![
            Value::Num(2.0),
            Value::Num(3.0),
            Value::String("missing".into()),
        ])
        .expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data.len(), 6);
                assert!(array.data.iter().all(|s| s == "<missing>"));
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_missing_without_dims_defaults_to_scalar() {
        let result = strings_builtin(vec![Value::String("missing".into())]).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 1]);
                assert_eq!(array.data, vec!["<missing>".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_like_prototype_shape() {
        let proto = StringArray::new(
            vec!["alpha".into(), "beta".into(), "gamma".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = strings_builtin(vec![
            Value::String("like".into()),
            Value::StringArray(proto.clone()),
        ])
        .expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, proto.shape);
                assert!(array.data.iter().all(|s| s.is_empty()));
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_like_numeric_prototype() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = strings_builtin(vec![
            Value::String("like".into()),
            Value::Tensor(tensor.clone()),
        ])
        .expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, tensor.shape);
                assert_eq!(array.data.len(), tensor.data.len());
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_like_overrides_shape_when_dims_provided() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let result = strings_builtin(vec![
            Value::String("like".into()),
            Value::Tensor(tensor),
            Value::Int(runmat_builtins::IntValue::I32(3)),
        ])
        .expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![3, 3]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_like_requires_prototype() {
        let err = error_message(
            strings_builtin(vec![Value::String("like".into())]).expect_err("expected error"),
        );
        assert!(err.contains("expected prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_like_rejects_multiple_specs() {
        let err = error_message(
            strings_builtin(vec![
                Value::String("like".into()),
                Value::Num(1.0),
                Value::String("like".into()),
                Value::Num(2.0),
            ])
            .expect_err("expected error"),
        );
        assert!(err.contains("multiple 'like'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_gpu_size_vector_argument() {
        test_support::with_test_provider(|provider| {
            let dims = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
            let view = HostTensorView {
                data: &dims.data,
                shape: &dims.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = strings_builtin(vec![Value::GpuTensor(handle)]).expect("strings");
            match result {
                Value::StringArray(array) => {
                    assert_eq!(array.shape, vec![2, 3]);
                    assert_eq!(array.data.len(), 6);
                }
                other => panic!("expected string array, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_like_accepts_gpu_prototype() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                strings_builtin(vec![Value::String("like".into()), Value::GpuTensor(handle)])
                    .expect("strings");
            match result {
                Value::StringArray(array) => {
                    assert_eq!(array.shape, vec![2, 2]);
                }
                other => panic!("expected string array, got {other:?}"),
            }
        });
    }

    #[test]
    fn strings_type_is_string_array() {
        assert_eq!(string_array_type(&[Type::Num]), Type::cell_of(Type::String));
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_handles_wgpu_size_vectors() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let dims = Tensor::new(vec![1.0, 4.0], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: &dims.data,
            shape: &dims.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload");
        let result = strings_builtin(vec![Value::GpuTensor(handle)]).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![1, 4]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }
}
