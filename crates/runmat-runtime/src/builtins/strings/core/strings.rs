//! MATLAB-compatible `strings` builtin that preallocates string arrays filled with empty scalars.

use runmat_builtins::{
    BuiltinErrorDescriptor, STRINGS_ERROR_INTERNAL, STRINGS_ERROR_INVALID_SIZE,
    STRINGS_ERROR_LIKE_DUPLICATE, STRINGS_ERROR_LIKE_MISSING_PROTOTYPE,
    STRINGS_ERROR_SIZE_OVERFLOW, STRINGS_FILL_EXTENSION, STRINGS_LIKE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, LogicalArray, NumericScalar, StringArray, Tensor, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::random_args::{keyword_of, shape_from_value};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const FN_NAME: &str = "strings";
const SIZE_INTEGER_ERR: &str = "size inputs must be integers";
const SIZE_FINITE_ERR: &str = "size inputs must be finite";
const SIZE_SCALAR_ERR: &str = "size inputs must be scalar";

fn strings_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    strings_error_with_message(error.message, error)
}

fn strings_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(FN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
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
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::strings"
)]
async fn strings_builtin(rest: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    preflight_strings_extensions(rest.view())?;
    let ParsedStrings { shape, fill } = parse_arguments(rest).await?;
    let total = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| strings_error(&STRINGS_ERROR_SIZE_OVERFLOW))
    })?;

    let fill_text = match fill {
        FillKind::Empty => runmat_value::StringElement::text(""),
        FillKind::Missing => runmat_value::StringElement::Missing,
    };

    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        data.push(fill_text.clone());
    }

    let array = StringArray::from_elements_unbounded(data, shape)
        .map_err(|_| strings_error(&STRINGS_ERROR_INTERNAL))?;
    Ok(Value::StringArray(array))
}

fn preflight_strings_extensions(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    for value in args {
        let Some(keyword) = keyword_of(value).map_err(remap_strings_flow)? else {
            continue;
        };
        match keyword.as_str() {
            "like" => crate::compatibility::ensure_builtin_extension_enabled(
                &STRINGS_LIKE_EXTENSION,
                FN_NAME,
            )?,
            "missing" | "empty" => crate::compatibility::ensure_builtin_extension_enabled(
                &STRINGS_FILL_EXTENSION,
                FN_NAME,
            )?,
            _ => {}
        }
    }
    Ok(())
}

async fn parse_arguments(args: crate::call::BuiltinArguments) -> BuiltinResult<ParsedStrings> {
    let mut size_values: Vec<Value> = Vec::new();
    let mut like_proto: Option<Value> = None;
    let mut fill = FillKind::Empty;

    let mut idx = 0;
    while idx < args.len() {
        let host = gather_if_needed_async(&args[idx])
            .await
            .map_err(remap_strings_flow)?;
        if let Some(keyword) = keyword_of(&host).map_err(|mut error| {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            remap_strings_flow(error)
        })? {
            match keyword.as_str() {
                "like" => {
                    if like_proto.is_some() {
                        return Err(strings_error(&STRINGS_ERROR_LIKE_DUPLICATE));
                    }
                    let Some(proto_raw) = args.get(idx + 1) else {
                        return Err(strings_error(&STRINGS_ERROR_LIKE_MISSING_PROTOTYPE));
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
        Value::StringArray(sa) => Ok(sa.shape().to_vec()),
        _ => {
            shape_from_value(value, FN_NAME).map_err(|_| strings_error(&STRINGS_ERROR_INVALID_SIZE))
        }
    }
}

fn err_integer() -> RuntimeError {
    strings_error_with_message(
        format!("{FN_NAME}: {SIZE_INTEGER_ERR}"),
        &STRINGS_ERROR_INVALID_SIZE,
    )
}

fn err_finite() -> RuntimeError {
    strings_error_with_message(
        format!("{FN_NAME}: {SIZE_FINITE_ERR}"),
        &STRINGS_ERROR_INVALID_SIZE,
    )
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
        Value::Int(iv) => parse_integer_dimension(&iv).map(|dimension| vec![dimension]),
        Value::Num(n) => Ok(vec![parse_numeric_dimension(n)?]),
        Value::Bool(b) => Ok(vec![if b { 1 } else { 0 }]),
        Value::Tensor(t) => parse_size_tensor(&t),
        Value::LogicalArray(arr) => parse_size_logical_array(&arr),
        _ => Err(strings_error(&STRINGS_ERROR_INVALID_SIZE)),
    }
}

fn parse_size_scalar(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Int(iv) => parse_integer_dimension(iv),
        Value::Num(n) => parse_numeric_dimension(*n),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Tensor(t) => {
            if !tensor::is_scalar_tensor(t) {
                return Err(strings_error_with_message(
                    format!("{FN_NAME}: {SIZE_SCALAR_ERR}"),
                    &STRINGS_ERROR_INVALID_SIZE,
                ));
            }
            match t.integer_storage().and_then(|storage| storage.value_at(0)) {
                Some(value) => parse_integer_dimension(&value),
                None => parse_numeric_dimension(tensor::tensor_value_f64(t, 0)),
            }
        }
        Value::LogicalArray(arr) => {
            if arr.data.len() != 1 {
                return Err(strings_error_with_message(
                    format!("{FN_NAME}: {SIZE_SCALAR_ERR}"),
                    &STRINGS_ERROR_INVALID_SIZE,
                ));
            }
            Ok(if arr.data[0] != 0 { 1 } else { 0 })
        }
        _ => Err(strings_error(&STRINGS_ERROR_INVALID_SIZE)),
    }
}

fn parse_size_tensor(tensor: &Tensor) -> BuiltinResult<Vec<usize>> {
    let len = tensor.len();
    if len == 0 {
        return Ok(vec![0, 0]);
    }
    if !is_vector_shape(tensor.shape()) {
        return Err(strings_error_with_message(
            format!("{FN_NAME}: size vector must be a row or column vector"),
            &STRINGS_ERROR_INVALID_SIZE,
        ));
    }
    (0..len)
        .map(|index| {
            let value = tensor
                .numeric_value_at(index)
                .expect("tensor storage length matches shape");
            match value {
                NumericScalar::F64(value) => parse_numeric_dimension(value),
                NumericScalar::F32(value) => parse_numeric_dimension(f64::from(value)),
                value => parse_integer_dimension(
                    &value
                        .into_int_value()
                        .expect("non-floating numeric scalar is integer"),
                ),
            }
        })
        .collect()
}

fn parse_size_logical_array(array: &LogicalArray) -> BuiltinResult<Vec<usize>> {
    if array.data.is_empty() {
        return Ok(vec![0, 0]);
    }
    if !is_vector_shape(&array.shape) {
        return Err(strings_error_with_message(
            format!("{FN_NAME}: size vector must be a row or column vector"),
            &STRINGS_ERROR_INVALID_SIZE,
        ));
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
        return Ok(0);
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(strings_error_with_message(
            format!("{FN_NAME}: requested dimension exceeds platform limits"),
            &STRINGS_ERROR_SIZE_OVERFLOW,
        ));
    }
    Ok(rounded as usize)
}

fn parse_integer_dimension(value: &IntValue) -> BuiltinResult<usize> {
    if value.try_to_i64().is_some_and(|value| value < 0) {
        return Ok(0);
    }
    value
        .try_to_usize()
        .ok_or_else(|| strings_error(&STRINGS_ERROR_INVALID_SIZE))
}

fn normalize_dims(mut dims: Vec<usize>) -> Vec<usize> {
    match dims.len() {
        0 => vec![0, 0],
        1 => {
            let side = dims[0];
            vec![side, side]
        }
        _ => {
            while dims.len() > 2 && dims.last() == Some(&1) {
                dims.pop();
            }
            dims
        }
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 | 1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => shape.iter().filter(|&&d| d > 1).count() <= 1,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntegerStorage, NumericStorage};

    fn strings_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        futures::executor::block_on(super::strings_builtin(
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn strings_low_rank_size_tensors_keep_scalar_square_and_vector_dimensions() {
        let square = strings_builtin(vec![Value::Tensor(
            Tensor::new(vec![2.0], Vec::new()).unwrap(),
        )])
        .expect("rank-zero size");
        let Value::StringArray(square) = square else {
            panic!("string array expected")
        };
        assert_eq!(square.shape, vec![2, 2]);
        assert_eq!(square.data().len(), 4);

        let rectangle = strings_builtin(vec![Value::Tensor(
            Tensor::new(vec![2.0, 3.0], vec![2]).unwrap(),
        )])
        .expect("rank-one size vector");
        let Value::StringArray(rectangle) = rectangle else {
            panic!("string array expected")
        };
        assert_eq!(rectangle.shape, vec![2, 3]);
        assert_eq!(rectangle.data().len(), 6);
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
                assert_eq!(array.data(), vec![String::new()]);
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
                assert!(array.data().iter().all(|s| s.as_str() == Some("")));
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_rectangular_multiple_args() {
        let args = vec![Value::Int(runmat_value::IntValue::I32(2)), Value::Num(3.0)];
        let result = strings_builtin(args).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data().len(), 6);
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
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data().len(), 6);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn strings_native_single_size_vector_reads_authoritative_storage() {
        let dims = Tensor::from_numeric_storage(NumericStorage::F32(vec![2.0, 3.0]), vec![1, 2])
            .expect("single dimensions");
        let result = strings_builtin(vec![Value::Tensor(dims)]).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data().len(), 6);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn strings_typed_integer_size_vector_preserves_large_uint64_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let dims =
            Tensor::new_integer(IntegerStorage::U64(vec![large, 0]), vec![1, 2]).expect("dims");
        let result = strings_builtin(vec![Value::Tensor(dims)]).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![large as usize, 0]);
                assert!(array.data().is_empty());
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn strings_typed_integer_scalar_tensor_parser_is_exact() {
        let large = 9_007_199_254_740_993_u64;
        let scalar =
            Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).expect("scalar");
        assert_eq!(
            parse_size_scalar(&Value::Tensor(scalar)).expect("parse scalar"),
            large as usize
        );

        let vector =
            Tensor::new_integer(IntegerStorage::U64(vec![large, 1]), vec![1, 2]).expect("vector");
        assert_eq!(
            parse_size_tensor(&vector).expect("parse vector"),
            vec![large as usize, 1]
        );
    }

    #[test]
    fn strings_typed_integer_tensor_dimensions_treat_negative_values_as_zero() {
        let scalar =
            Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("negative");
        assert_eq!(parse_size_scalar(&Value::Tensor(scalar)).unwrap(), 0);

        let vector =
            Tensor::new_integer(IntegerStorage::I16(vec![2, -1]), vec![1, 2]).expect("vector");
        assert_eq!(parse_size_tensor(&vector).unwrap(), vec![2, 0]);
    }

    #[test]
    fn strings_numeric_dimensions_reject_unrepresentable_double_boundary() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };

        assert!(parse_numeric_dimension(boundary).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_ignores_trailing_singletons_beyond_second_dimension() {
        let args = vec![
            Value::Num(3.0),
            Value::Int(runmat_value::IntValue::I32(1)),
            Value::Num(1.0),
            Value::Bool(true),
        ];
        let result = strings_builtin(args).expect("strings");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![3, 1]);
                assert_eq!(array.data().len(), 3);
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
                assert!(array.data().is_empty());
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
                assert_eq!(array.shape, vec![1, 0]);
                assert!(array.data().is_empty());
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_negative_dimension_is_treated_as_zero() {
        let result = strings_builtin(vec![Value::Num(-5.0)]).expect("negative becomes zero");
        match result {
            Value::StringArray(array) => {
                assert_eq!(array.shape, vec![0, 0]);
                assert!(array.data().is_empty());
            }
            other => panic!("expected string array, got {other:?}"),
        }
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
                assert!(array.data().is_empty());
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
                assert_eq!(array.data().len(), 6);
                assert!(array
                    .data()
                    .iter()
                    .all(runmat_value::StringElement::is_missing));
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
                assert_eq!(array.data(), &[runmat_value::StringElement::Missing]);
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
                assert!(array.data().iter().all(|s| s.as_str() == Some("")));
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
                assert_eq!(array.shape, tensor.shape());
                assert_eq!(array.data().len(), tensor.materialize_f64().len());
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
            Value::Int(runmat_value::IntValue::I32(3)),
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
                data: &dims.materialize_f64(),
                shape: dims.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = strings_builtin(vec![Value::GpuTensor(handle)]).expect("strings");
            match result {
                Value::StringArray(array) => {
                    assert_eq!(array.shape, vec![2, 3]);
                    assert_eq!(array.data().len(), 6);
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
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
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

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strings_handles_wgpu_size_vectors() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let dims = Tensor::new(vec![1.0, 4.0], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: &dims.materialize_f64(),
            shape: dims.shape(),
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
