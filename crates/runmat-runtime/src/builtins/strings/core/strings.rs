//! MATLAB-compatible `strings` builtin that preallocates string arrays filled with empty scalars.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    LogicalArray, StringArray, Tensor, Value,
};
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
const SIZE_SCALAR_ERR: &str = "size inputs must be scalar";

const STRINGS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Preallocated string array.",
}];

const STRINGS_INPUT_SZ: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector or scalar side length.",
}];

const STRINGS_INPUT_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First array dimension.",
    },
    BuiltinParamDescriptor {
        name: "n...",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional array dimensions.",
    },
];

const STRINGS_INPUT_LIKE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dims...",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional explicit size dimensions.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Literal option keyword \"like\".",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype value supplying output shape when dims are omitted.",
    },
];

const STRINGS_INPUT_FILL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dims...",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional explicit size dimensions.",
    },
    BuiltinParamDescriptor {
        name: "fill",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"empty\""),
        description: "Fill mode keyword: \"empty\" or \"missing\".",
    },
];

const STRINGS_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "S = strings()",
        inputs: &[],
        outputs: &STRINGS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = strings(sz)",
        inputs: &STRINGS_INPUT_SZ,
        outputs: &STRINGS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = strings(m, n...)",
        inputs: &STRINGS_INPUT_DIMS,
        outputs: &STRINGS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = strings(___, \"like\", p)",
        inputs: &STRINGS_INPUT_LIKE,
        outputs: &STRINGS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = strings(___, fill)",
        inputs: &STRINGS_INPUT_FILL,
        outputs: &STRINGS_OUTPUT,
    },
];

const STRINGS_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRINGS.INVALID_SIZE",
    identifier: Some("RunMat:strings:InvalidSize"),
    when: "Size arguments are not valid numeric scalar/vector dimensions.",
    message: "strings: size arguments must be numeric scalars or vectors",
};

const STRINGS_ERROR_LIKE_MISSING_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRINGS.LIKE_MISSING_PROTOTYPE",
    identifier: Some("RunMat:strings:LikeMissingPrototype"),
    when: "\"like\" is provided without a following prototype.",
    message: "strings: expected prototype after 'like'",
};

const STRINGS_ERROR_LIKE_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRINGS.LIKE_DUPLICATE",
    identifier: Some("RunMat:strings:LikeDuplicate"),
    when: "Multiple \"like\" options are supplied in one call.",
    message: "strings: multiple 'like' specifications are not supported",
};

const STRINGS_ERROR_SIZE_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRINGS.SIZE_OVERFLOW",
    identifier: Some("RunMat:strings:SizeOverflow"),
    when: "Requested dimensions overflow platform limits.",
    message: "strings: requested size exceeds platform limits",
};

const STRINGS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRINGS.INTERNAL",
    identifier: Some("RunMat:strings:InternalError"),
    when: "Internal string array construction failed.",
    message: "strings: internal error",
};

const STRINGS_ERRORS: [BuiltinErrorDescriptor; 5] = [
    STRINGS_ERROR_INVALID_SIZE,
    STRINGS_ERROR_LIKE_MISSING_PROTOTYPE,
    STRINGS_ERROR_LIKE_DUPLICATE,
    STRINGS_ERROR_SIZE_OVERFLOW,
    STRINGS_ERROR_INTERNAL,
];

pub const STRINGS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRINGS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRINGS_ERRORS,
};

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
    category = "strings/core",
    summary = "Preallocate string arrays filled with empty string scalars (`\"\"`).",
    keywords = "strings,string array,empty,preallocate",
    accel = "array_construct",
    type_resolver(string_array_type),
    descriptor(crate::builtins::strings::core::strings::STRINGS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::strings"
)]
async fn strings_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let ParsedStrings { shape, fill } = parse_arguments(rest).await?;
    let total = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| strings_error(&STRINGS_ERROR_SIZE_OVERFLOW))
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
        StringArray::new(data, shape).map_err(|_| strings_error(&STRINGS_ERROR_INTERNAL))?;
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
        Value::StringArray(sa) => Ok(sa.shape.clone()),
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

fn err_nonnegative() -> RuntimeError {
    strings_error_with_message(
        format!("{FN_NAME}: {SIZE_NONNEGATIVE_ERR}"),
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
        Value::Int(iv) => Ok(vec![validate_i64_dimension(iv.to_i64())?]),
        Value::Num(n) => Ok(vec![parse_numeric_dimension(n)?]),
        Value::Bool(b) => Ok(vec![if b { 1 } else { 0 }]),
        Value::Tensor(t) => parse_size_tensor(&t),
        Value::LogicalArray(arr) => parse_size_logical_array(&arr),
        _ => Err(strings_error(&STRINGS_ERROR_INVALID_SIZE)),
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
                return Err(strings_error_with_message(
                    format!("{FN_NAME}: {SIZE_SCALAR_ERR}"),
                    &STRINGS_ERROR_INVALID_SIZE,
                ));
            }
            parse_numeric_dimension(t.data[0])
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
    if tensor.data.is_empty() {
        return Ok(vec![0, 0]);
    }
    if !is_vector_shape(&tensor.shape) {
        return Err(strings_error_with_message(
            format!("{FN_NAME}: size vector must be a row or column vector"),
            &STRINGS_ERROR_INVALID_SIZE,
        ));
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
        return Err(err_nonnegative());
    }
    if rounded > usize::MAX as f64 {
        return Err(strings_error_with_message(
            format!("{FN_NAME}: requested dimension exceeds platform limits"),
            &STRINGS_ERROR_SIZE_OVERFLOW,
        ));
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
        return Err(strings_error_with_message(
            format!("{FN_NAME}: requested dimension exceeds platform limits"),
            &STRINGS_ERROR_SIZE_OVERFLOW,
        ));
    }
    Ok(raw as usize)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{ResolveContext, Type};

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
        assert_eq!(
            string_array_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::cell_of(Type::String)
        );
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
