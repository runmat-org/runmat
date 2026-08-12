//! MATLAB-compatible `reshape` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::shape::value_numel;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
use runmat_builtins::shape_rules::{element_count_if_known, unknown_shape};
use runmat_builtins::ResolveContext;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::shape::reshape")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "reshape",
    op_kind: GpuOpKind::Custom("reshape"),
    supported_precisions: &[
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::I32,
        ScalarType::Bool,
    ],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("reshape")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Providers update residency metadata via custom reshape hook; no kernel launches required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::shape::reshape")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "reshape",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Reshape influences fusion layout but emits no kernels; fusion planner treats it as a metadata op.",
};

fn reshape_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("reshape").build()
}

const RESHAPE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array data with reshaped dimensions.",
}];

const RESHAPE_INPUTS_SZ: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array/value.",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row/column size vector. May include one auto-sized dimension ([]).",
    },
];

const RESHAPE_INPUTS_DIMS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array/value.",
    },
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First output dimension size (or [] for auto).",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second output dimension size (or [] for auto).",
    },
];

const RESHAPE_INPUTS_DIMS_VARIADIC: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array/value.",
    },
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First output dimension size (or [] for auto).",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second output dimension size (or [] for auto).",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional output dimension sizes (or [] for auto).",
    },
];

const RESHAPE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "B = reshape(A, sz)",
        inputs: &RESHAPE_INPUTS_SZ,
        outputs: &RESHAPE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = reshape(A, m, n)",
        inputs: &RESHAPE_INPUTS_DIMS,
        outputs: &RESHAPE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = reshape(A, m, n, p...)",
        inputs: &RESHAPE_INPUTS_DIMS_VARIADIC,
        outputs: &RESHAPE_OUTPUT,
    },
];

const RESHAPE_ERROR_SIZE_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESHAPE.SIZE_MISSING",
    identifier: Some("RunMat:reshape:SizeMissing"),
    when: "No size vector/scalars are provided after A.",
    message: "reshape: size information missing",
};

const RESHAPE_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESHAPE.INVALID_SIZE",
    identifier: Some("RunMat:reshape:InvalidSize"),
    when:
        "Size arguments are malformed, non-scalar where scalar required, or contain invalid values.",
    message: "reshape: invalid size specification",
};

const RESHAPE_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RESHAPE.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:reshape:UnsupportedInput"),
    when: "Input value type is unsupported for reshape.",
    message: "reshape: unsupported input type",
};

const RESHAPE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    RESHAPE_ERROR_SIZE_MISSING,
    RESHAPE_ERROR_INVALID_SIZE,
    RESHAPE_ERROR_UNSUPPORTED_INPUT,
];

pub const RESHAPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RESHAPE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RESHAPE_ERRORS,
};

fn reshape_rank_from_args(args: &[Type]) -> Option<usize> {
    if args.len() < 2 {
        return None;
    }
    if args.len() > 2 {
        return Some(args.len() - 1);
    }
    match &args[1] {
        Type::Tensor { shape: Some(shape) } | Type::Logical { shape: Some(shape) } => {
            element_count_if_known(shape)
        }
        Type::Num | Type::Int | Type::Bool => Some(1),
        _ => None,
    }
}

fn reshape_type(args: &[Type], ctx: &ResolveContext) -> Type {
    let input = match args.first() {
        Some(value) => value,
        None => return Type::Unknown,
    };
    let dims = ctx.numeric_dims_from(1);
    let shape = if !dims.is_empty() {
        runmat_builtins::shape_rules::constructor_shape_from_dims(&dims)
    } else {
        let rank = reshape_rank_from_args(args);
        rank.map(unknown_shape)
    };
    match input {
        Type::Tensor { .. } => Type::Tensor { shape },
        Type::Logical { .. } => Type::Logical { shape },
        Type::Cell { element_type, .. } => Type::Cell {
            element_type: element_type.clone(),
            length: None,
        },
        Type::Num | Type::Int | Type::Bool => Type::Tensor { shape },
        Type::Unknown => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[runtime_builtin(
    name = "reshape",
    category = "array/shape",
    summary = "Rearrange the dimensions of an array without changing its data.",
    keywords = "reshape,resize,dimensions,gpu,auto",
    accel = "shape",
    type_resolver(reshape_type),
    descriptor(crate::builtins::array::shape::reshape::RESHAPE_DESCRIPTOR),
    builtin_path = "crate::builtins::array::shape::reshape"
)]
async fn reshape_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(reshape_error(RESHAPE_ERROR_SIZE_MISSING.message));
    }
    let tokens = parse_size_arguments(&rest).await?;
    let numel = value_numel(&value).await?;
    let dims = finalize_dimensions(tokens, numel)?;
    reshape_value(value, &dims)
}

fn reshape_value(value: Value, dims: &[usize]) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            tensor
                .reshape(dims.to_vec())
                .map(Value::Tensor)
                .map_err(|e| reshape_error(format!("reshape: {e}")))
        }
        Value::ComplexTensor(ct) => {
            ComplexTensor::from_complex_storage(ct.into_complex_storage(), dims.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|e| reshape_error(format!("reshape: {e}")))
        }
        Value::LogicalArray(logical) => {
            let LogicalArray { data, .. } = logical;
            LogicalArray::new(data, dims.to_vec())
                .map(Value::LogicalArray)
                .map_err(|e| reshape_error(format!("reshape: {e}")))
        }
        Value::String(s) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::String(s))
            } else {
                StringArray::new(vec![s], dims.to_vec())
                    .map(Value::StringArray)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::StringArray(strings) => {
            let StringArray { data, .. } = strings;
            StringArray::new(data, dims.to_vec())
                .map(Value::StringArray)
                .map_err(|e| reshape_error(format!("reshape: {e}")))
        }
        Value::CharArray(chars) => reshape_char_array(chars, dims),
        Value::Cell(cell) => reshape_cell_array(cell, dims),
        Value::ObjectArray(array) => {
            let class_name = array.class_name().to_string();
            runmat_value::ObjectArray::new(class_name, array.into_data(), dims.to_vec())
                .map(Value::ObjectArray)
                .map_err(|error| reshape_error(format!("reshape: {error}")))
        }
        Value::GpuTensor(handle) => reshape_gpu_tensor(handle, dims),
        Value::Num(n) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Num(n))
            } else {
                Tensor::new(vec![n], dims.to_vec())
                    .map(Value::Tensor)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::Int(i) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Int(i))
            } else {
                Tensor::new_integer(runmat_value::IntegerStorage::from_scalar(i), dims.to_vec())
                    .map(Value::Tensor)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::Bool(b) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Bool(b))
            } else {
                let fill = if b { 1u8 } else { 0u8 };
                let total: usize = dims.iter().product();
                LogicalArray::new(vec![fill; total], dims.to_vec())
                    .map(Value::LogicalArray)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::Complex(re, im) => reshape_complex_scalar(re, im, dims),
        other => Err(reshape_error(format!(
            "reshape: unsupported input type {:?}; expected numeric, logical, char, string, cell, or gpu array",
            other
        ))),
    }
}

fn reshape_complex_scalar(re: f64, im: f64, dims: &[usize]) -> crate::BuiltinResult<Value> {
    let total: usize = dims.iter().copied().product();
    if total != 1 {
        return Err(reshape_error(format!(
            "reshape: product of dimensions ({total}) must equal numel(A) (1)"
        )));
    }

    if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
        Ok(Value::Complex(re, im))
    } else {
        ComplexTensor::new(vec![(re, im)], dims.to_vec())
            .map(Value::ComplexTensor)
            .map_err(|e| reshape_error(format!("reshape: {e}")))
    }
}

fn reshape_char_array(ca: CharArray, dims: &[usize]) -> crate::BuiltinResult<Value> {
    if dims.is_empty() {
        return Err(reshape_error(
            "reshape: size vector must contain at least one element",
        ));
    }
    let shape = if dims.len() == 1 {
        vec![dims[0], 1]
    } else {
        dims.to_vec()
    };
    CharArray::from_column_major(ca.to_column_major(), shape)
        .map(Value::CharArray)
        .map_err(|e| reshape_error(format!("reshape: {e}")))
}

fn reshape_cell_array(ca: CellArray, dims: &[usize]) -> crate::BuiltinResult<Value> {
    if dims.is_empty() {
        return Err(reshape_error(
            "reshape: size vector must contain at least one element",
        ));
    }
    let shape = if dims.len() == 1 {
        vec![dims[0], 1]
    } else {
        dims.to_vec()
    };
    CellArray::from_column_major(ca.to_column_major(), shape)
        .map(Value::Cell)
        .map_err(|e| reshape_error(format!("reshape: {e}")))
}

fn reshape_gpu_tensor(
    handle: runmat_accelerate_api::GpuTensorHandle,
    dims: &[usize],
) -> crate::BuiltinResult<Value> {
    let complex_storage = runmat_accelerate_api::handle_storage(&handle)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
    if let Some(provider) =
        runmat_accelerate_api::provider_for_handle(&handle).or_else(runmat_accelerate_api::provider)
    {
        let reshaped = provider
            .reshape(&handle, dims)
            .map_err(|e| reshape_error(format!("reshape: {e}")))?;
        if complex_storage
            || runmat_accelerate_api::handle_storage(&reshaped)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            Ok(gpu_helpers::complex_gpu_value(reshaped))
        } else {
            Ok(gpu_helpers::resident_gpu_value(reshaped))
        }
    } else {
        let mut updated = handle;
        updated.shape = dims.to_vec();
        if complex_storage {
            Ok(gpu_helpers::complex_gpu_value(updated))
        } else {
            Ok(Value::GpuTensor(updated))
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum DimToken {
    Known(usize),
    Auto,
}

async fn parse_size_arguments(args: &[Value]) -> crate::BuiltinResult<Vec<DimToken>> {
    if args.len() == 1 {
        let value = &args[0];
        match value {
            Value::Tensor(t) => {
                if tensor::tensor_element_len(t) == 0 {
                    return Err(reshape_error(
                        "reshape: size vector must contain at least one element",
                    ));
                }
                let dims = tensor::dims_from_value_async(value)
                    .await
                    .map_err(|e| reshape_error(format!("reshape: {e}")))?;
                let Some(dims) = dims else {
                    return Err(reshape_error(
                        "reshape: size vector must be a row or column vector",
                    ));
                };
                Ok(dims.into_iter().map(DimToken::Known).collect())
            }
            Value::LogicalArray(la) => {
                if la.data.is_empty() {
                    return Err(reshape_error(
                        "reshape: size vector must contain at least one element",
                    ));
                }
                let dims = tensor::dims_from_value_async(value)
                    .await
                    .map_err(|e| reshape_error(format!("reshape: {e}")))?;
                let Some(dims) = dims else {
                    return Err(reshape_error(
                        "reshape: size vector must be a row or column vector",
                    ));
                };
                Ok(dims.into_iter().map(DimToken::Known).collect())
            }
            Value::GpuTensor(_) => {
                let dims = tensor::dims_from_value_async(value)
                    .await
                    .map_err(|e| reshape_error(format!("reshape: {e}")))?;
                let Some(dims) = dims else {
                    return Err(reshape_error(
                        "reshape: size vector must be a row or column vector",
                    ));
                };
                Ok(dims.into_iter().map(DimToken::Known).collect())
            }
            Value::Int(_) | Value::Num(_) | Value::Bool(_) => {
                Ok(vec![parse_size_scalar(value).await?])
            }
            other => Err(reshape_error(format!(
                "reshape: size vector must be numeric, got {:?}",
                other
            ))),
        }
    } else {
        let mut tokens = Vec::with_capacity(args.len());
        for value in args {
            tokens.push(parse_size_scalar(value).await?);
        }
        Ok(tokens)
    }
}
async fn parse_size_scalar(value: &Value) -> crate::BuiltinResult<DimToken> {
    match value {
        Value::Tensor(t) => {
            let len = tensor::tensor_element_len(t);
            if len == 0 {
                return Ok(DimToken::Auto);
            }
            if len != 1 {
                return Err(reshape_error("reshape: size arguments must be scalars"));
            }
        }
        Value::LogicalArray(la) => {
            if la.data.is_empty() {
                return Ok(DimToken::Auto);
            }
            if la.data.len() != 1 {
                return Err(reshape_error("reshape: size arguments must be scalars"));
            }
        }
        _ => {}
    }

    let Some(dim) = tensor::dimension_from_value_async(value, "reshape", true)
        .await
        .map_err(|e| reshape_error(format!("reshape: {e}")))?
    else {
        return Err(reshape_error(format!(
            "reshape: size arguments must be numeric scalars, got {:?}",
            value
        )));
    };
    Ok(DimToken::Known(dim))
}

fn finalize_dimensions(tokens: Vec<DimToken>, numel: usize) -> crate::BuiltinResult<Vec<usize>> {
    if tokens.is_empty() {
        return Err(reshape_error(
            "reshape: size vector must contain at least one element",
        ));
    }

    let mut dims = Vec::with_capacity(tokens.len());
    let mut known_product: usize = 1;
    let mut auto_index: Option<usize> = None;

    for (idx, token) in tokens.iter().enumerate() {
        match token {
            DimToken::Known(value) => {
                if *value == 0 {
                    known_product = 0;
                } else if known_product != 0 {
                    known_product = known_product.checked_mul(*value).ok_or_else(|| {
                        reshape_error("reshape: product of dimensions exceeds usize range")
                    })?;
                }
                dims.push(*value);
            }
            DimToken::Auto => {
                if auto_index.is_some() {
                    return Err(reshape_error(
                        "reshape: can only specify a single [] dimension",
                    ));
                }
                auto_index = Some(idx);
                dims.push(1); // placeholder
            }
        }
    }

    if let Some(auto) = auto_index {
        if known_product == 0 {
            if numel != 0 {
                return Err(reshape_error(format!(
                    "reshape: product of dimensions (0) must equal numel(A) ({numel})"
                )));
            }
            dims[auto] = 0;
        } else if !numel.is_multiple_of(known_product) {
            return Err(reshape_error(format!(
                "reshape: product of dimensions ({}) must equal numel(A) ({numel})",
                known_product
            )));
        } else {
            dims[auto] = numel / known_product;
        }
    } else if known_product != numel {
        return Err(reshape_error(format!(
            "reshape: product of dimensions ({known_product}) must equal numel(A) ({numel})"
        )));
    }

    Ok(dims)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use crate::dispatcher::download_handle_async;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;

    fn reshape_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::reshape_builtin(value, rest))
    }
    use crate::builtins::common::test_support;
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray};

    #[test]
    fn reshape_type_infers_rank_from_size_vector() {
        let size_vec = Type::Tensor {
            shape: Some(vec![Some(1), Some(2)]),
        };
        let out = reshape_type(
            &[Type::Tensor { shape: None }, size_vec],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[test]
    fn reshape_type_preserves_logical_kind() {
        let out = reshape_type(
            &[
                Type::Logical {
                    shape: Some(vec![Some(2), Some(2)]),
                },
                Type::Num,
                Type::Num,
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Logical {
                shape: Some(vec![None, None])
            }
        );
    }

    #[test]
    fn reshape_size_scalar_reads_typed_integer_storage_exactly() {
        let dim = Tensor::new_integer(IntegerStorage::U64(vec![3]), vec![1, 1])
            .expect("dimension tensor");

        match block_on(parse_size_scalar(&Value::Tensor(dim))).expect("dimension") {
            DimToken::Known(value) => assert_eq!(value, 3),
            other => panic!("expected known dimension, got {other:?}"),
        }
    }

    #[test]
    fn reshape_size_vector_reads_typed_integer_storage_without_mirror() {
        let dims = Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![1, 2])
            .expect("dimension tensor");

        let parsed =
            block_on(parse_size_arguments(&[Value::Tensor(dims)])).expect("dimension vector");
        assert!(matches!(
            parsed.as_slice(),
            [DimToken::Known(2), DimToken::Known(3)]
        ));
    }

    fn tensor_from_slice(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_vector_to_matrix() {
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[12, 1]);
        let result = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(tensor_from_slice(&[3.0, 4.0], &[1, 2]))],
        )
        .expect("reshape");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 4]);
                assert_eq!(
                    out.materialize_f64(),
                    (1..=12).map(|v| v as f64).collect::<Vec<_>>()
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_with_auto_dimension() {
        let data: Vec<f64> = (1..=18).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[18, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let args = vec![Value::from(3.0), Value::Tensor(empty)];
        let result = reshape_builtin(Value::Tensor(tensor), args).expect("reshape");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 6]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_logical_array_preserves_type() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0, 1, 0], vec![6, 1]).expect("logical");
        let result = reshape_builtin(
            Value::LogicalArray(logical),
            vec![Value::from(2.0), Value::from(3.0)],
        )
        .expect("reshape");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![1, 0, 1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_preserves_all_exact_integer_classes() {
        for storage in [
            IntegerStorage::I8(vec![i8::MIN, -1, 0, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, -1, 0, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, -1, 0, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, -1, 0, i64::MAX]),
            IntegerStorage::U8(vec![0, 1, 7, u8::MAX]),
            IntegerStorage::U16(vec![0, 1, 700, u16::MAX]),
            IntegerStorage::U32(vec![0, 1, 9_007_199, u32::MAX]),
            IntegerStorage::U64(vec![0, 1, 9_007_199_254_740_993, u64::MAX]),
        ] {
            let expected = storage.clone();
            let tensor = Tensor::new_integer(storage, vec![4, 1]).expect("integer tensor");
            let result = reshape_builtin(
                Value::Tensor(tensor),
                vec![Value::from(2.0), Value::from(2.0)],
            )
            .expect("reshape");

            match result {
                Value::Tensor(out) => {
                    assert_eq!(out.shape, vec![2, 2]);
                    assert_eq!(out.integer_storage(), Some(&expected));
                }
                other => panic!("expected typed integer tensor, got {other:?}"),
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_preserves_typed_complex_integer_storage() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![0, 1, 9_007_199_254_740_993, u64::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 1, 0]),
        )
        .expect("complex integer storage");
        let tensor =
            ComplexTensor::new_integer(storage.clone(), vec![4, 1]).expect("complex integer");

        let result = reshape_builtin(
            Value::ComplexTensor(tensor),
            vec![Value::from(2.0), Value::from(2.0)],
        )
        .expect("reshape");

        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.integer_storage().cloned(), Some(storage));
            }
            other => panic!("expected typed complex integer tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_char_array_single_dimension_becomes_column() {
        let chars = CharArray::new("abcd".chars().collect(), 1, 4).expect("char array");
        let result =
            reshape_builtin(Value::CharArray(chars), vec![Value::from(4.0)]).expect("reshape");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 4);
                assert_eq!(out.cols, 1);
                let collected: String = out.data.iter().collect();
                assert_eq!(collected, "abcd");
            }
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_cell_array_two_dimensional() {
        let cell = CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).expect("cell");
        let result = reshape_builtin(Value::Cell(cell), vec![Value::from(2.0), Value::from(1.0)])
            .expect("reshape");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 1);
                let first = out.get(0, 0).expect("first cell");
                let second = out.get(1, 0).expect("second cell");
                assert!(matches!(first, Value::Num(f) if (f - 1.0).abs() < 1e-12));
                assert!(matches!(second, Value::Num(f) if (f - 2.0).abs() < 1e-12));
            }
            other => panic!("expected CellArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_string_scalar_high_rank() {
        let result = reshape_builtin(
            Value::String("runmat".to_string()),
            vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)],
        )
        .expect("reshape");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1, 1]);
                assert_eq!(sa.data, vec!["runmat".to_string()]);
            }
            other => panic!("expected StringArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_gpu_preserves_handle_shape() {
        test_support::with_test_provider(|provider| {
            let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
            let tensor = tensor_from_slice(&data, &[3, 4]);
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = reshape_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::from(2.0), Value::from(6.0)],
            )
            .expect("reshape");
            match result {
                Value::GpuTensor(out) => {
                    assert_eq!(out.shape, vec![2, 6]);
                    assert_eq!(out.buffer_id, handle.buffer_id);
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn reshape_complex_gpu_preserves_storage_and_data() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(
                vec![
                    (1.0, -1.0),
                    (2.0, -2.0),
                    (3.0, -3.0),
                    (4.0, -4.0),
                    (5.0, -5.0),
                    (6.0, -6.0),
                ],
                vec![2, 3],
            )
            .unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = reshape_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::from(3.0), Value::from(2.0)],
            )
            .expect("reshape");
            let Value::GpuTensor(reshaped) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(reshaped.buffer_id, handle.buffer_id);
            assert_eq!(reshaped.shape, vec![3, 2]);
            assert_eq!(
                runmat_accelerate_api::handle_storage(&reshaped),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(
                crate::builtins::math::fft::common::gather_gpu_complex_tensor(&reshaped, "reshape"),
            )
            .expect("gather complex");
            assert_eq!(gathered.shape, vec![3, 2]);
            assert_eq!(gathered.materialize_f64(), complex.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn reshape_wgpu_updates_provider_shape() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[3, 4]);
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = reshape_builtin(
            Value::GpuTensor(handle.clone()),
            vec![Value::from(2.0), Value::from(6.0)],
        )
        .expect("reshape");
        let Value::GpuTensor(reshaped) = result else {
            panic!("expected gpu tensor");
        };
        assert_eq!(reshaped.shape, vec![2, 6]);
        let host = block_on(download_handle_async(provider, &reshaped)).expect("download");
        assert_eq!(host.shape, vec![2, 6]);
        assert_eq!(host.data, tensor.materialize_f64());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn reshape_wgpu_complex_updates_provider_shape_and_storage() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let complex = ComplexTensor::new(
            vec![
                (1.0, -1.0),
                (2.0, -2.0),
                (3.0, -3.0),
                (4.0, -4.0),
                (5.0, -5.0),
                (6.0, -6.0),
            ],
            vec![2, 3],
        )
        .unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let result = reshape_builtin(
            Value::GpuTensor(handle.clone()),
            vec![Value::from(3.0), Value::from(2.0)],
        )
        .expect("reshape");
        let Value::GpuTensor(reshaped) = result else {
            panic!("expected gpu tensor");
        };
        assert_eq!(reshaped.buffer_id, handle.buffer_id);
        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(&reshaped),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let host = block_on(download_handle_async(provider, &reshaped)).expect("download");
        assert_eq!(host.shape, vec![3, 2]);
        assert_eq!(
            host.storage,
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(
            host.data,
            vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0, 6.0, -6.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_mismatched_elements_errors() {
        let data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[6, 1]);
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(4.0), Value::from(4.0)],
        )
        .expect_err("should fail");
        assert!(err.to_string().contains("product of dimensions"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_multiple_auto_errors() {
        let data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[6, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(empty.clone()), Value::Tensor(empty)],
        )
        .expect_err("should fail");
        assert!(err.to_string().contains("single []"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_accepts_zero_sized_dimension() {
        let tensor = tensor_from_slice(&[], &[0, 1]);
        let result = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(0.0), Value::from(3.0)],
        )
        .expect("reshape zero");
        match result {
            Value::Tensor(out) => assert_eq!(out.shape, vec![0, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_int_scalar_to_vector() {
        let value = Value::Int(IntValue::I32(5));
        let err = reshape_builtin(value.clone(), vec![Value::from(1.0), Value::from(5.0)])
            .expect_err("should fail because numel mismatch");
        assert!(err.to_string().contains("numel"));
        let ok = reshape_builtin(value, vec![Value::from(1.0), Value::from(1.0)])
            .expect("reshape scalar");
        assert!(matches!(ok, Value::Int(_)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_int_scalar_high_rank_preserves_exact_integer_storage() {
        let result = reshape_builtin(
            Value::Int(IntValue::U64(9_007_199_254_740_993)),
            vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)],
        )
        .expect("reshape scalar");

        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![1, 1, 1]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::U64(vec![9_007_199_254_740_993]))
                );
            }
            other => panic!("expected typed integer tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_complex_scalar_high_rank() {
        let result = reshape_builtin(
            Value::Complex(1.0, 2.0),
            vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)],
        )
        .expect("reshape complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 1, 1]);
                assert_eq!(ct.materialize_f64(), vec![(1.0, 2.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_auto_dimension_mismatch_reports_product() {
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[12, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(5.0), Value::Tensor(empty)],
        )
        .expect_err("should fail");
        assert!(
            err.to_string().contains("5"),
            "expected product to appear in error message, got {err}"
        );
    }
}
