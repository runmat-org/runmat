//! MATLAB-compatible `cell` builtin implemented for the modern RunMat runtime.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexTensor, IntValue, LogicalArray, StringArray, StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::cells::type_resolvers::cell_type;
use crate::builtins::common::random_args::{keyword_of, shape_from_value};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::cells::core::cell")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cell",
    op_kind: GpuOpKind::Custom("container"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Cell arrays are allocated on the host heap; providers currently gather any GPU inputs and rely on host execution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::cells::core::cell")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cell",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Cell creation acts as a fusion sink and terminates GPU fusion plans.",
};

const BUILTIN_NAME: &str = "cell";

const CELL_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cell-like",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the cell \"like\" prototype selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CellLikeExtension"),
};

const CELL_GPU_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cell-gpu-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "resident GPU size controls for cell are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CellGpuSizeExtension"),
};

pub const CELL_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [CELL_LIKE_EXTENSION, CELL_GPU_SIZE_EXTENSION];

const CELL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output cell array.",
}];

const CELL_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const CELL_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const CELL_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector.",
}];

const CELL_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const CELL_SIG_LIKE_ONLY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype value used to infer element class.",
    },
];

const CELL_SIG_SIZE_VECTOR_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size vector.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype value used to infer element class.",
    },
];

const CELL_SIG_DIMS_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype value used to infer element class.",
    },
];

const CELL_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "C = cell()",
        inputs: &CELL_SIG_EMPTY_INPUTS,
        outputs: &CELL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(n)",
        inputs: &CELL_SIG_N_INPUTS,
        outputs: &CELL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(sz)",
        inputs: &CELL_SIG_SIZE_VECTOR_INPUTS,
        outputs: &CELL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(m, n, ...)",
        inputs: &CELL_SIG_DIMS_INPUTS,
        outputs: &CELL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(\"like\", prototype)",
        inputs: &CELL_SIG_LIKE_ONLY_INPUTS,
        outputs: &CELL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(sz, \"like\", prototype)",
        inputs: &CELL_SIG_SIZE_VECTOR_LIKE_INPUTS,
        outputs: &CELL_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(m, n, ..., \"like\", prototype)",
        inputs: &CELL_SIG_DIMS_LIKE_INPUTS,
        outputs: &CELL_OUTPUT,
    },
];

const CELL_INTEGER_SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n_or_size_dimensions",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Square, variadic-dimension, and row-size-vector controls accept every integer class and are decoded from authoritative integer storage without binary64 conversion.",
    }];

pub const CELL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = cell(n), cell(sz1, ..., szN), or cell(sz) with integer size controls",
        inputs: &CELL_INTEGER_SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Signed negative dimensions clamp to zero, oversized dimensions fail before allocation, and resident size controls use an independently mode-gated exact gather fallback; every cell allocated by a documented form contains a host 0-by-0 double array.",
    }];

const CELL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL.INVALID_INPUT",
    identifier: Some("RunMat:cell:InvalidInput"),
    when: "Input arguments or option forms are invalid.",
    message: "cell: invalid input arguments",
};

const CELL_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL.INVALID_SIZE",
    identifier: Some("RunMat:cell:InvalidSize"),
    when: "Requested size arguments are invalid or unsupported.",
    message: "cell: invalid size arguments",
};

const CELL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL.INTERNAL",
    identifier: None,
    when: "Internal cell allocation or conversion failed.",
    message: "cell: internal error",
};

const CELL_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CELL_ERROR_INVALID_INPUT,
    CELL_ERROR_INVALID_SIZE,
    CELL_ERROR_INTERNAL,
];

pub const CELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CELL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CELL_ERRORS,
};

fn cell_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "cell",
    category = "cells/core",
    summary = "Create empty cell arrays.",
    keywords = "cell,cell array,container,empty",
    accel = "array_construct",
    sink = true,
    type_resolver(cell_type),
    descriptor(crate::builtins::cells::core::cell::CELL_DESCRIPTOR),
    extensions(crate::builtins::cells::core::cell::CELL_EXTENSIONS),
    integer_capabilities(crate::builtins::cells::core::cell::CELL_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::cells::core::cell"
)]
async fn cell_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedCell::parse(args).await?;
    build_cell(parsed)
}

struct ParsedCell {
    shape: Vec<usize>,
    prototype: Option<Value>,
}

impl ParsedCell {
    async fn parse(args: Vec<Value>) -> BuiltinResult<Self> {
        let mut dims: Vec<Value> = Vec::new();
        let mut prototype: Option<Value> = None;
        let mut idx = 0;

        while idx < args.len() {
            let value = &args[idx];
            if let Some(keyword) = keyword_of(value) {
                match keyword.as_str() {
                    "like" => {
                        crate::compatibility::ensure_builtin_extension_enabled(
                            &CELL_LIKE_EXTENSION,
                            "cell",
                        )?;
                        if prototype.is_some() {
                            return Err(cell_error_with_message(
                                "cell: multiple 'like' specifications are not supported",
                                &CELL_ERROR_INVALID_INPUT,
                            ));
                        }
                        let Some(proto) = args.get(idx + 1) else {
                            return Err(cell_error_with_message(
                                "cell: expected prototype after 'like'",
                                &CELL_ERROR_INVALID_INPUT,
                            ));
                        };
                        prototype = Some(proto.clone());
                        idx += 2;
                        continue;
                    }
                    other => {
                        return Err(cell_error_with_message(
                            format!("cell: unrecognised option '{other}'"),
                            &CELL_ERROR_INVALID_INPUT,
                        ));
                    }
                }
            }

            if matches!(args[idx], Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &CELL_GPU_SIZE_EXTENSION,
                    "cell",
                )?;
            }
            dims.push(args[idx].clone());
            idx += 1;
        }

        let shape = parse_shape_arguments(&dims, prototype.as_ref()).await?;
        Ok(Self { shape, prototype })
    }
}

fn build_cell(parsed: ParsedCell) -> BuiltinResult<Value> {
    let shape = normalize_shape(parsed.shape);
    let total = if shape.is_empty() {
        0
    } else {
        shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| {
                cell_error_with_message(
                    "cell: requested size exceeds platform limits",
                    &CELL_ERROR_INVALID_SIZE,
                )
            })?
    };

    if total == 0 {
        return make_cell_with_shape(Vec::new(), shape)
            .map_err(|e| cell_error_with_message(format!("cell: {e}"), &CELL_ERROR_INTERNAL));
    }

    let default_value = empty_value_like(parsed.prototype.as_ref())?;
    let mut values = Vec::with_capacity(total);
    values.resize(total, default_value);
    make_cell_with_shape(values, shape)
        .map_err(|e| cell_error_with_message(format!("cell: {e}"), &CELL_ERROR_INTERNAL))
}

fn normalize_shape(mut dims: Vec<usize>) -> Vec<usize> {
    while dims.len() > 2 && dims.last() == Some(&1) {
        dims.pop();
    }
    match dims.len() {
        0 => vec![0, 0],
        1 => vec![dims[0], dims[0]],
        _ => dims,
    }
}

async fn parse_shape_arguments(
    args: &[Value],
    prototype: Option<&Value>,
) -> BuiltinResult<Vec<usize>> {
    if args.is_empty() {
        if let Some(proto) = prototype {
            return shape_from_value(proto, "cell")
                .map_err(|err| cell_error_with_message(err, &CELL_ERROR_INVALID_INPUT));
        }
        return Ok(vec![0, 0]);
    }

    if args.len() == 1 {
        let host = gather_if_needed_async(&args[0]).await?;
        return parse_single_argument(&host);
    }

    let mut dims = Vec::with_capacity(args.len());
    for value in args {
        let host = gather_if_needed_async(value).await?;
        dims.push(parse_size_scalar(&host, "cell")?);
    }
    Ok(dims)
}

fn parse_single_argument(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(_) | Value::Num(_) => {
            let n = parse_size_scalar(value, "cell")?;
            Ok(vec![n, n])
        }
        Value::Tensor(t) => parse_size_tensor(t),
        other => Err(cell_error_with_message(
            format!("cell: size arguments must be numeric scalars or vectors, got {other:?}"),
            &CELL_ERROR_INVALID_INPUT,
        )),
    }
}

fn parse_size_scalar(value: &Value, context: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(iv) => parse_intvalue(iv, context),
        Value::Num(n) => parse_numeric(*n, context),
        Value::Tensor(t) => {
            if !tensor::is_scalar_tensor(t) {
                return Err(cell_error_with_message(
                    format!("{context}: size inputs must be scalar"),
                    &CELL_ERROR_INVALID_SIZE,
                ));
            }
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                parse_intvalue(&int, context)
            } else {
                parse_numeric(tensor::tensor_value_f64(t, 0), context)
            }
        }
        other => Err(cell_error_with_message(
            format!("{context}: size inputs must be numeric scalars, got {other:?}"),
            &CELL_ERROR_INVALID_INPUT,
        )),
    }
}

fn parse_size_tensor(t: &Tensor) -> BuiltinResult<Vec<usize>> {
    let len = tensor::tensor_element_len(t);
    if len == 0 {
        return Ok(vec![0, 0]);
    }
    if !is_row_vector_shape(&t.shape) {
        return Err(cell_error_with_message(
            "cell: size vector must be a row vector",
            &CELL_ERROR_INVALID_SIZE,
        ));
    }
    let dims = t
        .integer_storage()
        .map(|storage| {
            (0..storage.len())
                .map(|index| {
                    storage.value_at(index).ok_or_else(|| {
                        cell_error_with_message(
                            "cell: size vector storage is inconsistent",
                            &CELL_ERROR_INTERNAL,
                        )
                    })
                })
                .map(|value| value.and_then(|int| parse_intvalue(&int, "cell")))
                .collect::<Result<Vec<_>, _>>()
        })
        .unwrap_or_else(|| {
            (0..t.len())
                .map(|index| parse_numeric(tensor::tensor_value_f64(t, index), "cell"))
                .collect::<Result<Vec<_>, _>>()
        })?;
    if dims.len() == 1 {
        Ok(vec![dims[0], dims[0]])
    } else {
        Ok(dims)
    }
}

fn is_row_vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] <= 1,
        _ => false,
    }
}

fn empty_value_like(proto: Option<&Value>) -> BuiltinResult<Value> {
    match proto {
        Some(value) => match value {
            Value::LogicalArray(_) | Value::Bool(_) => LogicalArray::new(Vec::new(), vec![0, 0])
                .map(Value::LogicalArray)
                .map_err(|e| cell_error_with_message(format!("cell: {e}"), &CELL_ERROR_INTERNAL)),
            Value::ComplexTensor(_) | Value::Complex(_, _) => {
                ComplexTensor::new(Vec::new(), vec![0, 0])
                    .map(Value::ComplexTensor)
                    .map_err(|e| {
                        cell_error_with_message(format!("cell: {e}"), &CELL_ERROR_INTERNAL)
                    })
            }
            Value::String(_) => Ok(Value::String(String::new())),
            Value::StringArray(_) => StringArray::new(Vec::new(), vec![0, 0])
                .map(Value::StringArray)
                .map_err(|e| cell_error_with_message(format!("cell: {e}"), &CELL_ERROR_INTERNAL)),
            Value::CharArray(_) => CharArray::new(Vec::new(), 0, 0)
                .map(Value::CharArray)
                .map_err(|e| cell_error_with_message(format!("cell: {e}"), &CELL_ERROR_INTERNAL)),
            Value::Cell(_) => make_cell_with_shape(Vec::new(), vec![0, 0])
                .map_err(|e| cell_error_with_message(format!("cell: {e}"), &CELL_ERROR_INTERNAL)),
            Value::Struct(_) => Ok(Value::Struct(StructValue::new())),
            Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::Num(_)
            | Value::Int(_)
            | Value::GpuTensor(_) => default_empty_double(),
            Value::Object(_)
            | Value::HandleObject(_)
            | Value::Listener(_)
            | Value::Symbolic(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
            | Value::ClassRef(_)
            | Value::MException(_)
            | Value::OutputList(_) => default_empty_double(),
        },
        None => default_empty_double(),
    }
}

fn default_empty_double() -> BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|e| cell_error_with_message(format!("cell: {e}"), &CELL_ERROR_INTERNAL))
}

fn parse_intvalue(value: &IntValue, _context: &str) -> BuiltinResult<usize> {
    let raw = match value {
        IntValue::I8(v) => *v as i128,
        IntValue::I16(v) => *v as i128,
        IntValue::I32(v) => *v as i128,
        IntValue::I64(v) => *v as i128,
        IntValue::U8(v) => *v as i128,
        IntValue::U16(v) => *v as i128,
        IntValue::U32(v) => *v as i128,
        IntValue::U64(v) => *v as i128,
    };
    if raw < 0 {
        return Ok(0);
    }
    if raw as u128 > usize::MAX as u128 {
        return Err(cell_error_with_message(
            "cell: requested size exceeds platform limits",
            &CELL_ERROR_INVALID_SIZE,
        ));
    }
    Ok(raw as usize)
}

fn parse_numeric(value: f64, context: &str) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(cell_error_with_message(
            format!("{context}: size inputs must be finite"),
            &CELL_ERROR_INVALID_SIZE,
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(cell_error_with_message(
            format!("{context}: size inputs must be integers"),
            &CELL_ERROR_INVALID_SIZE,
        ));
    }
    if rounded < 0.0 {
        return Ok(0);
    }
    if rounded > (1u64 << 53) as f64 {
        return Err(cell_error_with_message(
            "cell: size inputs larger than 2^53 are not supported",
            &CELL_ERROR_INVALID_SIZE,
        ));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(cell_error_with_message(
            "cell: requested size exceeds platform limits",
            &CELL_ERROR_INVALID_SIZE,
        ));
    }
    Ok(rounded as usize)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;

    fn cell_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::cell_builtin(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn descriptor_signatures_cover_cell_forms() {
        let labels: Vec<&str> = CELL_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"C = cell()"));
        assert!(labels.contains(&"C = cell(sz)"));
        assert!(labels.contains(&"C = cell(m, n, ..., \"like\", prototype)"));
        assert_eq!(CELL_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(
            CELL_INTEGER_CAPABILITIES[0].inputs[0].classes,
            crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES
        );
    }

    fn expect_cell_with<F>(value: Value, expected_shape: &[usize], mut check: F)
    where
        F: FnMut(&Value),
    {
        match value {
            Value::Cell(cell) => {
                assert_eq!(cell.shape, expected_shape, "shape mismatch");
                let expected_rows = expected_shape.first().copied().unwrap_or(0);
                let expected_cols = match expected_shape.len() {
                    0 => 0,
                    1 => 1,
                    _ => expected_shape[1],
                };
                assert_eq!(cell.rows, expected_rows, "rows mismatch");
                assert_eq!(cell.cols, expected_cols, "cols mismatch");
                let expected_total = expected_shape
                    .iter()
                    .fold(1usize, |acc, &dim| acc.saturating_mul(dim));
                let expected_total = if expected_shape.is_empty() {
                    0
                } else {
                    expected_total
                };
                assert_eq!(cell.data.len(), expected_total, "element count mismatch");
                for handle in cell.data {
                    let element = handle;
                    check(&element);
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    fn expect_cell(value: Value, expected_shape: &[usize]) {
        expect_cell_with(value, expected_shape, |element| match element {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty double array, found {other:?}"),
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_no_arguments_returns_empty() {
        let result = cell_builtin(Vec::new()).expect("cell()");
        expect_cell(result, &[0, 0]);
    }

    #[test]
    fn cell_size_tensor_preserves_typed_integer_bounds() {
        let dims =
            Tensor::new_integer(runmat_builtins::IntegerStorage::U64(vec![2, 3]), vec![1, 2])
                .expect("dims");
        assert_eq!(parse_size_tensor(&dims).unwrap(), vec![2, 3]);

        let scalar = Tensor::new_integer(runmat_builtins::IntegerStorage::U16(vec![4]), vec![1, 1])
            .expect("scalar");
        assert_eq!(
            parse_size_scalar(&Value::Tensor(scalar), "cell").unwrap(),
            4
        );

        let negative =
            Tensor::new_integer(runmat_builtins::IntegerStorage::I16(vec![-1]), vec![1, 1])
                .expect("negative");
        assert_eq!(parse_size_tensor(&negative).unwrap(), vec![0, 0]);

        assert!(parse_size_scalar(&Value::Num(usize::MAX as f64), "cell").is_err());
        assert!(parse_size_scalar(&Value::Num((usize::MAX as f64) + 1.0), "cell").is_err());
    }

    #[test]
    fn cell_size_tensor_reads_native_single_storage() {
        let dims = Tensor::from_f32(vec![2.0, 3.0], vec![1, 2]).expect("single dims");
        assert_eq!(parse_size_tensor(&dims).unwrap(), vec![2, 3]);

        let scalar = Tensor::from_f32(vec![4.0], vec![1, 1]).expect("single scalar");
        assert_eq!(
            parse_size_scalar(&Value::Tensor(scalar), "cell").unwrap(),
            4
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_requires_prototype() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = cell_builtin(vec![Value::from("like")])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("expected prototype"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_two_sizes() {
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = cell_builtin(args).expect("cell(2,4)");
        expect_cell(result, &[2, 4]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_size_vector() {
        let tensor = Tensor::new(vec![2.0, 5.0], vec![1, 2]).unwrap();
        let result = cell_builtin(vec![Value::Tensor(tensor)]).expect("cell([2 5])");
        expect_cell(result, &[2, 5]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_column_size_vector() {
        let tensor = Tensor::new(vec![4.0, 1.0], vec![2, 1]).unwrap();
        let error = cell_builtin(vec![Value::Tensor(tensor)]).unwrap_err();
        assert!(error.message().contains("row vector"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_accepts_gpu_size_vector() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.0, 2.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload size vector");
            let result = cell_builtin(vec![Value::GpuTensor(handle)]).expect("cell(gpu size)");
            expect_cell(result, &[3, 2]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cell_wgpu_size_vector_and_like() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![2.0, 3.0, 1.0], vec![1, 3]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let handle = provider.upload(&view).expect("upload size vector");
        let result = cell_builtin(vec![Value::GpuTensor(handle)]).expect("cell(wgpu size)");
        expect_cell(result, &[2, 3]);

        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let proto_view = runmat_accelerate_api::HostTensorView {
            data: &proto.materialize_f64(),
            shape: &proto.shape,
        };
        let proto_handle = provider.upload(&proto_view).expect("upload prototype");
        let like_result = cell_builtin(vec![Value::from("like"), Value::GpuTensor(proto_handle)])
            .expect("cell('like', gpu prototype)");
        expect_cell(like_result, &[2, 3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_multi_dimensional_vector() {
        let tensor = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let result = cell_builtin(vec![Value::Tensor(tensor)]).expect("cell([2 3 4])");
        expect_cell(result, &[2, 3, 4]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_variadic_dimensions() {
        let args = vec![Value::Num(2.0), Value::Num(3.0), Value::Num(5.0)];
        let result = cell_builtin(args).expect("cell(2,3,5)");
        expect_cell(result, &[2, 3, 5]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_with_single_element_vector_is_square() {
        let tensor = Tensor::new(vec![4.0], vec![1, 1]).unwrap();
        let result = cell_builtin(vec![Value::Tensor(tensor)]).expect("cell([4])");
        expect_cell(result, &[4, 4]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_clamps_negative_sizes_to_zero() {
        expect_cell(
            cell_builtin(vec![Value::Num(-1.0)]).expect("negative square"),
            &[0, 0],
        );
        expect_cell(
            cell_builtin(vec![
                Value::Int(IntValue::I64(-7)),
                Value::Int(IntValue::U8(3)),
            ])
            .expect("negative dimension"),
            &[0, 3],
        );
    }

    #[test]
    fn cell_accepts_all_eight_integer_size_classes_exactly() {
        use runmat_builtins::IntegerStorage;

        let storages = [
            IntegerStorage::I8(vec![2, 3]),
            IntegerStorage::I16(vec![2, 3]),
            IntegerStorage::I32(vec![2, 3]),
            IntegerStorage::I64(vec![2, 3]),
            IntegerStorage::U8(vec![2, 3]),
            IntegerStorage::U16(vec![2, 3]),
            IntegerStorage::U32(vec![2, 3]),
            IntegerStorage::U64(vec![2, 3]),
        ];
        for storage in storages {
            let dims = Tensor::new_integer(storage, vec![1, 2]).expect("integer size vector");
            expect_cell(
                cell_builtin(vec![Value::Tensor(dims)]).expect("integer cell size"),
                &[2, 3],
            );
        }
    }

    #[test]
    fn cell_ignores_trailing_singleton_dimensions_after_second() {
        expect_cell(
            cell_builtin(vec![
                Value::Int(IntValue::U8(3)),
                Value::Int(IntValue::U8(1)),
                Value::Int(IntValue::U8(1)),
                Value::Int(IntValue::U8(1)),
            ])
            .expect("trailing singletons"),
            &[3, 1],
        );
    }

    #[test]
    fn cell_rejects_logical_size_controls() {
        assert!(cell_builtin(vec![Value::Bool(true)]).is_err());
        let logical = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        assert!(cell_builtin(vec![Value::LogicalArray(logical)]).is_err());
    }

    #[test]
    fn cell_extensions_follow_compatibility_mode() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
        expect_cell(cell_builtin(Vec::new()).expect("cell() shorthand"), &[0, 0]);
        let like = cell_builtin(vec![Value::from("like"), Value::Num(1.0)]).unwrap_err();
        assert_eq!(
            like.identifier(),
            Some("RunMat:compatibility:CellLikeExtension")
        );
        let resident = runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 2],
            device_id: u32::MAX,
            buffer_id: u64::MAX - 388,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(
            runmat_accelerate_api::NumericElementType::U64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
        let gpu = cell_builtin(vec![Value::GpuTensor(resident.clone())]).unwrap_err();
        assert_eq!(
            gpu.identifier(),
            Some("RunMat:compatibility:CellGpuSizeExtension")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_rejects_fractional() {
        let err = cell_builtin(vec![Value::Num(2.5)]).unwrap_err().to_string();
        assert!(err.contains("integers"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_infers_shape_from_prototype() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(proto)];
        let result = cell_builtin(args).expect("cell('like', tensor)");
        expect_cell(result, &[2, 2]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_logical_uses_logical_empty() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let logical = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let args = vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::LogicalArray(logical),
        ];
        let result = cell_builtin(args).expect("cell(___, 'like', logical)");
        expect_cell_with(result, &[2, 2], |element| match element {
            Value::LogicalArray(arr) => {
                assert!(arr.data.is_empty());
                assert_eq!(arr.shape, vec![0, 0]);
            }
            other => panic!("expected logical empty, got {other:?}"),
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_cell_prototype_produces_empty_cell_elements() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let proto = crate::make_cell_with_shape(Vec::new(), vec![0, 0]).unwrap();
        let args = vec![Value::Num(1.0), Value::from("like"), proto.clone()];
        let result = cell_builtin(args).expect("cell(1,'like',cell)");
        expect_cell_with(result, &[1, 1], |element| match element {
            Value::Cell(inner) => {
                assert_eq!(inner.shape, vec![0, 0]);
                assert_eq!(inner.data.len(), 0);
            }
            other => panic!("expected nested empty cell, got {other:?}"),
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_is_case_insensitive() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let proto = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let result = cell_builtin(vec![Value::from("LIKE"), Value::Tensor(proto)])
            .expect("cell('LIKE', ...)");
        expect_cell(result, &[1, 1]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_rejects_multiple_keywords() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let proto = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let err = cell_builtin(vec![
            Value::Num(1.0),
            Value::from("like"),
            Value::Tensor(proto.clone()),
            Value::from("like"),
            Value::Tensor(proto),
        ])
        .unwrap_err()
        .to_string();
        assert!(err.contains("multiple 'like'"), "unexpected error: {err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell_like_gpu_prototype_falls_back_to_host() {
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload prototype");
            let result = cell_builtin(vec![Value::from("like"), Value::GpuTensor(handle)])
                .expect("cell('like', gpu)");
            expect_cell(result, &[2, 1]);
        });
    }
}
