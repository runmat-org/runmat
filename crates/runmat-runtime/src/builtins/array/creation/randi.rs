//! MATLAB-compatible `randi` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{
    GpuTensorHandle, HostIntegerDataView, HostIntegerTensorView, HostTensorView,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, IntegerStorage, LogicalArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::array::type_resolvers::tensor_type_from_rank;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{random, tensor};
use crate::builtins::math::elementwise::integer_cast::IntegerTarget;
use runmat_builtins::{ResolveContext, Type};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::randi")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "randi",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("random_integer_range"),
        ProviderHook::Custom("random_integer_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may offer integer RNG kernels via random_integer_range / random_integer_like; integer gpuArray prototypes stay provider-resident through random generation plus native integer cast when supported, with exact host upload retained for bounds outside the provider contract.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("randi").build()
}

fn randi_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Unknown;
    }
    if args.len() == 1 {
        return Type::Num;
    }
    let rest = &args[1..];
    if rest.iter().any(|arg| matches!(arg, Type::String)) {
        return Type::Unknown;
    }
    let rest_ctx = ResolveContext::new(ctx.literal_args.get(1..).unwrap_or(&[]).to_vec());
    tensor_type_from_rank(rest, &rest_ctx)
}

const RANDI_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Uniform random integers.",
}];

const RANDI_SIG_IMAX_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "imax",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper bound; lower bound defaults to 1.",
}];

const RANDI_SIG_BOUNDS_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bounds",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element bounds vector [imin imax].",
}];

const RANDI_SIG_IMAX_N_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "imax",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound; lower bound defaults to 1.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Square size.",
    },
];

const RANDI_SIG_BOUNDS_N_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element bounds vector [imin imax].",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Square size.",
    },
];

const RANDI_SIG_IMAX_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "imax",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound; lower bound defaults to 1.",
    },
    BuiltinParamDescriptor {
        name: "size_vector",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Size vector defining output dimensions.",
    },
];

const RANDI_SIG_BOUNDS_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element bounds vector [imin imax].",
    },
    BuiltinParamDescriptor {
        name: "size_vector",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Size vector defining output dimensions.",
    },
];

const RANDI_SIG_IMAX_DIMS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "imax",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound; lower bound defaults to 1.",
    },
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
];

const RANDI_SIG_BOUNDS_DIMS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element bounds vector [imin imax].",
    },
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
];

const RANDI_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound scalar or two-element bounds vector.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Class override ('double'|'logical').",
    },
];

const RANDI_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound scalar or two-element bounds vector.",
    },
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
        description: "Prototype array used for class/device.",
    },
];

const RANDI_SIG_BOUNDS_PROTOTYPE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound scalar or two-element bounds vector.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype value when no numeric dimension arguments are provided.",
    },
];

const RANDI_SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "R = randi(imax)",
        inputs: &RANDI_SIG_IMAX_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi([imin imax])",
        inputs: &RANDI_SIG_BOUNDS_VECTOR_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(imax, n)",
        inputs: &RANDI_SIG_IMAX_N_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi([imin imax], n)",
        inputs: &RANDI_SIG_BOUNDS_N_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(imax, size_vector)",
        inputs: &RANDI_SIG_IMAX_SIZE_VECTOR_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi([imin imax], size_vector)",
        inputs: &RANDI_SIG_BOUNDS_SIZE_VECTOR_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(imax, m, n, ...)",
        inputs: &RANDI_SIG_IMAX_DIMS_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi([imin imax], m, n, ...)",
        inputs: &RANDI_SIG_BOUNDS_DIMS_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(..., typename)",
        inputs: &RANDI_SIG_CLASS_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(..., \"like\", prototype)",
        inputs: &RANDI_SIG_LIKE_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(bounds, prototype)",
        inputs: &RANDI_SIG_BOUNDS_PROTOTYPE_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
];

const RANDI_ERRORS: [BuiltinErrorDescriptor; 9] = [
    BuiltinErrorDescriptor {
        code: "RM.RANDI.MISSING_BOUNDS",
        identifier: None,
        when: "No bounds argument is provided.",
        message: "randi: requires at least one input argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.INVALID_BOUNDS",
        identifier: None,
        when: "Bounds are invalid or unsupported (empty, non-finite, non-integer, or malformed).",
        message: "randi: bounds must be numeric scalars or vectors",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.BOUNDS_ORDER",
        identifier: None,
        when: "Lower bound exceeds upper bound.",
        message: "randi: lower bound must be <= upper bound",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "randi: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "randi: multiple 'like' specifications are not supported",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.CLASS_CONFLICT",
        identifier: None,
        when: "A class keyword and a 'like' prototype are both provided.",
        message: "randi: cannot combine 'like' with class specifiers",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.UNSUPPORTED_CLASS",
        identifier: None,
        when: "An unsupported output class is requested.",
        message: "randi: output class is not implemented",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not recognized.",
        message: "randi: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.INVALID_DIMS",
        identifier: None,
        when: "Dimension arguments fail numeric/shape parsing.",
        message: "randi: dimension arguments must be numeric and nonnegative",
    },
];

pub const RANDI_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RANDI_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RANDI_ERRORS,
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::randi")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "randi",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Random integer generation is treated as a sink and excluded from fusion planning.",
};

#[runtime_builtin(
    name = "randi",
    category = "array/creation",
    summary = "Uniform random integers with inclusive bounds.",
    keywords = "randi,random,integer,gpu,like",
    accel = "array_construct",
    type_resolver(randi_type),
    descriptor(crate::builtins::array::creation::randi::RANDI_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::randi"
)]
async fn randi_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let parsed = ParsedRandi::parse(args).await?;
    build_output(parsed).await
}

struct ParsedRandi {
    bounds: Bounds,
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    Single,
    Logical,
    Integer(IntegerTarget),
    Like(Value),
}

#[derive(Clone, Copy)]
struct Bounds {
    lower: i128,
    upper: i128,
    span: u64,
}

impl Bounds {
    fn new(lower: i128, upper: i128) -> crate::BuiltinResult<Self> {
        if lower > upper {
            return Err(builtin_error("randi: lower bound must be <= upper bound"));
        }
        let span = upper
            .checked_sub(lower)
            .and_then(|delta| delta.checked_add(1))
            .ok_or_else(|| builtin_error("randi: range width overflows 64-bit arithmetic"))?;
        if span <= 0 {
            return Err(builtin_error("randi: invalid bounds"));
        }
        if span > (1u64 << 53) as i128 {
            return Err(builtin_error(
                "randi: range width exceeds RNG precision (2^53)",
            ));
        }
        Ok(Self {
            lower,
            upper,
            span: span as u64,
        })
    }
}

impl ParsedRandi {
    async fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
        if args.is_empty() {
            return Err(builtin_error("randi: requires at least one input argument"));
        }

        let mut iter = args.into_iter();
        let bounds_value = iter.next().unwrap();
        let bounds = parse_bounds(bounds_value).await?;

        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<OutputTemplate> = None;
        let mut implicit_proto: Option<Value> = None;

        let rest: Vec<Value> = iter.collect();
        let mut idx = 0;
        while idx < rest.len() {
            let arg = rest[idx].clone();
            if let Some(keyword) = keyword_of(&arg) {
                if matches!(
                    keyword.as_str(),
                    "double"
                        | "single"
                        | "logical"
                        | "int8"
                        | "int16"
                        | "int32"
                        | "int64"
                        | "uint8"
                        | "uint16"
                        | "uint32"
                        | "uint64"
                ) && like_proto.is_some()
                {
                    return Err(builtin_error(format!(
                        "randi: cannot combine 'like' with '{keyword}'"
                    )));
                }
                match keyword.as_str() {
                    "like" => {
                        if like_proto.is_some() {
                            return Err(builtin_error(
                                "randi: multiple 'like' specifications are not supported",
                            ));
                        }
                        if let Some(spec) = &class_override {
                            let keyword = match spec {
                                OutputTemplate::Logical => "'logical'",
                                OutputTemplate::Double => "'double'",
                                OutputTemplate::Single => "'single'",
                                OutputTemplate::Integer(_) => "an integer class specifier",
                                OutputTemplate::Like(_) => "another class specifier",
                            };
                            return Err(builtin_error(format!(
                                "randi: cannot combine 'like' with {keyword}"
                            )));
                        }
                        let Some(proto) = rest.get(idx + 1).cloned() else {
                            return Err(builtin_error("randi: expected prototype after 'like'"));
                        };
                        like_proto = Some(proto.clone());
                        if shape_source.is_none() && !saw_dims_arg {
                            shape_source = Some(shape_from_value(&proto)?);
                        }
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        if like_proto.is_some() {
                            return Err(builtin_error(
                                "randi: cannot combine 'like' with 'double'",
                            ));
                        }
                        class_override = Some(OutputTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "logical" => {
                        if like_proto.is_some() {
                            return Err(builtin_error(
                                "randi: cannot combine 'like' with 'logical'",
                            ));
                        }
                        class_override = Some(OutputTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        class_override = Some(OutputTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    "int8" => class_override = Some(OutputTemplate::Integer(IntegerTarget::I8)),
                    "int16" => class_override = Some(OutputTemplate::Integer(IntegerTarget::I16)),
                    "int32" => class_override = Some(OutputTemplate::Integer(IntegerTarget::I32)),
                    "int64" => class_override = Some(OutputTemplate::Integer(IntegerTarget::I64)),
                    "uint8" => class_override = Some(OutputTemplate::Integer(IntegerTarget::U8)),
                    "uint16" => class_override = Some(OutputTemplate::Integer(IntegerTarget::U16)),
                    "uint32" => class_override = Some(OutputTemplate::Integer(IntegerTarget::U32)),
                    "uint64" => class_override = Some(OutputTemplate::Integer(IntegerTarget::U64)),
                    other => {
                        return Err(builtin_error(format!(
                            "randi: unrecognised option '{other}'"
                        )));
                    }
                }
                idx += 1;
                continue;
            }

            if let Some(parsed_dims) = extract_dims(&arg).await? {
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            if shape_source.is_none() {
                shape_source = Some(shape_from_value(&arg)?);
            }
            if implicit_proto.is_none() {
                implicit_proto = Some(arg.clone());
            }
            idx += 1;
        }

        let shape = if saw_dims_arg {
            if dims.is_empty() {
                vec![0, 0]
            } else if dims.len() == 1 {
                vec![dims[0], dims[0]]
            } else {
                dims
            }
        } else if let Some(shape) = shape_source {
            shape
        } else {
            vec![1, 1]
        };

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
        } else if let Some(proto) = implicit_proto {
            OutputTemplate::Like(proto)
        } else {
            OutputTemplate::Double
        };

        Ok(Self {
            bounds,
            shape,
            template,
        })
    }
}

async fn build_output(parsed: ParsedRandi) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => randi_double(&parsed.bounds, &parsed.shape),
        OutputTemplate::Single => randi_single(&parsed.bounds, &parsed.shape),
        OutputTemplate::Logical => randi_logical(&parsed.bounds, &parsed.shape),
        OutputTemplate::Integer(target) => randi_integer(&parsed.bounds, &parsed.shape, target),
        OutputTemplate::Like(proto) => randi_like(&proto, &parsed.bounds, &parsed.shape).await,
    }
}

fn randi_double(bounds: &Bounds, shape: &[usize]) -> crate::BuiltinResult<Value> {
    let tensor = integer_tensor(bounds, shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn randi_single(bounds: &Bounds, shape: &[usize]) -> crate::BuiltinResult<Value> {
    let data = generate_integer_values(bounds, tensor::element_count(shape))?
        .into_iter()
        .map(|value| value as f64)
        .collect();
    Tensor::new_with_dtype(data, shape.to_vec(), runmat_builtins::NumericDType::F32)
        .map(tensor::tensor_into_value)
        .map_err(|error| builtin_error(format!("randi: {error}")))
}

fn randi_integer(
    bounds: &Bounds,
    shape: &[usize],
    target: IntegerTarget,
) -> crate::BuiltinResult<Value> {
    validate_integer_bounds(bounds, target)?;
    let values = generate_integer_values(bounds, tensor::element_count(shape))?;
    let storage = integer_storage_from_values(target, values)?;
    Tensor::new_integer(storage, shape.to_vec())
        .map(tensor::tensor_into_value)
        .map_err(|error| builtin_error(format!("randi: {error}")))
}

fn randi_logical(bounds: &Bounds, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if bounds.lower < 0 || bounds.upper > 1 {
        return Err(builtin_error(
            "randi: logical output requires bounds contained within the inclusive range [0, 1]",
        ));
    }

    let len = tensor::element_count(shape);
    let mut data: Vec<u8> = Vec::with_capacity(len);
    if len == 0 {
        let logical = LogicalArray::new(data, shape.to_vec())
            .map_err(|e| builtin_error(format!("randi: {e}")))?;
        return Ok(Value::LogicalArray(logical));
    }

    if bounds.span == 1 {
        let byte = if bounds.lower == 0 { 0u8 } else { 1u8 };
        data.resize(len, byte);
    } else {
        let samples = generate_integer_values(bounds, len)?;
        data = samples
            .into_iter()
            .map(|value| if value != 0 { 1u8 } else { 0u8 })
            .collect();
    }

    let logical = LogicalArray::new(data, shape.to_vec())
        .map_err(|e| builtin_error(format!("randi: {e}")))?;
    Ok(Value::LogicalArray(logical))
}

#[async_recursion::async_recursion(?Send)]
async fn randi_like(
    proto: &Value,
    bounds: &Bounds,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    match proto {
        Value::GpuTensor(handle) => randi_like_gpu(handle, bounds, shape).await,
        Value::LogicalArray(_) | Value::Bool(_) => randi_logical(bounds, shape),
        Value::Tensor(tensor) => match tensor.integer_storage() {
            Some(storage) => randi_integer(bounds, shape, IntegerTarget::from_storage(storage)),
            None if tensor.dtype == runmat_builtins::NumericDType::F32 => {
                randi_single(bounds, shape)
            }
            None => randi_double(bounds, shape),
        },
        Value::Int(value) => randi_integer(bounds, shape, IntegerTarget::from_int_value(value)),
        Value::Num(_) => randi_double(bounds, shape),
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) => {
            randi_double(bounds, shape)
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(builtin_error(
            "randi: complex prototypes are not supported; expected real-valued arrays",
        )),
        Value::Cell(_) => Err(builtin_error("randi: cell prototypes are not supported")),
        other => Err(builtin_error(format!(
            "randi: unsupported prototype {other:?}"
        ))),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn randi_like_gpu(
    handle: &GpuTensorHandle,
    bounds: &Bounds,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Some(integer_type) = runmat_accelerate_api::handle_integer_type(handle) {
            let target = integer_target_from_accelerator_type(integer_type);
            validate_integer_bounds(bounds, target)?;
            if let Some(gpu) =
                try_provider_integer_randi(provider, handle, bounds, shape, target).await
            {
                return Ok(Value::GpuTensor(gpu));
            }
            let values = generate_integer_values(bounds, tensor::element_count(shape))?;
            let storage = integer_storage_from_values(target, values)?;
            let view = integer_tensor_view(&storage, shape);
            return provider
                .upload_integer(&view)
                .map(Value::GpuTensor)
                .map_err(|e| {
                    builtin_error(format!(
                        "randi: provider cannot preserve native integer gpuArray output: {e}"
                    ))
                });
        }

        let lower = i64::try_from(bounds.lower).map_err(|_| {
            builtin_error(
                "randi: GPU integer generation currently requires int64-representable bounds",
            )
        })?;
        let upper = i64::try_from(bounds.upper).map_err(|_| {
            builtin_error(
                "randi: GPU integer generation currently requires int64-representable bounds",
            )
        })?;
        let attempt = if handle.shape == shape {
            provider.random_integer_like(handle, lower, upper)
        } else {
            provider.random_integer_range(lower, upper, shape)
        };
        if let Ok(gpu) = attempt {
            return Ok(Value::GpuTensor(gpu));
        }

        let tensor = integer_tensor(bounds, shape)?;
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            return Ok(Value::GpuTensor(gpu));
        }
        return Ok(tensor::tensor_into_value(tensor));
    }

    let gathered = crate::dispatcher::gather_if_needed_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(|e| builtin_error(format!("randi: {e}")))?;
    randi_like(&gathered, bounds, shape).await
}

async fn try_provider_integer_randi(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    handle: &GpuTensorHandle,
    bounds: &Bounds,
    shape: &[usize],
    target: IntegerTarget,
) -> Option<GpuTensorHandle> {
    let lower = i64::try_from(bounds.lower).ok()?;
    let upper = i64::try_from(bounds.upper).ok()?;
    if target.uses_extended_scalar_precision()
        && (!provider_integer_bound_is_exact(lower) || !provider_integer_bound_is_exact(upper))
    {
        return None;
    }
    let generated = if handle.shape == shape {
        provider.random_integer_like(handle, lower, upper).ok()?
    } else {
        provider.random_integer_range(lower, upper, shape).ok()?
    };
    provider
        .cast_to_integer(&generated, target.accelerator_type())
        .await
        .ok()
}

fn provider_integer_bound_is_exact(value: i64) -> bool {
    const MAX_EXACT_INTEGER: i64 = 1_i64 << 53;
    (-MAX_EXACT_INTEGER..=MAX_EXACT_INTEGER).contains(&value)
}

fn integer_tensor(bounds: &Bounds, shape: &[usize]) -> crate::BuiltinResult<Tensor> {
    let len = tensor::element_count(shape);
    let data = generate_integer_values(bounds, len)?
        .into_iter()
        .map(|value| value as f64)
        .collect();
    Tensor::new(data, shape.to_vec()).map_err(|e| builtin_error(format!("randi: {e}")))
}

fn generate_integer_values(bounds: &Bounds, len: usize) -> crate::BuiltinResult<Vec<i128>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if bounds.span == 1 {
        return Ok(vec![bounds.lower; len]);
    }

    let uniforms = random::generate_uniform(len, "randi")?;
    let span = bounds.span as f64;
    let mut out = Vec::with_capacity(len);
    for u in uniforms {
        let mut offset = (u * span).floor() as u64;
        if offset >= bounds.span {
            offset = bounds.span - 1;
        }
        let mut value = bounds
            .lower
            .checked_add(offset as i128)
            .ok_or_else(|| builtin_error("randi: integer overflow while sampling"))?;
        if value > bounds.upper {
            value = bounds.upper;
        }
        out.push(value);
    }
    Ok(out)
}

fn validate_integer_bounds(bounds: &Bounds, target: IntegerTarget) -> crate::BuiltinResult<()> {
    let (min, max) = match target {
        IntegerTarget::I8 => (i8::MIN as i128, i8::MAX as i128),
        IntegerTarget::I16 => (i16::MIN as i128, i16::MAX as i128),
        IntegerTarget::I32 => (i32::MIN as i128, i32::MAX as i128),
        IntegerTarget::I64 => (i64::MIN as i128, i64::MAX as i128),
        IntegerTarget::U8 => (0, u8::MAX as i128),
        IntegerTarget::U16 => (0, u16::MAX as i128),
        IntegerTarget::U32 => (0, u32::MAX as i128),
        IntegerTarget::U64 => (0, u64::MAX as i128),
    };
    if bounds.lower < min || bounds.upper > max {
        return Err(builtin_error(
            "randi: bounds must be representable in the requested output class",
        ));
    }
    Ok(())
}

fn integer_storage_from_values(
    target: IntegerTarget,
    values: Vec<i128>,
) -> crate::BuiltinResult<IntegerStorage> {
    let storage = match target {
        IntegerTarget::I8 => IntegerStorage::I8(values.into_iter().map(|v| v as i8).collect()),
        IntegerTarget::I16 => IntegerStorage::I16(values.into_iter().map(|v| v as i16).collect()),
        IntegerTarget::I32 => IntegerStorage::I32(values.into_iter().map(|v| v as i32).collect()),
        IntegerTarget::I64 => IntegerStorage::I64(values.into_iter().map(|v| v as i64).collect()),
        IntegerTarget::U8 => IntegerStorage::U8(values.into_iter().map(|v| v as u8).collect()),
        IntegerTarget::U16 => IntegerStorage::U16(values.into_iter().map(|v| v as u16).collect()),
        IntegerTarget::U32 => IntegerStorage::U32(values.into_iter().map(|v| v as u32).collect()),
        IntegerTarget::U64 => IntegerStorage::U64(values.into_iter().map(|v| v as u64).collect()),
    };
    Ok(storage)
}

fn integer_target_from_accelerator_type(
    element_type: runmat_accelerate_api::IntegerElementType,
) -> IntegerTarget {
    match element_type {
        runmat_accelerate_api::IntegerElementType::I8 => IntegerTarget::I8,
        runmat_accelerate_api::IntegerElementType::I16 => IntegerTarget::I16,
        runmat_accelerate_api::IntegerElementType::I32 => IntegerTarget::I32,
        runmat_accelerate_api::IntegerElementType::I64 => IntegerTarget::I64,
        runmat_accelerate_api::IntegerElementType::U8 => IntegerTarget::U8,
        runmat_accelerate_api::IntegerElementType::U16 => IntegerTarget::U16,
        runmat_accelerate_api::IntegerElementType::U32 => IntegerTarget::U32,
        runmat_accelerate_api::IntegerElementType::U64 => IntegerTarget::U64,
    }
}

fn integer_tensor_view<'a>(
    storage: &'a IntegerStorage,
    shape: &'a [usize],
) -> HostIntegerTensorView<'a> {
    let data = match storage {
        IntegerStorage::I8(values) => HostIntegerDataView::I8(values),
        IntegerStorage::I16(values) => HostIntegerDataView::I16(values),
        IntegerStorage::I32(values) => HostIntegerDataView::I32(values),
        IntegerStorage::I64(values) => HostIntegerDataView::I64(values),
        IntegerStorage::U8(values) => HostIntegerDataView::U8(values),
        IntegerStorage::U16(values) => HostIntegerDataView::U16(values),
        IntegerStorage::U32(values) => HostIntegerDataView::U32(values),
        IntegerStorage::U64(values) => HostIntegerDataView::U64(values),
    };
    HostIntegerTensorView { data, shape }
}

async fn parse_bounds(value: Value) -> crate::BuiltinResult<Bounds> {
    let value = match value {
        Value::GpuTensor(_) => crate::dispatcher::gather_if_needed_async(&value)
            .await
            .map_err(|e| builtin_error(format!("randi: {e}")))?,
        other => other,
    };
    match value {
        Value::Int(value) => parse_upper_integer(int_value_to_i128(value)),
        Value::Tensor(t) => parse_bounds_tensor(&t),
        Value::LogicalArray(_) | Value::Bool(_) => Err(builtin_error(
            "randi: bounds must be numeric scalars or vectors",
        )),
        Value::String(s) => Err(builtin_error(format!(
            "randi: unexpected option '{s}' in first argument"
        ))),
        Value::StringArray(_) => Err(builtin_error(
            "randi: unexpected string array in first argument",
        )),
        Value::CharArray(_) => Err(builtin_error("randi: string bounds are not supported")),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(builtin_error("randi: complex bounds are not supported"))
        }
        other => {
            let Some(raw) = tensor::scalar_f64_from_value_async(&other)
                .await
                .map_err(|e| builtin_error(format!("randi: {e}")))?
            else {
                return Err(builtin_error(format!(
                    "randi: unsupported bounds argument {other:?}"
                )));
            };
            parse_upper_num(raw)
        }
    }
}

fn parse_upper_integer(upper: i128) -> crate::BuiltinResult<Bounds> {
    if upper < 1 {
        return Err(builtin_error("randi: upper bound must be >= 1"));
    }
    Bounds::new(1, upper)
}

fn parse_upper_num(n: f64) -> crate::BuiltinResult<Bounds> {
    if !n.is_finite() {
        return Err(builtin_error("randi: bounds must be finite"));
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err(builtin_error("randi: bounds must be integers"));
    }
    parse_upper_integer(rounded as i128)
}

fn parse_bounds_tensor(tensor: &Tensor) -> crate::BuiltinResult<Bounds> {
    if let Some(storage) = tensor.integer_storage() {
        let len = storage.len();
        if len == 0 {
            return Err(builtin_error("randi: empty bound vector is not allowed"));
        }
        if len == 1 {
            return parse_upper_integer(int_value_to_i128(
                storage
                    .value_at(0)
                    .expect("integer storage length matches tensor"),
            ));
        }
        if len == 2 && is_vector_like(tensor) {
            let lower = int_value_to_i128(
                storage
                    .value_at(0)
                    .expect("integer storage length matches tensor"),
            );
            let upper = int_value_to_i128(
                storage
                    .value_at(1)
                    .expect("integer storage length matches tensor"),
            );
            return Bounds::new(lower, upper);
        }
    }
    let len = tensor.data.len();
    if len == 0 {
        return Err(builtin_error("randi: empty bound vector is not allowed"));
    }
    if len == 1 {
        return parse_upper_num(tensor.data[0]);
    }
    if len == 2 && is_vector_like(tensor) {
        let lower = parse_integer_component(tensor.data[0])?;
        let upper = parse_integer_component(tensor.data[1])?;
        Bounds::new(lower, upper)
    } else {
        Err(builtin_error(
            "randi: bound vector must contain exactly two elements",
        ))
    }
}

fn parse_integer_component(value: f64) -> crate::BuiltinResult<i128> {
    if !value.is_finite() {
        return Err(builtin_error("randi: bounds must be finite"));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(builtin_error("randi: bounds must be integers"));
    }
    Ok(rounded as i128)
}

fn int_value_to_i128(value: IntValue) -> i128 {
    match value {
        IntValue::I8(value) => value as i128,
        IntValue::I16(value) => value as i128,
        IntValue::I32(value) => value as i128,
        IntValue::I64(value) => value as i128,
        IntValue::U8(value) => value as i128,
        IntValue::U16(value) => value as i128,
        IntValue::U32(value) => value as i128,
        IntValue::U64(value) => value as i128,
    }
}

fn is_vector_like(tensor: &Tensor) -> bool {
    tensor.rows() == 1 || tensor.cols() == 1 || tensor.shape.len() == 1
}

fn keyword_of(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            Some(text.to_ascii_lowercase())
        }
        _ => None,
    }
}

async fn extract_dims(value: &Value) -> crate::BuiltinResult<Option<Vec<usize>>> {
    if matches!(value, Value::LogicalArray(_)) {
        return Ok(None);
    }
    let gpu_scalar = match value {
        Value::GpuTensor(handle) => tensor::element_count(&handle.shape) == 1,
        _ => false,
    };
    match tensor::dims_from_value_async(value).await {
        Ok(dims) => Ok(dims),
        Err(err) => {
            if matches!(value, Value::Tensor(_))
                || (matches!(value, Value::GpuTensor(_)) && !gpu_scalar)
            {
                Ok(None)
            } else {
                Err(builtin_error(format!("randi: {err}")))
            }
        }
    }
}

fn shape_from_value(value: &Value) -> crate::BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => Ok(t.shape.clone()),
        Value::ComplexTensor(_) => {
            Err(builtin_error("randi: complex prototypes are not supported"))
        }
        Value::LogicalArray(l) => Ok(l.shape.clone()),
        Value::GpuTensor(h) => Ok(h.shape.clone()),
        Value::CharArray(ca) => Ok(vec![ca.rows, ca.cols]),
        Value::Cell(cell) => Ok(vec![cell.rows, cell.cols]),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok(vec![1, 1]),
        other => Err(builtin_error(format!(
            "randi: unsupported prototype {other:?}"
        ))),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{random, test_support};
    use futures::executor::block_on;
    use runmat_builtins::LogicalArray;

    fn reset_rng_clean() {
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
    }

    fn expected_sequence(bounds: &Bounds, count: usize) -> Vec<i128> {
        let uniforms = random::expected_uniform_sequence(count);
        let span = bounds.span as f64;
        uniforms
            .into_iter()
            .map(|u| {
                let mut offset = (u * span).floor() as u64;
                if offset >= bounds.span {
                    offset = bounds.span - 1;
                }
                bounds.lower + offset as i128
            })
            .collect()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_default_scalar() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let result = block_on(randi_builtin(vec![Value::Num(6.0)])).expect("randi");
        let expected = expected_sequence(&Bounds::new(1, 6).unwrap(), 1)[0] as f64;
        match result {
            Value::Num(v) => {
                assert!((1.0..=6.0).contains(&v));
                assert!((v - expected).abs() < 1e-12);
            }
            other => panic!("expected scalar double, got {other:?}"),
        }
    }

    #[test]
    fn randi_type_single_bound_is_num() {
        assert_eq!(
            randi_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Num
        );
    }

    #[test]
    fn randi_type_infers_rank_from_dims() {
        let ctx = ResolveContext::new(Vec::new());
        assert_eq!(
            randi_type(&[Type::Num, Type::Num, Type::Num], &ctx),
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_range_with_dims() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let bounds = Tensor::new(vec![3.0, 8.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(bounds), Value::Num(2.0), Value::Num(3.0)];
        let result = block_on(randi_builtin(args)).expect("randi");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3]);
                let expected = expected_sequence(&Bounds::new(3, 8).unwrap(), 6);
                for (observed, exp) in t.data.iter().zip(expected.iter().map(|v| *v as f64)) {
                    assert!((*observed - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_like_tensor() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let proto = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let args = vec![Value::Num(5.0), Value::from("like"), Value::Tensor(proto)];
        let result = block_on(randi_builtin(args)).expect("randi");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                for v in &t.data {
                    assert!((1.0..=5.0).contains(v));
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn randi_preserves_every_explicit_integer_class_and_integer_like_prototype() {
        let _guard = random::test_lock().lock().unwrap();
        let cases = [
            ("int8", IntegerTarget::I8),
            ("int16", IntegerTarget::I16),
            ("int32", IntegerTarget::I32),
            ("int64", IntegerTarget::I64),
            ("uint8", IntegerTarget::U8),
            ("uint16", IntegerTarget::U16),
            ("uint32", IntegerTarget::U32),
            ("uint64", IntegerTarget::U64),
        ];

        for (class, target) in cases {
            reset_rng_clean();
            let result = block_on(randi_builtin(vec![
                Value::Num(5.0),
                Value::Num(2.0),
                Value::Num(3.0),
                Value::from(class),
            ]))
            .expect("explicit integer randi");
            let Value::Tensor(tensor) = result else {
                panic!("expected integer tensor");
            };
            assert_eq!(tensor.shape, vec![2, 3]);
            let storage = tensor.integer_storage().expect("exact integer storage");
            assert!(IntegerTarget::from_storage(storage) == target);
            assert!(storage
                .exact_values()
                .iter()
                .all(|value| (1..=5).contains(&value.to_i64())));
        }

        let prototype = Tensor::new_integer(IntegerStorage::U64(vec![0; 4]), vec![2, 2])
            .expect("uint64 prototype");
        let result = block_on(randi_builtin(vec![
            Value::Num(5.0),
            Value::from("like"),
            Value::Tensor(prototype),
        ]))
        .expect("integer like randi");
        let Value::Tensor(tensor) = result else {
            panic!("expected integer tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert!(matches!(
            tensor.integer_storage(),
            Some(IntegerStorage::U64(_))
        ));

        let result = block_on(randi_builtin(vec![
            Value::Num(5.0),
            Value::Num(2.0),
            Value::from("single"),
        ]))
        .expect("single randi");
        let Value::Tensor(tensor) = result else {
            panic!("expected single tensor");
        };
        assert_eq!(tensor.dtype, runmat_builtins::NumericDType::F32);
        assert!(tensor.integer_storage().is_none());
    }

    #[test]
    fn randi_preserves_exact_high_uint64_bounds_and_rejects_out_of_class_bounds() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let lower = u64::MAX - 4;
        let mut bounds =
            Tensor::new_integer(IntegerStorage::U64(vec![lower, u64::MAX]), vec![1, 2])
                .expect("uint64 bounds");
        bounds.data.clear();
        let result = block_on(randi_builtin(vec![
            Value::Tensor(bounds),
            Value::Num(1.0),
            Value::Num(8.0),
            Value::from("uint64"),
        ]))
        .expect("high uint64 randi");
        let Value::Tensor(tensor) = result else {
            panic!("expected uint64 tensor");
        };
        let IntegerStorage::U64(values) = tensor.integer_storage().expect("uint64 storage") else {
            panic!("expected uint64 storage");
        };
        assert!(values
            .iter()
            .all(|&value| (lower..=u64::MAX).contains(&value)));

        let err = block_on(randi_builtin(vec![Value::Num(300.0), Value::from("uint8")]))
            .expect_err("out-of-class bounds must fail");
        assert!(err
            .to_string()
            .contains("representable in the requested output class"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_logical_output() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let bounds = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Tensor(bounds),
            Value::Num(2.0),
            Value::Num(2.0),
            Value::from("logical"),
        ];
        let result = block_on(randi_builtin(args)).expect("randi logical");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 2]);
                let expected = expected_sequence(&Bounds::new(0, 1).unwrap(), 4);
                for (idx, &byte) in logical.data.iter().enumerate() {
                    assert!(byte <= 1);
                    assert_eq!(byte, if expected[idx] == 0 { 0 } else { 1 });
                }
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_logical_requires_binary_bounds() {
        let err =
            block_on(randi_builtin(vec![Value::Num(3.0), Value::from("logical")])).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("logical output requires"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_like_logical_prototype() {
        let _guard = random::test_lock().lock().unwrap();
        reset_rng_clean();
        let proto = LogicalArray::zeros(vec![2, 3]);
        let bounds = Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Tensor(bounds),
            Value::from("like"),
            Value::LogicalArray(proto),
        ];
        let result = block_on(randi_builtin(args)).expect("randi logical like");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 3]);
                assert!(logical.data.iter().all(|&b| b <= 1));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_like_requires_prototype() {
        let err = block_on(randi_builtin(vec![Value::Num(5.0), Value::from("like")])).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("expected prototype"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_duplicate_like_is_error() {
        let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let args = vec![
            Value::Num(5.0),
            Value::from("like"),
            Value::Tensor(proto.clone()),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        let err = block_on(randi_builtin(args)).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("multiple 'like' specifications"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_like_logical_conflict_is_error() {
        let proto = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let args = vec![
            Value::Num(1.0),
            Value::from("logical"),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        let err = block_on(randi_builtin(args)).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("cannot combine 'like' with 'logical'"));
    }

    #[test]
    fn randi_like_integer_class_conflict_is_error() {
        let proto = Tensor::new_integer(IntegerStorage::I64(vec![0]), vec![1, 1])
            .expect("integer prototype");
        let err = block_on(randi_builtin(vec![
            Value::Num(5.0),
            Value::from("like"),
            Value::Tensor(proto),
            Value::from("int64"),
        ]))
        .expect_err("like and integer class must conflict");
        assert!(err
            .to_string()
            .contains("cannot combine 'like' with 'int64'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_gpu_like_roundtrip() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(4.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(randi_builtin(args)).expect("randi");
            match result {
                Value::GpuTensor(gpu) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(gpu)).expect("gather to host");
                    assert_eq!(gathered.shape, vec![2, 2]);
                    for value in gathered.data {
                        assert!((1.0..=4.0).contains(&value));
                    }
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_gpu_like_shape_override() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let proto = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &proto.data,
                shape: &proto.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let bounds = Tensor::new(vec![1.0, 4.0], vec![1, 2]).unwrap();
            let args = vec![
                Value::Tensor(bounds),
                Value::Num(3.0),
                Value::Num(1.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(randi_builtin(args)).expect("randi gpu override");
            match result {
                Value::GpuTensor(gpu) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(gpu)).expect("gather override");
                    assert_eq!(gathered.shape, vec![3, 1]);
                    for value in gathered.data {
                        assert!((1.0..=4.0).contains(&value));
                    }
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_gpu_like_integer_prototype_uses_resident_provider_path() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let prototype_values = [-3_i16, 7_i16];
            let prototype = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::I16(&prototype_values),
                    shape: &[1, 2],
                })
                .expect("upload int16 prototype");

            let result = block_on(randi_builtin(vec![
                Value::Num(9.0),
                Value::Num(3.0),
                Value::from("like"),
                Value::GpuTensor(prototype),
            ]))
            .expect("randi int16 gpu like");

            let Value::GpuTensor(gpu) = result else {
                panic!("expected resident gpuArray result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&gpu),
                Some(runmat_accelerate_api::IntegerElementType::I16)
            );
            assert_eq!(gpu.shape, vec![3, 3]);
            let downloaded = block_on(provider.download_integer(&gpu))
                .expect("download int16 randi")
                .data;
            let runmat_accelerate_api::HostIntegerDataOwned::I16(values) = downloaded else {
                panic!("expected int16 storage");
            };
            assert_eq!(values.len(), 9);
            assert!(values.iter().all(|value| (1..=9).contains(value)));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_gpu_like_preserves_native_uint64_storage() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let prototype_values = [1_u64 << 53, u64::MAX];
            let prototype = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&prototype_values),
                    shape: &[1, 2],
                })
                .expect("upload uint64 prototype");
            let lower = (1_u64 << 53) + 1;
            let upper = lower + 3;
            let bounds = Tensor::new_integer(IntegerStorage::U64(vec![lower, upper]), vec![1, 2])
                .expect("uint64 bounds");

            let result = block_on(randi_builtin(vec![
                Value::Tensor(bounds),
                Value::from("like"),
                Value::GpuTensor(prototype),
            ]))
            .expect("randi uint64 gpu like");

            let Value::GpuTensor(gpu) = result else {
                panic!("expected resident gpuArray result");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&gpu),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            assert_eq!(gpu.shape, vec![1, 2]);
            let downloaded = block_on(provider.download_integer(&gpu))
                .expect("download uint64 randi")
                .data;
            let runmat_accelerate_api::HostIntegerDataOwned::U64(values) = downloaded else {
                panic!("expected uint64 storage");
            };
            assert_eq!(values.len(), 2);
            assert!(values.iter().all(|value| (lower..=upper).contains(value)));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn randi_invalid_upper_errors() {
        let err = block_on(randi_builtin(vec![Value::Num(0.0)])).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("upper bound"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn randi_wgpu_like_produces_in_range_values() {
        let _guard = random::test_lock().lock().unwrap();
        random::reset_rng();
        let provider = match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(_) => runmat_accelerate_api::provider().expect("wgpu provider registered"),
            Err(err) => {
                tracing::warn!("randi_wgpu_like_produces_in_range_values skipped: {err}");
                return;
            }
        };

        let proto = Tensor::new(vec![0.0; 6], vec![2, 3]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &proto.data,
            shape: &proto.shape,
        };
        let handle = provider.upload(&view).expect("upload prototype");
        let bounds = Tensor::new(vec![1.0, 8.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Tensor(bounds),
            Value::from("like"),
            Value::GpuTensor(handle),
        ];

        let result = block_on(randi_builtin(args)).expect("randi");
        match result {
            Value::GpuTensor(gpu) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(gpu)).expect("gather gpu result");
                assert_eq!(gathered.shape, vec![2, 3]);
                for value in gathered.data {
                    assert!(
                        (1.0..=8.0).contains(&value),
                        "expected value within [1, 8], got {value}"
                    );
                }
            }
            other => panic!("expected GPU tensor result, got {other:?}"),
        }
    }
}
