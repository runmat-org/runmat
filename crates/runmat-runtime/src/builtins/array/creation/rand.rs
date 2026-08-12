//! MATLAB-compatible `rand` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{GpuTensorHandle, HostTensorView, ProviderPrecision};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, IntValue, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;
use std::sync::OnceLock;

use crate::build_runtime_error;
use crate::builtins::array::type_resolvers::tensor_type_from_rank;
use crate::builtins::common::random_args::{
    complex_tensor_into_value, extract_constructor_dimensions, keyword_of,
    normalize_constructor_shape, validate_constructor_gpu_output,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, random, tensor};
use runmat_builtins::ResolveContext;
use runmat_builtins::Type;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::rand")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "rand",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("random_uniform"),
        ProviderHook::Custom("random_uniform_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Delegates to provider random_uniform hooks; falls back to host sampling + upload when hooks are unavailable.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("rand").build()
}

fn rand_type(args: &[Type], ctx: &ResolveContext) -> Type {
    if args.is_empty() {
        return Type::Num;
    }
    if args.iter().any(|arg| matches!(arg, Type::String)) {
        return Type::Unknown;
    }
    tensor_type_from_rank(args, ctx)
}

const RAND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Uniform random array in (0,1).",
}];

const RAND_SEED_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "seed",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legacy restorable RNG state token.",
}];

const RAND_NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];

const RAND_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const RAND_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const RAND_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const RAND_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const RAND_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Class override ('double'|'single'|'gpuArray').",
    },
];

const RAND_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
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

const RAND_SIG_SEED_QUERY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "seed_option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"seed\""),
    description: "Legacy seed query option.",
}];

const RAND_SIG_SEED_SET_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "seed_option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"seed\""),
        description: "Legacy seed control option.",
    },
    BuiltinParamDescriptor {
        name: "seed",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Non-negative integer seed.",
    },
];

const RAND_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "A = rand()",
        inputs: &RAND_SIG_EMPTY_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = rand(n)",
        inputs: &RAND_SIG_N_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = rand(size_vector)",
        inputs: &RAND_SIG_SIZE_VECTOR_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = rand(m, n, ...)",
        inputs: &RAND_SIG_DIMS_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = rand(..., typename)",
        inputs: &RAND_SIG_CLASS_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = rand(..., \"like\", prototype)",
        inputs: &RAND_SIG_LIKE_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "seed = rand(\"seed\")",
        inputs: &RAND_SIG_SEED_QUERY_INPUTS,
        outputs: &RAND_SEED_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "rand(\"seed\", seed)",
        inputs: &RAND_SIG_SEED_SET_INPUTS,
        outputs: &RAND_NO_OUTPUTS,
    },
];

const RAND_COLUMN_SIZE_VECTOR_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rand-column-size-vector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rand with a column size vector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandColumnSizeVectorExtension"),
};
const RAND_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rand-resident-size-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rand with a resident size control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandResidentSizeControlExtension"),
};
pub const RAND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    RAND_COLUMN_SIZE_VECTOR_EXTENSION,
    RAND_RESIDENT_SIZE_EXTENSION,
];
const RAND_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz1...szN/sz",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are exact structural size controls; negative signed values clamp to zero and trailing singleton dimensions normalize away.",
    }];
const RAND_INTEGER_SEED_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "seed",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The discouraged legacy seed syntax accepts a nonnegative exact integer scalar within the runtime's restorable seed-token domain.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "X = rand(integer_n[, integer_sz2, ...])",
        inputs: &RAND_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The default output is host double; typename can select single and explicit gpuArray syntax selects residency.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = rand(integer_sz)",
        inputs: &RAND_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented size vector is a row vector of exact integer values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "rand('seed', integer_seed)",
        inputs: &RAND_INTEGER_SEED_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "[integer-audit-open] This legacy control form updates the shared RunMat random stream and synchronizes the active provider when supported, but exact MATLAB stream/state equivalence is not yet established.",
    },
];

const RAND_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.RAND.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "rand: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.RAND.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not supported.",
        message: "rand: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.RAND.UNSUPPORTED_PROTOTYPE",
        identifier: None,
        when: "A prototype type cannot be used for rand(..., 'like', prototype).",
        message: "rand: unsupported prototype",
    },
    BuiltinErrorDescriptor {
        code: "RM.RAND.INVALID_DIMS",
        identifier: None,
        when: "Dimension arguments fail numeric/shape parsing.",
        message: "rand: dimension arguments must be numeric and nonnegative",
    },
];

pub const RAND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RAND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RAND_ERRORS,
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::rand")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "rand",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Random generation is treated as a sink and is not eligible for fusion.",
};

#[runtime_builtin(
    name = "rand",
    category = "array/creation",
    summary = "Generate uniformly distributed pseudorandom numbers on the open interval (0, 1).",
    keywords = "rand,random,uniform,gpu,like",
    accel = "array_construct",
    type_resolver(rand_type),
    descriptor(crate::builtins::array::creation::rand::RAND_DESCRIPTOR),
    extensions(RAND_EXTENSIONS),
    integer_capabilities(crate::builtins::array::creation::rand::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::creation::rand"
)]
async fn rand_builtin(rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if let Some(command) = LegacyRandCommand::parse(&rest)? {
        return command.apply();
    }
    let parsed = ParsedRand::parse(rest).await?;
    build_output(parsed).await
}

enum LegacyRandCommand {
    QuerySeed,
    SetSeed(u64),
}

impl LegacyRandCommand {
    fn parse(args: &[Value]) -> crate::BuiltinResult<Option<Self>> {
        let Some(first) = args.first() else {
            return Ok(None);
        };
        if keyword_of(first).as_deref() != Some("seed") {
            return Ok(None);
        }
        match args.len() {
            1 => Ok(Some(Self::QuerySeed)),
            2 => Ok(Some(Self::SetSeed(parse_legacy_seed(&args[1])?))),
            _ => Err(builtin_error(
                "rand: legacy seed option expects zero or one seed argument",
            )),
        }
    }

    fn apply(self) -> crate::BuiltinResult<Value> {
        match self {
            Self::QuerySeed => {
                let seed = random::legacy_seed_value()?;
                Ok(Value::Num(seed as f64))
            }
            Self::SetSeed(seed) => {
                random::set_legacy_seed(seed)?;
                let snapshot = random::snapshot()?;
                sync_provider_rng_state(snapshot.state);
                Tensor::new(Vec::new(), vec![0, 0])
                    .map(Value::Tensor)
                    .map_err(|e| builtin_error(format!("rand: {e}")))
            }
        }
    }
}

fn parse_legacy_seed(value: &Value) -> crate::BuiltinResult<u64> {
    match value {
        Value::Int(value) => match legacy_seed_from_int(value)? {
            Some(value) => Ok(value),
            None => Err(builtin_error("rand: seed must be non-negative")),
        },
        Value::Num(value) => {
            if !value.is_finite() {
                return Err(builtin_error("rand: seed must be finite"));
            }
            if *value < 0.0 {
                return Err(builtin_error("rand: seed must be non-negative"));
            }
            let rounded = value.round();
            if (rounded - value).abs() > f64::EPSILON {
                return Err(builtin_error("rand: seed must be an integer"));
            }
            if rounded > (1_u64 << 53) as f64 {
                return Err(builtin_error("rand: seed exceeds 53-bit integer precision"));
            }
            Ok(rounded as u64)
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                return parse_legacy_seed(&Value::Int(
                    storage
                        .value_at(0)
                        .expect("scalar integer storage has one element"),
                ));
            }
            parse_legacy_seed(&Value::Num(tensor::tensor_value_f64(tensor, 0)))
        }
        _ => Err(builtin_error("rand: seed must be a scalar numeric value")),
    }
}

fn legacy_seed_from_int(value: &IntValue) -> crate::BuiltinResult<Option<u64>> {
    let parsed = match value {
        IntValue::I8(value) => (*value >= 0).then_some(*value as u64),
        IntValue::I16(value) => (*value >= 0).then_some(*value as u64),
        IntValue::I32(value) => (*value >= 0).then_some(*value as u64),
        IntValue::I64(value) => (*value >= 0).then_some(*value as u64),
        IntValue::U8(value) => Some(*value as u64),
        IntValue::U16(value) => Some(*value as u64),
        IntValue::U32(value) => Some(*value as u64),
        IntValue::U64(value) => Some(*value),
    };
    let Some(parsed) = parsed else {
        return Ok(None);
    };
    if parsed > (1_u64 << 53) {
        return Err(builtin_error("rand: seed exceeds 53-bit integer precision"));
    }
    Ok(Some(parsed))
}

fn sync_provider_rng_state(state: u64) {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Err(err) = provider.set_rng_state(state) {
            log::debug!("rand: provider seed sync failed: {err}");
        }
    }
}

struct ParsedRand {
    shape: Vec<usize>,
    template: RandTemplate,
}

#[derive(Clone)]
enum RandTemplate {
    Double,
    Single,
    GpuArray(NumericDType),
    Like(Value),
}

impl ParsedRand {
    async fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut template: Option<RandTemplate> = None;
        let mut saw_size_vector = false;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();

            if let Some(keyword) = keyword_of(&arg) {
                if matches!(template.as_ref(), Some(RandTemplate::GpuArray(_))) {
                    return Err(builtin_error("rand: invalid gpuArray class specification"));
                }
                match keyword.as_str() {
                    "like" => {
                        if template.is_some() {
                            return Err(builtin_error("rand: conflicting class specifications"));
                        }
                        let Some(proto) = args.get(idx + 1).cloned() else {
                            return Err(builtin_error("rand: expected prototype after 'like'"));
                        };
                        template = Some(RandTemplate::Like(proto));
                        idx += 2;
                        continue;
                    }
                    "double" => {
                        if template.is_some() {
                            return Err(builtin_error("rand: conflicting class specifications"));
                        }
                        template = Some(RandTemplate::Double);
                        idx += 1;
                        continue;
                    }
                    "single" => {
                        if template.is_some() {
                            return Err(builtin_error("rand: conflicting class specifications"));
                        }
                        template = Some(RandTemplate::Single);
                        idx += 1;
                        continue;
                    }
                    "gpuarray" => {
                        // MATLAB class-specification syntax: rand(m,n,"gpuArray") or
                        // gpuArray.rand(m,n). Produce a GPU-resident double-precision
                        // array; rand_double already prefers the GPU provider when one
                        // is registered and falls back to host when it is not.
                        let dtype = match template.take() {
                            Some(RandTemplate::Single) => NumericDType::F32,
                            Some(RandTemplate::Double) | None => NumericDType::F64,
                            Some(RandTemplate::Like(_)) | Some(RandTemplate::GpuArray(_)) => {
                                return Err(builtin_error(
                                    "rand: invalid gpuArray class specification",
                                ));
                            }
                        };
                        template = Some(RandTemplate::GpuArray(dtype));
                        idx += 1;
                        continue;
                    }
                    other => {
                        return Err(builtin_error(format!(
                            "rand: unrecognised option '{other}'"
                        )));
                    }
                }
            }

            if matches!(arg, Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &RAND_RESIDENT_SIZE_EXTENSION,
                    "rand",
                )?;
            }
            if let Some(parsed_dims) = extract_constructor_dimensions(&arg, "rand")
                .await
                .map_err(builtin_error)?
            {
                if parsed_dims.is_column_vector {
                    crate::compatibility::ensure_builtin_extension_enabled(
                        &RAND_COLUMN_SIZE_VECTOR_EXTENSION,
                        "rand",
                    )?;
                }
                if parsed_dims.values.len() > 1 {
                    if saw_size_vector || saw_dims_arg {
                        return Err(builtin_error(
                            "rand: a size vector must be the only dimension argument",
                        ));
                    }
                    saw_size_vector = true;
                } else if saw_size_vector {
                    return Err(builtin_error(
                        "rand: a size vector must be the only dimension argument",
                    ));
                }
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims.values;
                } else {
                    dims.extend(parsed_dims.values);
                }
                idx += 1;
                continue;
            }

            return Err(builtin_error(format!(
                "rand: unsupported dimension or option {arg:?}"
            )));
        }

        let shape = if saw_dims_arg {
            normalize_constructor_shape(dims)
        } else {
            vec![1, 1]
        };

        let template = template.unwrap_or(RandTemplate::Double);

        Ok(Self { shape, template })
    }
}

async fn build_output(parsed: ParsedRand) -> crate::BuiltinResult<Value> {
    match parsed.template {
        RandTemplate::Double => rand_double(&parsed.shape),
        RandTemplate::Single => rand_single(&parsed.shape),
        RandTemplate::GpuArray(dtype) => rand_gpu(&parsed.shape, dtype),
        RandTemplate::Like(proto) => rand_like(&proto, &parsed.shape).await,
    }
}

fn rand_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let data = random::generate_uniform(len, "rand")?;
    let tensor =
        Tensor::new(data, shape.to_vec()).map_err(|e| builtin_error(format!("rand: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[async_recursion::async_recursion(?Send)]
async fn rand_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::GpuTensor(handle) => rand_like_gpu(handle, shape).await,
        Value::ComplexTensor(tensor) => {
            crate::builtins::common::validation::reject_typed_complex_integer_tensor(
                tensor, "rand",
            )?;
            rand_complex(shape, tensor.numeric_dtype())
        }
        Value::Complex(_, _) => rand_complex(shape, NumericDType::F64),
        Value::Tensor(tensor) => match tensor.numeric_dtype() {
            NumericDType::F32 => rand_single(shape),
            NumericDType::F64 => rand_double(shape),
            _ => Err(builtin_error(
                "rand: 'like' prototype must be single or double",
            )),
        },
        Value::Num(_) => rand_double(shape),
        Value::Int(_)
        | Value::Bool(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::Cell(_) => Err(builtin_error(
            "rand: 'like' prototype must be single or double",
        )),
        other => Err(builtin_error(format!(
            "rand: unsupported prototype {other:?}"
        ))),
    }
}

fn rand_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let data = random::generate_uniform_single(len, "rand")?;
    let tensor = Tensor::new_with_dtype(data, shape.to_vec(), NumericDType::F32)
        .map_err(|e| builtin_error(format!("rand: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn rand_gpu(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Value> {
    let Some(value) = try_gpu_uniform(shape, dtype)? else {
        return Err(builtin_error(
            "rand: gpuArray output requires a provider with the requested precision",
        ));
    };
    Ok(value)
}

fn rand_complex(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let data = random::generate_complex(len, "rand")?;
    let tensor = ComplexTensor::from_f64_values_with_dtype(data, shape.to_vec(), dtype)
        .map_err(|e| builtin_error(format!("rand: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

#[async_recursion::async_recursion(?Send)]
async fn rand_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    if runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        return Err(builtin_error(
            "rand: 'like' prototype must have single or double underlying type",
        ));
    }
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) {
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or_else(|| provider.precision());
        let dtype = dtype_from_precision(precision);
        if runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        {
            let host = random::generate_complex(tensor::element_count(shape), "rand")?;
            let tensor = ComplexTensor::from_f64_values_with_dtype(host, shape.to_vec(), dtype)
                .map_err(|error| builtin_error(format!("rand: {error}")))?;
            if let Ok(gpu) = gpu_helpers::upload_complex_tensor(provider, &tensor) {
                if let Ok(gpu) = validate_constructor_gpu_output(
                    "rand",
                    provider,
                    gpu,
                    shape,
                    runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                    Some(precision),
                    None,
                    false,
                ) {
                    return Ok(Value::GpuTensor(gpu));
                }
            }
            return Err(builtin_error(
                "rand: provider cannot preserve explicit complex gpuArray output",
            ));
        }
        let attempt = if handle.shape == shape {
            provider.random_uniform_like(handle)
        } else {
            provider.random_uniform(shape)
        };
        if let Ok(gpu) = attempt {
            if let Ok(gpu) = validate_constructor_gpu_output(
                "rand",
                provider,
                gpu,
                shape,
                runmat_accelerate_api::GpuTensorStorage::Real,
                Some(precision),
                None,
                false,
            ) {
                let len = tensor::element_count(shape);
                random::skip_uniform(len, "rand")?;
                return Ok(Value::GpuTensor(gpu));
            }
            log_rand_fallback(shape, dtype, "invalid-provider-like-result");
        } else {
            log_rand_fallback(shape, dtype, "provider-like-error");
        }

        let len = tensor::element_count(shape);
        let data = random::generate_uniform(len, "rand")?;

        let tensor =
            Tensor::new(data, shape.to_vec()).map_err(|e| builtin_error(format!("rand: {e}")))?;
        let view = HostTensorView {
            data: tensor
                .as_f64_slice()
                .expect("rand fallback constructs double storage"),
            shape: &tensor.shape,
        };
        if let Ok(gpu) = provider.upload(&view) {
            if let Ok(gpu) = validate_constructor_gpu_output(
                "rand",
                provider,
                gpu,
                shape,
                runmat_accelerate_api::GpuTensorStorage::Real,
                Some(precision),
                None,
                false,
            ) {
                return Ok(Value::GpuTensor(gpu));
            }
            log_rand_fallback(shape, dtype, "invalid-upload-result");
        } else {
            log_rand_fallback(shape, dtype, "upload-error");
        }
    } else {
        log_rand_fallback(shape, NumericDType::F32, "no-provider-like");
    }

    Err(builtin_error(
        "rand: provider cannot preserve explicit gpuArray output",
    ))
}

fn try_gpu_uniform(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        log_rand_fallback(shape, dtype, "no-provider");
        return Ok(None);
    };
    let precision = match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
        NumericDType::I8
        | NumericDType::I16
        | NumericDType::I32
        | NumericDType::I64
        | NumericDType::U8
        | NumericDType::U16
        | NumericDType::U32
        | NumericDType::U64 => {
            log_rand_fallback(shape, dtype, "integer-dtype");
            return Ok(None);
        }
    };
    if provider.precision() != precision {
        log_rand_fallback(shape, dtype, "precision-mismatch");
        return Ok(None);
    }
    match provider.random_uniform(shape) {
        Ok(handle) => {
            let Ok(handle) = validate_constructor_gpu_output(
                "rand",
                provider,
                handle,
                shape,
                runmat_accelerate_api::GpuTensorStorage::Real,
                Some(precision),
                None,
                false,
            ) else {
                return Ok(None);
            };
            let len = tensor::element_count(shape);
            random::skip_uniform(len, "rand")?;
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(err) => {
            log::warn!(
                "rand: provider random_uniform failed ({err}); falling back to host tensor path"
            );
            log_rand_fallback(shape, dtype, "provider-error");
            Ok(None)
        }
    }
}

fn rand_fallback_debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("RUNMAT_DEBUG_RAND_FALLBACK"),
            Ok(value) if value == "1"
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
        )
    })
}

fn log_rand_fallback(shape: &[usize], dtype: NumericDType, reason: &str) {
    if !rand_fallback_debug_enabled() {
        return;
    }
    let elems = tensor::element_count(shape);
    tracing::debug!(
        dtype = ?dtype,
        elems,
        shape = ?shape,
        reason,
        "[rand_debug] fallback"
    );
}

fn dtype_from_precision(precision: ProviderPrecision) -> NumericDType {
    match precision {
        ProviderPrecision::F32 => NumericDType::F32,
        ProviderPrecision::F64 => NumericDType::F64,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{random, test_support};
    use crate::dispatcher::download_handle_async;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerComplexStorage, IntegerStorage};

    fn reset_rng_clean() -> impl Drop {
        let guard = random::test_guard();
        runmat_accelerate_api::clear_provider();
        random::reset_rng();
        guard
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_default_scalar() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let result = block_on(rand_builtin(Vec::new())).expect("rand");
        let expected = random::expected_uniform_sequence(1)[0];
        match result {
            Value::Num(v) => {
                assert!((0.0..1.0).contains(&v));
                assert!((v - expected).abs() < 1e-12);
            }
            other => panic!("expected scalar double, got {other:?}"),
        }
    }

    #[test]
    fn rand_type_defaults_to_num() {
        assert_eq!(rand_type(&[], &ResolveContext::new(Vec::new())), Type::Num);
    }

    #[test]
    fn rand_type_infers_rank_from_scalar_dim() {
        assert_eq!(
            rand_type(&[Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![None, None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_square_from_single_dimension() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let args = vec![Value::Num(3.0)];
        let result = block_on(rand_builtin(args)).expect("rand");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 3]);
                let expected = random::expected_uniform_sequence(9);
                assert_eq!(t.materialize_f64().len(), expected.len());
                for (observed, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((*observed - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn rand_integer_dimensions_clamp_negative_and_normalize_trailing_singletons() {
        let _guard = reset_rng_clean();
        let result = block_on(rand_builtin(vec![
            Value::Int(IntValue::I32(-2)),
            Value::Int(IntValue::U8(3)),
            Value::Int(IntValue::U64(1)),
        ]))
        .expect("rand integer dimensions");
        let Value::Tensor(tensor) = result else {
            panic!("expected empty host tensor");
        };
        assert_eq!(tensor.shape, vec![0, 3]);
    }

    #[test]
    fn rand_column_size_vector_follows_compatibility_mode() {
        let _guard = reset_rng_clean();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let size = Tensor::new_integer(IntegerStorage::I16(vec![2, 3]), vec![2, 1])
            .expect("column size vector");
        let error = block_on(rand_builtin(vec![Value::Tensor(size)])).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:RandColumnSizeVectorExtension")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_legacy_seed_string_resets_sequence_and_queries_seed() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let seed_result = block_on(rand_builtin(vec![Value::from("seed"), Value::Num(2026.0)]))
            .expect("rand seed");
        assert!(matches!(seed_result, Value::Tensor(t) if t.shape == vec![0, 0]));
        let query = block_on(rand_builtin(vec![Value::from("seed")])).expect("rand seed query");
        assert!(matches!(query, Value::Num(seed) if (seed - 2026.0).abs() < f64::EPSILON));

        let prefix = block_on(rand_builtin(vec![Value::Num(1.0), Value::Num(4.0)])).expect("rand");
        let saved = block_on(rand_builtin(vec![Value::from("seed")])).expect("rand seed query");
        assert!(matches!(&saved, Value::Num(seed) if (*seed - 2026.0).abs() > f64::EPSILON));
        let first = block_on(rand_builtin(vec![Value::Num(1.0), Value::Num(3.0)])).expect("rand");
        block_on(rand_builtin(vec![Value::from("seed"), saved])).expect("rand restore state token");
        let second = block_on(rand_builtin(vec![Value::Num(1.0), Value::Num(3.0)])).expect("rand");
        match (prefix, first, second) {
            (Value::Tensor(prefix), Value::Tensor(a), Value::Tensor(b)) => {
                assert_eq!(prefix.shape, vec![1, 4]);
                assert_eq!(a.shape, vec![1, 3]);
                assert_eq!(b.shape, vec![1, 3]);
                assert_eq!(a.materialize_f64(), b.materialize_f64());
            }
            other => panic!("expected tensor outputs, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_legacy_seed_syncs_provider_state() {
        let _guard = random::test_guard();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            block_on(rand_builtin(vec![Value::from("seed"), Value::Num(9.0)])).expect("rand seed");
            let handle = provider.random_uniform(&[4, 1]).expect("gpu uniform");
            let host_after_gpu =
                random::generate_uniform(4, "rand provider sync").expect("uniform");
            let gpu = block_on(download_handle_async(provider, &handle)).expect("download");
            assert_eq!(gpu.data, host_after_gpu);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_legacy_seed_literal_restarts_sequence() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        block_on(rand_builtin(vec![Value::from("seed"), Value::Num(2026.0)])).expect("rand seed");
        let first = block_on(rand_builtin(vec![Value::Num(1.0), Value::Num(4.0)])).expect("rand");
        block_on(rand_builtin(vec![Value::from("seed"), Value::Num(2026.0)]))
            .expect("rand seed again");
        let second = block_on(rand_builtin(vec![Value::Num(1.0), Value::Num(4.0)])).expect("rand");
        match (first, second) {
            (Value::Tensor(a), Value::Tensor(b)) => {
                assert_eq!(a.shape, vec![1, 4]);
                assert_eq!(b.shape, vec![1, 4]);
                assert_eq!(a.materialize_f64(), b.materialize_f64());
            }
            other => panic!("expected tensor outputs, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_legacy_seed_char_array_resets_sequence() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let seed_keyword = Value::CharArray(runmat_builtins::CharArray::new_row("seed"));
        block_on(rand_builtin(vec![seed_keyword.clone(), Value::Num(17.0)]))
            .expect("rand char seed");
        let first = block_on(rand_builtin(vec![Value::Num(1.0), Value::Num(3.0)])).expect("rand");
        block_on(rand_builtin(vec![seed_keyword, Value::Num(17.0)])).expect("rand char seed again");
        let second = block_on(rand_builtin(vec![Value::Num(1.0), Value::Num(3.0)])).expect("rand");
        match (first, second) {
            (Value::Tensor(a), Value::Tensor(b)) => {
                assert_eq!(a.shape, vec![1, 3]);
                assert_eq!(a.materialize_f64(), b.materialize_f64());
            }
            other => panic!("expected tensor outputs, got {other:?}"),
        }
    }

    #[test]
    fn rand_legacy_seed_reads_typed_integer_tensor_storage_exactly() {
        let _guard = random::test_guard();
        reset_rng_clean();
        block_on(rand_builtin(vec![
            Value::from("seed"),
            poisoned_int_tensor(IntegerStorage::U16(vec![2026]), vec![1, 1]),
        ]))
        .expect("typed integer tensor seed");
        let query = block_on(rand_builtin(vec![Value::from("seed")])).expect("rand seed query");
        assert!(matches!(query, Value::Num(seed) if (seed - 2026.0).abs() < f64::EPSILON));

        let err = block_on(rand_builtin(vec![
            Value::from("seed"),
            poisoned_int_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1]),
        ]))
        .expect_err("negative typed integer seed");
        assert!(err.message().contains("non-negative"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_legacy_seed_rejects_invalid_seed_values() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let err = block_on(rand_builtin(vec![Value::from("seed"), Value::Num(-1.0)]))
            .expect_err("negative seed");
        assert!(err.message().contains("non-negative"));

        let err = block_on(rand_builtin(vec![
            Value::from("seed"),
            Value::Int(IntValue::U64((1_u64 << 53) + 1)),
        ]))
        .expect_err("imprecise integer seed");
        assert!(err.message().contains("53-bit"));

        let err = block_on(rand_builtin(vec![
            Value::from("seed"),
            Value::Num(1.0),
            Value::Num(2.0),
        ]))
        .expect_err("extra seed argument");
        assert!(err.message().contains("expects zero or one seed"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_like_without_dims_is_scalar_and_implicit_prototype_is_rejected() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(tensor.clone())];
        let result = block_on(rand_builtin(args)).expect("rand");
        assert!(matches!(result, Value::Num(value) if (0.0..1.0).contains(&value)));

        let error = block_on(rand_builtin(vec![Value::Tensor(tensor)])).unwrap_err();
        assert!(error.message().contains("unsupported dimension or option"));
    }

    #[test]
    fn rand_like_rejects_every_integer_class_and_nonfloating_prototypes() {
        let _guard = random::test_guard();
        let storages = vec![
            IntegerStorage::I8(vec![i8::MIN]),
            IntegerStorage::I16(vec![i16::MIN]),
            IntegerStorage::I32(vec![i32::MIN]),
            IntegerStorage::I64(vec![i64::MIN]),
            IntegerStorage::U8(vec![u8::MAX]),
            IntegerStorage::U16(vec![u16::MAX]),
            IntegerStorage::U32(vec![u32::MAX]),
            IntegerStorage::U64(vec![u64::MAX]),
        ];

        for storage in storages {
            let prototype = Tensor::new_integer(storage, vec![1, 1]).expect("prototype");
            let error = block_on(rand_builtin(vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::Tensor(prototype),
            ]))
            .unwrap_err();
            assert!(error.message().contains("must be single or double"));
        }

        let scalar = block_on(rand_builtin(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::Int(IntValue::U64(u64::MAX)),
        ]))
        .unwrap_err();
        assert!(scalar.message().contains("must be single or double"));

        let logical = block_on(rand_builtin(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::Bool(true),
        ]))
        .unwrap_err();
        assert!(logical.message().contains("must be single or double"));

        let complex_integer = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![2]))
                .unwrap(),
            vec![1, 1],
        )
        .unwrap();
        let complex_integer = block_on(rand_builtin(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::ComplexTensor(complex_integer),
        ]))
        .unwrap_err();
        assert!(complex_integer
            .message()
            .contains("complex numbers with integer types"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_single_matrix_has_f32_dtype() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("single")];
        let result = block_on(rand_builtin(args)).expect("rand single");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.numeric_dtype(), NumericDType::F32);
                let expected = random::expected_uniform_sequence(4)
                    .into_iter()
                    .map(|v| {
                        let val = v as f32;
                        val as f64
                    })
                    .collect::<Vec<f64>>();
                for (observed, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((*observed - *exp).abs() < 1e-7);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_like_complex_produces_complex_tensor() {
        let _guard = random::test_guard();
        let _guard = reset_rng_clean();
        let args = vec![
            Value::Num(2.0),
            Value::Num(2.0),
            Value::from("like"),
            Value::Complex(0.0, 1.0),
        ];
        let result = block_on(rand_builtin(args)).expect("rand");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                let expected = random::expected_complex_sequence(4);
                for ((re, im), (eref, eim)) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((*re - *eref).abs() < 1e-12);
                    assert!((*im - *eim).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn rand_like_complex_single_preserves_native_single() {
        let _guard = reset_rng_clean();
        let prototype =
            ComplexTensor::from_f32(vec![(1.0, 2.0)], vec![1, 1]).expect("complex single");
        let result = block_on(rand_builtin(vec![
            Value::Num(2.0),
            Value::from("like"),
            Value::ComplexTensor(prototype),
        ]))
        .expect("rand complex single like");
        let Value::ComplexTensor(output) = result else {
            panic!("expected complex tensor");
        };
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_gpuarray_keyword_produces_valid_output() {
        let _guard = reset_rng_clean();
        test_support::with_test_provider(|_| {
            let args = vec![Value::Num(3.0), Value::Num(4.0), Value::from("gpuArray")];
            let result = block_on(rand_builtin(args)).expect("rand gpuArray");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpuArray");
            };
            assert_eq!(handle.shape, vec![3, 4]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_gpu_like_uniform() {
        let _guard = random::test_guard();
        random::reset_rng();
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(rand_builtin(args)).expect("rand");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape, vec![2, 2]);
                    let gathered =
                        test_support::gather(Value::GpuTensor(gpu)).expect("gather to host");
                    assert_eq!(gathered.shape, vec![2, 2]);
                    for value in gathered.materialize_f64() {
                        assert!((0.0..1.0).contains(&value));
                    }
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rand_rejects_integer_gpu_like_prototype() {
        let _guard = random::test_guard();
        test_support::with_test_provider(|provider| {
            let values = [u64::MAX, 9_007_199_254_740_993];
            let shape = [1usize, 2usize];
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&values),
                    shape: &shape,
                })
                .expect("upload integer gpu prototype");
            let error = block_on(rand_builtin(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ]))
            .unwrap_err();
            assert!(error.message().contains("single or double underlying type"));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rand_wgpu_like_uniform_and_gather() {
        let Ok(_provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        // Create a GPU prototype and request rand like it
        let tensor = Tensor::new(vec![0.0; 4], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let provider = runmat_accelerate_api::provider().unwrap();
        let handle = provider.upload(&view).expect("upload");
        let result =
            block_on(rand_like(&Value::GpuTensor(handle), &[2, 2])).expect("rand like gpu");
        match result {
            Value::GpuTensor(h) => {
                let gathered = test_support::gather(Value::GpuTensor(h)).expect("gather to host");
                assert_eq!(gathered.shape, vec![2, 2]);
                for v in gathered.materialize_f64() {
                    assert!((0.0..1.0).contains(&v));
                }
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rand_wgpu_fusion_then_sin_then_sum() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let r = rand_double(&[2, 2]).expect("rand");
        let s = block_on(crate::call_builtin_async("sin", &[r])).expect("sin");
        let summed =
            block_on(crate::call_builtin_async("sum", &[s, Value::Num(1.0)])).expect("sum");
        let gathered = test_support::gather(summed).expect("gather");
        assert_eq!(gathered.shape, vec![1, 2]);
    }

    #[test]
    fn rand_same_shape_complex_gpu_like_stays_complex_and_resident() {
        test_support::with_f32_test_provider(|provider| {
            let prototype = ComplexTensor::from_f32(vec![(1.0, -1.0); 4], vec![2, 2])
                .expect("complex single prototype");
            let handle = gpu_helpers::upload_complex_tensor(provider, &prototype).expect("upload");
            let result = block_on(rand_like(&Value::GpuTensor(handle), &[2, 2]))
                .expect("rand complex gpu like");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_storage(&output),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            );
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn rand_single_without_explicit_gpu_intent_remains_host_resident() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let value = rand_single(&[2, 2]).expect("rand single");
        let Value::Tensor(tensor) = value else {
            panic!("expected host single tensor");
        };
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    }
}
