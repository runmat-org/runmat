//! MATLAB-compatible `colon` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    LiteralValue, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, IntValue, LogicalArray, NumericDType, Tensor, Value};

use crate::build_runtime_error;
use crate::builtins::array::type_resolvers::row_vector_type;
use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::elementwise::integer_cast::IntegerTarget;
use runmat_builtins::shape_rules::infer_range_shape;
use runmat_builtins::ResolveContext;

const MIN_RATIO_TOL: f64 = f64::EPSILON * 8.0;
const MAX_RATIO_TOL: f64 = 1e-9;
const ZERO_IM_TOL: f64 = f64::EPSILON * 32.0;
const CHAR_TOL: f64 = 1e-6;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarOrigin {
    Numeric,
    Char,
}

#[derive(Clone, Copy)]
struct ParsedScalar {
    value: f64,
    prefer_gpu: bool,
    origin: ScalarOrigin,
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::colon")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "colon",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("linspace")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Falls back to uploading the host-generated vector when provider linspace kernels are unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::colon")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "colon",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sequence generation is treated as a sink; it does not participate in fusion.",
};

fn colon_type(_args: &[Type], ctx: &ResolveContext) -> Type {
    let (start, step, end) = match ctx.literal_args.as_slice() {
        [LiteralValue::Number(start), LiteralValue::Number(end)] => {
            (Some(*start), None, Some(*end))
        }
        [LiteralValue::Number(start), LiteralValue::Number(step), LiteralValue::Number(end)] => {
            (Some(*start), Some(*step), Some(*end))
        }
        _ => (None, None, None),
    };
    infer_range_shape(start, step, end)
        .map(|shape| Type::Tensor { shape: Some(shape) })
        .unwrap_or_else(|| row_vector_type(ctx))
}

const BUILTIN_NAME: &str = "colon";

pub const COLON_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "colon-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "colon with logical operands is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ColonLogicalInputExtension"),
};

pub const COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "colon-zero-imaginary-complex",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "colon with zero-imaginary complex operands is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ColonZeroImaginaryComplexExtension"),
    };

pub const COLON_GPU_64_BIT_INTEGER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "colon-gpu-64-bit-integer",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "colon with resident int64 or uint64 operands is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ColonGpu64BitIntegerExtension"),
    };

pub const COLON_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    COLON_LOGICAL_INPUT_EXTENSION,
    COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION,
    COLON_GPU_64_BIT_INTEGER_EXTENSION,
];

const COLON_INTEGER_INPUTS_TWO: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "start", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "A typed-integer operand selects the output integer class; all typed-integer operands must use that class." },
    BuiltinIntegerInputCapability { name: "stop", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The bound must be representable by the selected output class." },
];
const COLON_INTEGER_INPUTS_THREE: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability { name: "start", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "A typed-integer operand selects the output integer class; all typed-integer operands must use that class." },
    BuiltinIntegerInputCapability { name: "step", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The explicit step is decoded exactly and may be negative even for an unsigned output class." },
    BuiltinIntegerInputCapability { name: "stop", classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The bound must be representable by the selected output class." },
];

pub const COLON_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "x = colon(integer_start, integer_stop)", inputs: &COLON_INTEGER_INPUTS_TWO, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "All eight integer classes use exact native storage with an implicit +1 step." },
    BuiltinIntegerCapabilityDescriptor { form: "x = colon(integer_start, integer_step, integer_stop)", inputs: &COLON_INTEGER_INPUTS_THREE, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "All eight integer classes use exact native storage. MATLAB-compatible GPU execution excludes int64 and uint64; RunMat mode accepts them through an exact owning-provider gather/upload fallback." },
];

const COLON_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Arithmetic progression row vector (numeric or character).",
}];

const COLON_SIG_TWO_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Start scalar value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Stop scalar value (implicit step = 1).",
    },
];

const COLON_SIG_THREE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Start scalar value.",
    },
    BuiltinParamDescriptor {
        name: "step",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Increment; zero returns an empty row vector.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Stop scalar value.",
    },
];

const COLON_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "x = colon(start, stop)",
        inputs: &COLON_SIG_TWO_INPUTS,
        outputs: &COLON_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "x = colon(start, step, stop)",
        inputs: &COLON_SIG_THREE_INPUTS,
        outputs: &COLON_OUTPUT,
    },
];

const COLON_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.ARG_COUNT",
    identifier: None,
    when: "More than three input arguments are provided.",
    message: "colon: expected two or three input arguments",
};

const COLON_ERROR_NON_SCALAR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.NON_SCALAR_INPUT",
    identifier: None,
    when: "At least one input is not scalar.",
    message: "colon: expected scalar input",
};

const COLON_ERROR_NON_FINITE_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.NON_FINITE_INPUT",
    identifier: None,
    when: "At least one input scalar is non-finite.",
    message: "colon: inputs must be finite numeric scalars",
};

const COLON_ERROR_COMPLEX_IMAGINARY_NONZERO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.COMPLEX_IMAGINARY_NONZERO",
    identifier: None,
    when: "Complex inputs have non-zero imaginary parts.",
    message: "colon: complex inputs must have zero imaginary part",
};

const COLON_ERROR_UNSUPPORTED_STRING_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.UNSUPPORTED_STRING_INPUT",
    identifier: None,
    when: "String-like values are used as scalar bounds/step.",
    message: "colon: inputs must be real scalar values; received a string-like argument",
};

const COLON_ERROR_CHAR_NON_INTEGER_CODEPOINT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.CHAR_NON_INTEGER_CODEPOINT",
    identifier: None,
    when: "Character sequence values are non-integer.",
    message: "colon: character sequence requires integer code points",
};

const COLON_ERROR_CHAR_CODEPOINT_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.CHAR_CODEPOINT_RANGE",
    identifier: None,
    when: "Character sequence values are outside valid Unicode range.",
    message: "colon: character code point out of range",
};

const COLON_ERROR_SEQUENCE_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.SEQUENCE_RANGE",
    identifier: None,
    when: "Computed progression span/ratio is non-finite.",
    message: "colon: sequence length exceeds representable range",
};

const COLON_ERROR_SEQUENCE_LIMIT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.SEQUENCE_LIMIT",
    identifier: None,
    when: "Computed progression length exceeds platform limits.",
    message: "colon: sequence length exceeds platform limits",
};

const COLON_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.INTERNAL",
    identifier: None,
    when: "Internal tensor/character output construction failed.",
    message: "colon: internal error",
};

const COLON_ERRORS: [BuiltinErrorDescriptor; 10] = [
    COLON_ERROR_ARG_COUNT,
    COLON_ERROR_NON_SCALAR_INPUT,
    COLON_ERROR_NON_FINITE_INPUT,
    COLON_ERROR_COMPLEX_IMAGINARY_NONZERO,
    COLON_ERROR_UNSUPPORTED_STRING_INPUT,
    COLON_ERROR_CHAR_NON_INTEGER_CODEPOINT,
    COLON_ERROR_CHAR_CODEPOINT_RANGE,
    COLON_ERROR_SEQUENCE_RANGE,
    COLON_ERROR_SEQUENCE_LIMIT,
    COLON_ERROR_INTERNAL,
];

pub const COLON_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COLON_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COLON_ERRORS,
};

fn colon_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    colon_error_with_message(error.message, error)
}

fn colon_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "colon",
    category = "array/creation",
    summary = "Generate arithmetic progressions with MATLAB colon semantics.",
    keywords = "colon,sequence,range,step,gpu",
    accel = "array_construct",
    type_resolver(colon_type),
    extensions(COLON_EXTENSIONS),
    integer_capabilities(COLON_INTEGER_CAPABILITIES),
    descriptor(crate::builtins::array::creation::colon::COLON_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::colon"
)]
async fn colon_builtin(
    start: Value,
    step_or_end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(colon_error(&COLON_ERROR_ARG_COUNT));
    }

    ensure_colon_extensions(&start, &step_or_end, &rest)?;

    if [&start, &step_or_end].into_iter().chain(rest.iter()).any(|value| matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())) {
        return resident_integer_colon(start, step_or_end, rest).await;
    }

    let (integer_step, integer_stop) = match rest.first() {
        Some(stop) => (Some(&step_or_end), stop),
        None => (None, &step_or_end),
    };
    if let Some(result) = try_integer_sequence(&start, integer_step, integer_stop)? {
        return Ok(result);
    }

    let start_scalar = parse_real_scalar("colon", start).await?;

    if rest.is_empty() {
        let stop_scalar = parse_real_scalar("colon", step_or_end).await?;
        let step = default_step(start_scalar.value, stop_scalar.value);
        let char_mode =
            start_scalar.origin == ScalarOrigin::Char && stop_scalar.origin == ScalarOrigin::Char;
        let explicit_gpu = if char_mode {
            false
        } else {
            start_scalar.prefer_gpu || stop_scalar.prefer_gpu
        };
        build_sequence(
            start_scalar.value,
            step,
            stop_scalar.value,
            explicit_gpu,
            char_mode,
        )
    } else {
        let step_scalar = parse_real_scalar("colon", step_or_end).await?;
        let stop_scalar = parse_real_scalar("colon", rest[0].clone()).await?;
        let char_mode =
            start_scalar.origin == ScalarOrigin::Char && stop_scalar.origin == ScalarOrigin::Char;
        let explicit_gpu = if char_mode {
            false
        } else {
            start_scalar.prefer_gpu || step_scalar.prefer_gpu || stop_scalar.prefer_gpu
        };
        if step_scalar.value == 0.0 {
            return if char_mode {
                build_char_sequence(Vec::new())
            } else {
                finalize_numeric_sequence(Vec::new(), explicit_gpu)
            };
        }
        build_sequence(
            start_scalar.value,
            step_scalar.value,
            stop_scalar.value,
            explicit_gpu,
            char_mode,
        )
    }
}

async fn resident_integer_colon(
    start: Value,
    step_or_end: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let mut owner_device = None;
    for value in [&start, &step_or_end].into_iter().chain(rest.iter()) {
        if let Value::GpuTensor(handle) = value {
            if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COLON_LOGICAL_INPUT_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            let device = runmat_accelerate_api::provider_for_handle(handle)
                .ok_or_else(|| {
                    colon_error_with_message(
                        "colon: resident input provider is unavailable",
                        &COLON_ERROR_INTERNAL,
                    )
                })?
                .device_id();
            if owner_device.is_some_and(|owner| owner != device) {
                return Err(colon_error_with_message(
                    "colon: resident inputs must belong to the same provider",
                    &COLON_ERROR_INTERNAL,
                ));
            }
            owner_device = Some(device);
        }
    }
    let provider = [&start, &step_or_end]
        .into_iter()
        .chain(rest.iter())
        .find_map(|value| match value {
            Value::GpuTensor(handle) => runmat_accelerate_api::provider_for_handle(handle),
            _ => None,
        });
    let start = gather_colon_operand(start).await?;
    let step_or_end = gather_colon_operand(step_or_end).await?;
    let rest = match rest.into_iter().next() {
        Some(value) => vec![gather_colon_operand(value).await?],
        None => Vec::new(),
    };
    let (step, stop) = match rest.first() {
        Some(stop) => (Some(&step_or_end), stop),
        None => (None, &step_or_end),
    };
    let result = try_integer_sequence(&start, step, stop)?.ok_or_else(|| {
        colon_error_with_message(
            "colon: resident integer operands require a typed-integer sequence",
            &COLON_ERROR_INTERNAL,
        )
    })?;
    let Some(provider) = provider else {
        return Ok(result);
    };
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, result).map_err(|error| {
        colon_error_with_message(format!("colon: {error}"), &COLON_ERROR_INTERNAL)
    })?;
    let handle = gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| {
        colon_error_with_message(format!("colon: {error}"), &COLON_ERROR_INTERNAL)
    })?;
    Ok(gpu_helpers::resident_gpu_value(handle))
}

async fn gather_colon_operand(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(&value).await,
        other => Ok(other),
    }
}

fn ensure_colon_extensions(
    start: &Value,
    step_or_end: &Value,
    rest: &[Value],
) -> crate::BuiltinResult<()> {
    for value in std::iter::once(start)
        .chain(std::iter::once(step_or_end))
        .chain(rest.iter())
    {
        if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &COLON_LOGICAL_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if let Value::GpuTensor(handle) = value {
            if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COLON_LOGICAL_INPUT_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if matches!(
                runmat_accelerate_api::handle_integer_type(handle),
                Some(
                    runmat_accelerate_api::IntegerElementType::I64
                        | runmat_accelerate_api::IntegerElementType::U64
                )
            ) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COLON_GPU_64_BIT_INTEGER_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
        }
    }
    Ok(())
}

/// Integer colon expressions must retain their exact class and cannot use the
/// floating compatibility view of a typed tensor. In particular, range checks
/// happen before materialization so `int8(1):256` fails rather than producing
/// a double vector or a saturated final element.
fn try_integer_sequence(
    start: &Value,
    step: Option<&Value>,
    stop: &Value,
) -> crate::BuiltinResult<Option<Value>> {
    let target = integer_target_from_values([start, stop].into_iter().chain(step))?;
    let Some(target) = target else {
        return Ok(None);
    };

    let start = integer_colon_value(start, target, true)?;
    let stop = integer_colon_value(stop, target, true)?;
    let step = match step {
        Some(step) => integer_colon_value(step, target, false)?,
        None => 1,
    };
    if step == 0 {
        let tensor =
            Tensor::new_integer(target.storage(Vec::new()), vec![1, 0]).map_err(|error| {
                colon_error_with_message(format!("colon: {error}"), &COLON_ERROR_INTERNAL)
            })?;
        return Ok(Some(tensor::tensor_into_value(tensor)));
    }

    let count = integer_progression_count(start, step, stop)?;
    let count = usize::try_from(count).map_err(|_| colon_error(&COLON_ERROR_SEQUENCE_LIMIT))?;
    let mut values = Vec::with_capacity(count);
    let mut value = start;
    for index in 0..count {
        values.push(integer_value_from_i128(target, value));
        if index + 1 < count {
            value = value
                .checked_add(step)
                .ok_or_else(|| colon_error(&COLON_ERROR_SEQUENCE_RANGE))?;
        }
    }
    let tensor = Tensor::new_integer(target.storage(values), vec![1, count]).map_err(|error| {
        colon_error_with_message(format!("colon: {error}"), &COLON_ERROR_INTERNAL)
    })?;
    Ok(Some(tensor::tensor_into_value(tensor)))
}

fn integer_target_from_values<'a>(
    values: impl IntoIterator<Item = &'a Value>,
) -> crate::BuiltinResult<Option<IntegerTarget>> {
    let mut target = None;
    for value in values {
        let Some(candidate) = typed_integer_target(value)? else {
            continue;
        };
        if let Some(target) = target {
            if target != candidate {
                return Err(colon_error_with_message(
                    "colon: integer operands must have the same integer class",
                    &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
                ));
            }
        } else {
            target = Some(candidate);
        }
    }
    Ok(target)
}

fn typed_integer_target(value: &Value) -> crate::BuiltinResult<Option<IntegerTarget>> {
    match value {
        Value::Int(value) => Ok(Some(IntegerTarget::from_int_value(value))),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            if !tensor::is_scalar_tensor(tensor) {
                return Err(colon_error_with_message(
                    "colon: expected scalar input",
                    &COLON_ERROR_NON_SCALAR_INPUT,
                ));
            }
            Ok(tensor.integer_storage().map(IntegerTarget::from_storage))
        }
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
            if complex_tensor_element_len(tensor) != 1 {
                return Err(colon_error_with_message(
                    "colon: expected scalar input",
                    &COLON_ERROR_NON_SCALAR_INPUT,
                ));
            }
            let storage = tensor
                .integer_storage()
                .expect("typed complex integer storage");
            if !storage
                .imag
                .value_at(0)
                .expect("scalar imaginary value")
                .is_zero()
            {
                return Err(colon_error(&COLON_ERROR_COMPLEX_IMAGINARY_NONZERO));
            }
            Ok(Some(IntegerTarget::from_storage(&storage.real)))
        }
        _ => Ok(None),
    }
}

fn integer_colon_value(
    value: &Value,
    target: IntegerTarget,
    require_target_range: bool,
) -> crate::BuiltinResult<i128> {
    let integer = match value {
        Value::Int(value) => {
            if IntegerTarget::from_int_value(value) != target {
                return Err(colon_error_with_message(
                    "colon: integer operands must have the same integer class",
                    &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
                ));
            }
            int_value_to_i128(value)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            if !tensor::is_scalar_tensor(tensor) {
                return Err(colon_error_with_message(
                    "colon: expected scalar input",
                    &COLON_ERROR_NON_SCALAR_INPUT,
                ));
            }
            let storage = tensor
                .integer_storage()
                .expect("typed tensor storage is present");
            if IntegerTarget::from_storage(storage) != target {
                return Err(colon_error_with_message(
                    "colon: integer operands must have the same integer class",
                    &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
                ));
            }
            int_value_to_i128(&storage.value_at(0).expect("scalar typed tensor value"))
        }
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
            if complex_tensor_element_len(tensor) != 1 {
                return Err(colon_error_with_message(
                    "colon: expected scalar input",
                    &COLON_ERROR_NON_SCALAR_INPUT,
                ));
            }
            let storage = tensor
                .integer_storage()
                .expect("typed complex integer storage");
            if !storage
                .imag
                .value_at(0)
                .expect("scalar imaginary value")
                .is_zero()
            {
                return Err(colon_error(&COLON_ERROR_COMPLEX_IMAGINARY_NONZERO));
            }
            if IntegerTarget::from_storage(&storage.real) != target {
                return Err(colon_error_with_message(
                    "colon: integer operands must have the same integer class",
                    &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
                ));
            }
            int_value_to_i128(&storage.real.value_at(0).expect("scalar real value"))
        }
        Value::Num(value) => float_to_integral_i128(*value)?,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if tensor.numeric_dtype() != NumericDType::F64 {
                return Err(colon_error_with_message(
                    "colon: integer sequences permit only a full scalar double as a noninteger operand",
                    &COLON_ERROR_SEQUENCE_RANGE,
                ));
            }
            float_to_integral_i128(tensor::tensor_value_f64(tensor, 0))?
        }
        Value::Bool(value) => i128::from(u8::from(*value)),
        Value::LogicalArray(array) if array.len() == 1 => i128::from(u8::from(array.data[0] != 0)),
        _ => {
            return Err(colon_error_with_message(
                "colon: integer sequences require real scalar operands",
                &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
            ))
        }
    };
    if require_target_range && !integer_target_contains(target, integer) {
        return Err(colon_error_with_message(
            "colon: integer range values must be representable in the output class",
            &COLON_ERROR_SEQUENCE_RANGE,
        ));
    }
    Ok(integer)
}

fn float_to_integral_i128(value: f64) -> crate::BuiltinResult<i128> {
    if !value.is_finite() {
        return Err(colon_error(&COLON_ERROR_NON_FINITE_INPUT));
    }
    if value.fract() != 0.0 || value < i64::MIN as f64 || value >= 18_446_744_073_709_551_616.0 {
        return Err(colon_error_with_message(
            "colon: integer sequences require integer-valued scalar operands",
            &COLON_ERROR_SEQUENCE_RANGE,
        ));
    }
    Ok(value as i128)
}

fn int_value_to_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn integer_target_contains(target: IntegerTarget, value: i128) -> bool {
    match target {
        IntegerTarget::I8 => value >= i128::from(i8::MIN) && value <= i128::from(i8::MAX),
        IntegerTarget::I16 => value >= i128::from(i16::MIN) && value <= i128::from(i16::MAX),
        IntegerTarget::I32 => value >= i128::from(i32::MIN) && value <= i128::from(i32::MAX),
        IntegerTarget::I64 => value >= i128::from(i64::MIN) && value <= i128::from(i64::MAX),
        IntegerTarget::U8 => value >= 0 && value <= i128::from(u8::MAX),
        IntegerTarget::U16 => value >= 0 && value <= i128::from(u16::MAX),
        IntegerTarget::U32 => value >= 0 && value <= i128::from(u32::MAX),
        IntegerTarget::U64 => value >= 0 && value <= i128::from(u64::MAX),
    }
}

fn integer_value_from_i128(target: IntegerTarget, value: i128) -> IntValue {
    debug_assert!(integer_target_contains(target, value));
    match target {
        IntegerTarget::I8 => IntValue::I8(value as i8),
        IntegerTarget::I16 => IntValue::I16(value as i16),
        IntegerTarget::I32 => IntValue::I32(value as i32),
        IntegerTarget::I64 => IntValue::I64(value as i64),
        IntegerTarget::U8 => IntValue::U8(value as u8),
        IntegerTarget::U16 => IntValue::U16(value as u16),
        IntegerTarget::U32 => IntValue::U32(value as u32),
        IntegerTarget::U64 => IntValue::U64(value as u64),
    }
}

fn integer_progression_count(start: i128, step: i128, stop: i128) -> crate::BuiltinResult<u128> {
    if (step > 0 && start > stop) || (step < 0 && start < stop) {
        return Ok(0);
    }
    let distance = if step > 0 { stop - start } else { start - stop };
    let step = step.unsigned_abs();
    let count = (distance as u128)
        .checked_div(step)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| colon_error(&COLON_ERROR_SEQUENCE_LIMIT))?;
    if count > usize::MAX as u128 {
        return Err(colon_error(&COLON_ERROR_SEQUENCE_LIMIT));
    }
    Ok(count)
}

fn build_sequence(
    start: f64,
    step: f64,
    stop: f64,
    explicit_gpu: bool,
    char_mode: bool,
) -> crate::BuiltinResult<Value> {
    if !start.is_finite() || !step.is_finite() || !stop.is_finite() {
        return Err(colon_error(&COLON_ERROR_NON_FINITE_INPUT));
    }
    if step == 0.0 {
        return finalize_numeric_sequence(Vec::new(), explicit_gpu);
    }

    let plan = plan_progression(start, step, stop)?;

    if char_mode {
        let data = materialize_progression(&plan, start, step);
        return build_char_sequence(data);
    }

    if plan.count == 0 {
        return finalize_numeric_sequence(Vec::new(), explicit_gpu);
    }

    let prefer_gpu =
        sequence_gpu_preference(plan.count, SequenceIntent::Colon, explicit_gpu).prefer_gpu;

    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(handle) = provider.linspace(start, plan.final_end, plan.count) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    let data = materialize_progression(&plan, start, step);
    finalize_numeric_sequence(data, prefer_gpu)
}

fn finalize_numeric_sequence(data: Vec<f64>, prefer_gpu: bool) -> crate::BuiltinResult<Value> {
    let len = data.len();
    let shape = vec![1usize, len];

    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
        if let Some(provider) = runmat_accelerate_api::provider() {
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                return Ok(Value::GpuTensor(handle));
            }
        }
    }

    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|e| colon_error_with_message(format!("colon: {e}"), &COLON_ERROR_INTERNAL))
}

struct ProgressionPlan {
    count: usize,
    final_end: f64,
}

fn plan_progression(start: f64, step: f64, stop: f64) -> crate::BuiltinResult<ProgressionPlan> {
    let tol = tolerance(start, step, stop);
    let step_abs = step.abs();

    if step > 0.0 && start > stop + tol {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }
    if step < 0.0 && start < stop - tol {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }

    let diff = (stop - start) / step;
    if !diff.is_finite() {
        return Err(colon_error(&COLON_ERROR_SEQUENCE_RANGE));
    }

    let ratio_raw = (tol / step_abs).abs();
    let ratio_tol = ratio_raw
        .max(MIN_RATIO_TOL)
        .clamp(f64::EPSILON, MAX_RATIO_TOL);
    let mut approx = diff + ratio_tol;

    if approx < 0.0 {
        if approx.abs() <= ratio_tol {
            approx = 0.0;
        } else {
            return Ok(ProgressionPlan {
                count: 0,
                final_end: start,
            });
        }
    }

    if approx.is_infinite() || approx > usize::MAX as f64 {
        return Err(colon_error(&COLON_ERROR_SEQUENCE_LIMIT));
    }

    let floor = approx.floor();
    let count = floor as usize;
    let count = count
        .checked_add(1)
        .ok_or_else(|| colon_error(&COLON_ERROR_SEQUENCE_LIMIT))?;

    if count == 0 {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }

    let computed_end = start + step * ((count - 1) as f64);
    let final_end = if (computed_end - stop).abs() <= tol {
        stop
    } else {
        computed_end
    };

    Ok(ProgressionPlan { count, final_end })
}

fn materialize_progression(plan: &ProgressionPlan, start: f64, step: f64) -> Vec<f64> {
    if plan.count == 0 {
        return Vec::new();
    }
    let mut data = Vec::with_capacity(plan.count);
    for idx in 0..plan.count {
        data.push(start + step * (idx as f64));
    }
    if let Some(last) = data.last_mut() {
        *last = plan.final_end;
    }
    data
}

fn default_step(_start: f64, _stop: f64) -> f64 {
    // MATLAB's implicit step is always +1. Descending sequences require an explicit
    // negative increment (three-argument form); otherwise the result is empty.
    1.0
}

fn tolerance(start: f64, step: f64, stop: f64) -> f64 {
    let span = (stop - start).abs();
    let base = start.abs().max(stop.abs()).max(span).max(1.0);
    let step_term = step.abs().max(1.0);
    let tol = base * f64::EPSILON * 32.0 + step_term * f64::EPSILON * 16.0;
    tol.max(f64::EPSILON)
}

async fn parse_real_scalar(name: &str, value: Value) -> crate::BuiltinResult<ParsedScalar> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            let scalar = tensor_scalar(name, &tensor)?;
            Ok(ParsedScalar {
                value: scalar,
                prefer_gpu: true,
                origin: ScalarOrigin::Numeric,
            })
        }
        other => parse_real_scalar_host(name, other),
    }
}

fn parse_real_scalar_host(name: &str, value: Value) -> crate::BuiltinResult<ParsedScalar> {
    match value {
        Value::Num(n) => ensure_finite(name, n).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Int(i) => Ok(ParsedScalar {
            value: i.to_f64(),
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Bool(b) => Ok(ParsedScalar {
            value: if b { 1.0 } else { 0.0 },
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Tensor(t) => tensor_scalar(name, &t).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::LogicalArray(logical) => logical_scalar(name, &logical).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::Complex(re, im) => complex_to_real(name, re, im).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::ComplexTensor(t) => complex_tensor_scalar(name, &t).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
        }),
        Value::CharArray(chars) => char_scalar(name, &chars).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Char,
        }),
        Value::String(_) | Value::StringArray(_) => Err(colon_error_with_message(
            format!("{name}: inputs must be real scalar values; received a string-like argument"),
            &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
        )),
        Value::GpuTensor(_) => unreachable!("GpuTensor handled by parse_real_scalar"),
        other => Err(colon_error_with_message(
            format!("{name}: inputs must be real scalar values; received {other:?}"),
            &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
        )),
    }
}

fn ensure_finite(name: &str, value: f64) -> crate::BuiltinResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(colon_error_with_message(
            format!("{name}: inputs must be finite numeric scalars"),
            &COLON_ERROR_NON_FINITE_INPUT,
        ))
    }
}

fn tensor_scalar(name: &str, tensor: &Tensor) -> crate::BuiltinResult<f64> {
    if !tensor::is_scalar_tensor(tensor) {
        return Err(colon_error_with_message(
            format!("{name}: expected scalar input"),
            &COLON_ERROR_NON_SCALAR_INPUT,
        ));
    }
    ensure_finite(name, tensor::tensor_value_f64(tensor, 0))
}

fn logical_scalar(name: &str, logical: &LogicalArray) -> crate::BuiltinResult<f64> {
    if logical.len() != 1 {
        return Err(colon_error_with_message(
            format!("{name}: expected scalar input"),
            &COLON_ERROR_NON_SCALAR_INPUT,
        ));
    }
    Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
}

fn complex_to_real(name: &str, re: f64, im: f64) -> crate::BuiltinResult<f64> {
    if im.abs() > ZERO_IM_TOL * re.abs().max(1.0) {
        return Err(colon_error_with_message(
            format!("{name}: complex inputs must have zero imaginary part"),
            &COLON_ERROR_COMPLEX_IMAGINARY_NONZERO,
        ));
    }
    ensure_finite(name, re)
}

fn complex_tensor_scalar(name: &str, tensor: &ComplexTensor) -> crate::BuiltinResult<f64> {
    if complex_tensor_element_len(tensor) != 1 {
        return Err(colon_error_with_message(
            format!("{name}: expected scalar input"),
            &COLON_ERROR_NON_SCALAR_INPUT,
        ));
    }
    if let Some(storage) = tensor.integer_storage() {
        let re = storage
            .real
            .value_at(0)
            .expect("scalar complex integer tensor has one real value")
            .to_f64();
        let im = storage
            .imag
            .value_at(0)
            .expect("scalar complex integer tensor has one imaginary value")
            .to_f64();
        return complex_to_real(name, re, im);
    }
    let (re, im) = tensor.materialize_f64()[0];
    complex_to_real(name, re, im)
}

fn complex_tensor_element_len(tensor: &ComplexTensor) -> usize {
    tensor
        .integer_storage()
        .as_ref()
        .map_or(tensor.materialize_f64().len(), |storage| storage.real.len())
}

fn char_scalar(name: &str, array: &CharArray) -> crate::BuiltinResult<f64> {
    if array.rows * array.cols != 1 {
        return Err(colon_error_with_message(
            format!("{name}: expected scalar input"),
            &COLON_ERROR_NON_SCALAR_INPUT,
        ));
    }
    let ch = array.data[0];
    Ok(ch as u32 as f64)
}

fn build_char_sequence(data: Vec<f64>) -> crate::BuiltinResult<Value> {
    let len = data.len();
    let mut chars = Vec::with_capacity(len);
    for value in data {
        let rounded = value.round();
        if (value - rounded).abs() > CHAR_TOL {
            return Err(colon_error(&COLON_ERROR_CHAR_NON_INTEGER_CODEPOINT));
        }
        if !(0.0..=(u32::MAX as f64)).contains(&rounded) {
            return Err(colon_error(&COLON_ERROR_CHAR_CODEPOINT_RANGE));
        }
        let code = rounded as u32;
        let ch = std::char::from_u32(code)
            .ok_or_else(|| colon_error(&COLON_ERROR_CHAR_CODEPOINT_RANGE))?;
        chars.push(ch);
    }

    let array = CharArray::new(chars, 1, len)
        .map_err(|e| colon_error_with_message(format!("colon: {e}"), &COLON_ERROR_INTERNAL))?;
    Ok(Value::CharArray(array))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{
        CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, Tensor,
    };

    fn colon_builtin(start: Value, stop: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::colon_builtin(start, stop, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_basic_increasing() {
        let result = colon_builtin(Value::Num(1.0), Value::Num(5.0), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_preserves_exact_integer_storage_for_all_classes() {
        let cases = [
            (
                IntValue::I8(-2),
                IntValue::I8(2),
                IntegerStorage::I8(vec![-2, -1, 0, 1, 2]),
            ),
            (
                IntValue::I16(-2),
                IntValue::I16(2),
                IntegerStorage::I16(vec![-2, -1, 0, 1, 2]),
            ),
            (
                IntValue::I32(-2),
                IntValue::I32(2),
                IntegerStorage::I32(vec![-2, -1, 0, 1, 2]),
            ),
            (
                IntValue::I64(-2),
                IntValue::I64(2),
                IntegerStorage::I64(vec![-2, -1, 0, 1, 2]),
            ),
            (
                IntValue::U8(0),
                IntValue::U8(4),
                IntegerStorage::U8(vec![0, 1, 2, 3, 4]),
            ),
            (
                IntValue::U16(0),
                IntValue::U16(4),
                IntegerStorage::U16(vec![0, 1, 2, 3, 4]),
            ),
            (
                IntValue::U32(0),
                IntValue::U32(4),
                IntegerStorage::U32(vec![0, 1, 2, 3, 4]),
            ),
            (
                IntValue::U64(9_007_199_254_740_992),
                IntValue::U64(9_007_199_254_740_994),
                IntegerStorage::U64(vec![
                    9_007_199_254_740_992,
                    9_007_199_254_740_993,
                    9_007_199_254_740_994,
                ]),
            ),
        ];

        for (start, stop, expected) in cases {
            let value = colon_builtin(Value::Int(start), Value::Int(stop), Vec::new())
                .expect("integer colon");
            let Value::Tensor(tensor) = value else {
                panic!("integer sequence should be an array");
            };
            assert_eq!(tensor.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn colon_integer_sequences_support_signed_steps_and_reject_unrepresentable_ranges() {
        let value = colon_builtin(
            Value::Int(IntValue::U32(5)),
            Value::Num(-2.0),
            vec![Value::Int(IntValue::U32(1))],
        )
        .expect("unsigned endpoint with signed step");
        let Value::Tensor(tensor) = value else {
            panic!("integer sequence should be an array");
        };
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U32(vec![5, 3, 1]))
        );

        let err = colon_builtin(Value::Int(IntValue::I8(1)), Value::Num(256.0), Vec::new())
            .expect_err("out-of-range integer endpoint must fail");
        assert!(err.message().contains("representable"));

        let err = colon_builtin(
            Value::Int(IntValue::I16(1)),
            Value::Int(IntValue::I8(2)),
            Vec::new(),
        )
        .expect_err("mixed integer classes must fail");
        assert!(err.message().contains("same integer class"));

        assert_eq!(
            colon_builtin(
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Int(IntValue::U64(u64::MAX)),
                Vec::new(),
            )
            .expect("singleton uint64 range"),
            Value::Int(IntValue::U64(u64::MAX))
        );
    }

    #[test]
    fn colon_type_is_row_vector() {
        assert_eq!(
            colon_type(&[Type::Num, Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![Some(1), None])
            }
        );
    }

    #[test]
    fn colon_type_infers_literal_length() {
        let ctx = ResolveContext::new(vec![
            LiteralValue::Number(-2.0),
            LiteralValue::Number(0.02),
            LiteralValue::Number(2.0),
        ]);
        assert_eq!(
            colon_type(&[Type::Num, Type::Num, Type::Num], &ctx),
            Type::Tensor {
                shape: Some(vec![Some(1), Some(201)])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_two_arg_descending_returns_empty() {
        let result = colon_builtin(Value::Num(5.0), Value::Num(1.0), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_three_arg_descending() {
        let result =
            colon_builtin(Value::Num(5.0), Value::Num(-1.0), vec![Value::Num(1.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.materialize_f64(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_custom_step_reaches_stop() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(0.5), vec![Value::Num(2.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                assert_eq!(t.materialize_f64(), vec![0.0, 0.5, 1.0, 1.5, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_custom_step_stops_before_bound() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(2.0), vec![Value::Num(5.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.materialize_f64(), vec![0.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_sign_mismatch_returns_empty() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(-1.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_zero_increment_returns_empty_in_selected_class() {
        let Value::Tensor(result) =
            colon_builtin(Value::Num(0.0), Value::Num(0.0), vec![Value::Num(1.0)]).expect("empty")
        else {
            panic!("expected empty tensor");
        };
        assert_eq!(result.shape, vec![1, 0]);
        let Value::Tensor(typed) = colon_builtin(
            Value::Int(IntValue::U16(0)),
            Value::Num(0.0),
            vec![Value::Int(IntValue::U16(1))],
        )
        .expect("typed empty") else {
            panic!("expected typed empty");
        };
        assert_eq!(
            typed.integer_storage(),
            Some(&IntegerStorage::U16(Vec::new()))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_accepts_scalar_tensors() {
        let start = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let result =
            colon_builtin(Value::Tensor(start), Value::Tensor(stop), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_scalar_tensors_read_typed_integer_storage_exactly() {
        let start = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993]),
            vec![1, 1],
        )
        .unwrap();
        let stop = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_995]),
            vec![1, 1],
        )
        .unwrap();

        let result =
            colon_builtin(Value::Tensor(start), Value::Tensor(stop), Vec::new()).expect("colon");
        match result {
            Value::Tensor(tensor) => assert_eq!(
                tensor.integer_storage(),
                Some(&runmat_value::IntegerStorage::U64(vec![
                    9_007_199_254_740_993,
                    9_007_199_254_740_994,
                    9_007_199_254_740_995,
                ]))
            ),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_double_path_scalar_tensor_reads_typed_integer_storage_exactly() {
        let start =
            Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![4]), vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![6.0], vec![1, 1]).unwrap();

        let result =
            colon_builtin(Value::Tensor(start), Value::Tensor(stop), Vec::new()).expect("colon");
        match result {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![4.0, 5.0, 6.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_int64_accepts_integral_scalar_double_values_and_tensors_exactly() {
        let start = Value::Int(IntValue::I64(1));
        let Value::Tensor(scalar) =
            colon_builtin(start.clone(), Value::Num(3.0), Vec::new()).expect("scalar double")
        else {
            panic!("expected typed range");
        };
        assert_eq!(
            scalar.integer_storage(),
            Some(&IntegerStorage::I64(vec![1, 2, 3]))
        );

        let double_tensor = Tensor::new(vec![3.0], vec![1, 1]).expect("double scalar");
        let Value::Tensor(tensor) = colon_builtin(start, Value::Tensor(double_tensor), Vec::new())
            .expect("scalar double tensor")
        else {
            panic!("expected typed range");
        };
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::I64(vec![1, 2, 3]))
        );
    }

    #[test]
    fn colon_resident_uint32_preserves_exact_class_and_residency() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new_integer(IntegerStorage::U32(vec![u32::MAX - 2]), vec![1, 1])
                .expect("start");
            let stop =
                Tensor::new_integer(IntegerStorage::U32(vec![u32::MAX]), vec![1, 1]).expect("stop");
            let start = gpu_helpers::upload_tensor(provider, &start).expect("upload start");
            let stop = gpu_helpers::upload_tensor(provider, &stop).expect("upload stop");
            let result = colon_builtin(Value::GpuTensor(start), Value::GpuTensor(stop), Vec::new())
                .expect("resident colon");
            let Value::GpuTensor(handle) = &result else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(handle),
                Some(runmat_accelerate_api::IntegerElementType::U32)
            );
            let gathered = block_on(gpu_helpers::gather_value_async(&result)).expect("gather");
            let Value::Tensor(tensor) = gathered else {
                panic!("expected typed tensor");
            };
            assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::U32(vec![
                    u32::MAX - 2,
                    u32::MAX - 1,
                    u32::MAX
                ]))
            );
        });
    }

    #[test]
    fn colon_resident_uint64_is_independently_mode_gated() {
        test_support::with_test_provider(|provider| {
            let value = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("value");
            let handle = gpu_helpers::upload_tensor(provider, &value).expect("upload");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = colon_builtin(
                Value::GpuTensor(handle.clone()),
                Value::GpuTensor(handle),
                Vec::new(),
            )
            .expect_err("uint64 GPU colon is extension");
            assert_eq!(
                error.identifier(),
                COLON_GPU_64_BIT_INTEGER_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn colon_resident_logical_and_complex_metadata_are_gated_before_dispatch() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).expect("tensor");
            let logical = provider
                .upload(&HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("logical upload");
            runmat_accelerate_api::set_handle_logical(&logical, true);
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = colon_builtin(
                Value::GpuTensor(logical.clone()),
                Value::GpuTensor(logical),
                Vec::new(),
            )
            .expect_err("logical gate");
            assert_eq!(
                error.identifier(),
                COLON_LOGICAL_INPUT_EXTENSION.error_identifier
            );
            drop(_compat);
            let complex = provider
                .upload(&HostTensorView {
                    data: &[1.0, 0.0],
                    shape: &[1, 1],
                })
                .expect("complex upload");
            runmat_accelerate_api::set_handle_storage(
                &complex,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            );
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = colon_builtin(
                Value::GpuTensor(complex.clone()),
                Value::GpuTensor(complex),
                Vec::new(),
            )
            .expect_err("complex gate");
            assert_eq!(
                error.identifier(),
                COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn colon_complex_integer_scalar_tensors_read_exact_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let start = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![0]))
                .expect("complex storage"),
            vec![1, 1],
        )
        .expect("start");
        let stop = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![3]), IntegerStorage::I16(vec![0]))
                .expect("complex storage"),
            vec![1, 1],
        )
        .expect("stop");

        let result = colon_builtin(
            Value::ComplexTensor(start),
            Value::ComplexTensor(stop),
            Vec::new(),
        )
        .expect("colon");
        match result {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let nonzero_imag = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![1]))
                .expect("complex storage"),
            vec![1, 1],
        )
        .expect("nonzero imag");
        let err = colon_builtin(
            Value::ComplexTensor(nonzero_imag),
            Value::Num(3.0),
            Vec::new(),
        )
        .expect_err("nonzero imaginary part must reject");
        assert_eq!(
            err.identifier(),
            COLON_ERROR_COMPLEX_IMAGINARY_NONZERO.identifier
        );

        let wide_start = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_993]),
                IntegerStorage::U64(vec![0]),
            )
            .expect("wide start"),
            vec![1, 1],
        )
        .expect("wide start tensor");
        let wide_stop = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_995]),
                IntegerStorage::U64(vec![0]),
            )
            .expect("wide stop"),
            vec![1, 1],
        )
        .expect("wide stop tensor");
        let Value::Tensor(wide) = colon_builtin(
            Value::ComplexTensor(wide_start),
            Value::ComplexTensor(wide_stop),
            Vec::new(),
        )
        .expect("wide exact complex colon") else {
            panic!("expected typed range");
        };
        assert_eq!(
            wide.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                9_007_199_254_740_994,
                9_007_199_254_740_995
            ]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let start_view = HostTensorView {
                data: &start.materialize_f64(),
                shape: &start.shape,
            };
            let start_handle = provider.upload(&start_view).expect("upload start");

            let result = colon_builtin(
                Value::GpuTensor(start_handle),
                Value::Num(0.5),
                vec![Value::Num(2.0)],
            )
            .expect("colon");

            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 5]);
                    assert_eq!(gathered.materialize_f64(), vec![0.0, 0.5, 1.0, 1.5, 2.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn colon_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());

        let cpu = colon_builtin(Value::Num(-2.0), Value::Num(0.5), vec![Value::Num(1.0)])
            .expect("colon host");

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let start = Tensor::new(vec![-2.0], vec![1, 1]).unwrap();
        let start_view = HostTensorView {
            data: &start.materialize_f64(),
            shape: &start.shape,
        };
        let start_handle = provider.upload(&start_view).expect("upload start");
        let gpu = colon_builtin(
            Value::GpuTensor(start_handle),
            Value::Num(0.5),
            vec![Value::Num(1.0)],
        )
        .expect("colon gpu");

        let gathered = match gpu {
            Value::GpuTensor(handle) => {
                test_support::gather(Value::GpuTensor(handle)).expect("gather gpu")
            }
            other => panic!("expected GPU tensor, got {other:?}"),
        };

        let expected = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected CPU result {other:?}"),
        };

        assert_eq!(gathered.shape, expected.shape);
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_bool_inputs_promote() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result =
            colon_builtin(Value::Bool(false), Value::Bool(true), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![0.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_increasing() {
        let start = Value::CharArray(CharArray::new_row("a"));
        let stop = Value::CharArray(CharArray::new_row("e"));
        let result = colon_builtin(start, stop, Vec::new()).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 5);
                let expected: Vec<char> = "abcde".chars().collect();
                assert_eq!(arr.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_with_step() {
        let start = Value::CharArray(CharArray::new_row("a"));
        let step = Value::Num(2.0);
        let stop = Value::CharArray(CharArray::new_row("g"));
        let result = colon_builtin(start, step, vec![stop]).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 4);
                let expected: Vec<char> = "aceg".chars().collect();
                assert_eq!(arr.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_equal_endpoints_singleton() {
        let result = colon_builtin(Value::Num(3.0), Value::Num(3.0), Vec::new()).expect("colon");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![3.0]);
            }
            other => panic!("expected scalar-compatible result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_complex_imaginary_errors() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = colon_builtin(Value::Complex(1.0, 1e-2), Value::Num(2.0), Vec::new())
            .expect_err("colon should reject complex inputs");
        assert!(
            err.message().contains("zero imaginary part"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_string_input_errors() {
        let err = colon_builtin(Value::from("hello"), Value::Num(2.0), Vec::new())
            .expect_err("colon should reject string inputs");
        assert!(
            err.message().contains("string-like"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_two_arg_descending_returns_empty() {
        let start = Value::CharArray(CharArray::new_row("f"));
        let stop = Value::CharArray(CharArray::new_row("b"));
        let result = colon_builtin(start, stop, Vec::new()).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 0);
                assert!(arr.data.is_empty());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_three_arg_descending() {
        let start = Value::CharArray(CharArray::new_row("f"));
        let step = Value::Num(-1.0);
        let stop = Value::CharArray(CharArray::new_row("b"));
        let result = colon_builtin(start, step, vec![stop]).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 5);
                let expected: Vec<char> = "fedcb".chars().collect();
                assert_eq!(arr.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_fractional_step_errors() {
        let start = Value::CharArray(CharArray::new_row("a"));
        let stop = Value::CharArray(CharArray::new_row("d"));
        let err = colon_builtin(start, Value::Num(1.5), vec![stop])
            .expect_err("colon should reject fractional char steps");
        assert!(
            err.message()
                .contains("character sequence requires integer"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_gpu_step_scalar_residency() {
        test_support::with_test_provider(|provider| {
            let step = Tensor::new(vec![0.5], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &step.materialize_f64(),
                shape: &step.shape,
            };
            let step_handle = provider.upload(&view).expect("upload step");
            let result = colon_builtin(
                Value::Num(0.0),
                Value::GpuTensor(step_handle),
                vec![Value::Num(2.0)],
            )
            .expect("colon");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.materialize_f64(), vec![0.0, 0.5, 1.0, 1.5, 2.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }
}
