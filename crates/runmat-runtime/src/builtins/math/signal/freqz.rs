//! Focused MATLAB-compatible `freqz` digital filter response.

use num_complex::Complex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexStorage, ComplexTensor, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::signal::common::{
    parse_nonnegative_integer, parse_scalar_f64, value_to_complex_vector,
};
use crate::builtins::math::signal::type_resolvers::freqz_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "freqz";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::freqz")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "freqz",
    op_kind: GpuOpKind::Custom("frequency-response"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Interactive resident inputs are a RunMat-only extension; after compatibility preflight they gather through the owning provider and produce host outputs.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::freqz")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "freqz",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "freqz materialises response vectors and is not fused.",
};

const FREQZ_OUTPUT_H: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "H",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex frequency response.",
}];

const FREQZ_OUTPUT_H_W: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "H",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex frequency response.",
    },
    BuiltinParamDescriptor {
        name: "w",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Frequencies in radians/sample or Hz when fs is supplied.",
    },
];

const FREQZ_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
];

const FREQZ_INPUTS_N: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("512"),
        description: "Number of response samples.",
    },
];

const FREQZ_INPUTS_N_FS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficient vector.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("512"),
        description: "Number of response samples.",
    },
    BuiltinParamDescriptor {
        name: "fs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Sampling frequency for output frequencies in Hz.",
    },
];

const FREQZ_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "H = freqz(b, a)",
        inputs: &FREQZ_INPUTS_CORE,
        outputs: &FREQZ_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "H = freqz(b, a, n)",
        inputs: &FREQZ_INPUTS_N,
        outputs: &FREQZ_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "H = freqz(b, a, n, fs)",
        inputs: &FREQZ_INPUTS_N_FS,
        outputs: &FREQZ_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "[H, w] = freqz(b, a)",
        inputs: &FREQZ_INPUTS_CORE,
        outputs: &FREQZ_OUTPUT_H_W,
    },
    BuiltinSignatureDescriptor {
        label: "[H, w] = freqz(b, a, n)",
        inputs: &FREQZ_INPUTS_N,
        outputs: &FREQZ_OUTPUT_H_W,
    },
    BuiltinSignatureDescriptor {
        label: "[H, w] = freqz(b, a, n, fs)",
        inputs: &FREQZ_INPUTS_N_FS,
        outputs: &FREQZ_OUTPUT_H_W,
    },
];

const FREQZ_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.ARG_COUNT",
    identifier: Some("RunMat:freqz:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "freqz: expected freqz(b, a, [n, [fs]])",
};

const FREQZ_ERROR_INVALID_COEFFICIENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INVALID_COEFFICIENTS",
    identifier: Some("RunMat:freqz:InvalidCoefficients"),
    when: "Coefficient inputs are empty or not numeric vectors.",
    message: "freqz: invalid coefficient input",
};

const FREQZ_ERROR_INVALID_N: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INVALID_N",
    identifier: Some("RunMat:freqz:InvalidN"),
    when: "The response length is not an integer scalar greater than or equal to two.",
    message: "freqz: n must be an integer greater than or equal to 2",
};

const FREQZ_ERROR_INVALID_FS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INVALID_FS",
    identifier: Some("RunMat:freqz:InvalidFs"),
    when: "The sampling frequency is not a positive finite scalar.",
    message: "freqz: fs must be a positive finite scalar",
};

const FREQZ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FREQZ.INTERNAL",
    identifier: Some("RunMat:freqz:Internal"),
    when: "Response tensor construction fails internally.",
    message: "freqz: internal error",
};

const FREQZ_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FREQZ_ERROR_ARG_COUNT,
    FREQZ_ERROR_INVALID_COEFFICIENTS,
    FREQZ_ERROR_INVALID_N,
    FREQZ_ERROR_INVALID_FS,
    FREQZ_ERROR_INTERNAL,
];

macro_rules! freqz_extension {
    ($name:ident, $id:literal, $description:literal, $error:literal) => {
        const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $description,
            error_identifier: Some($error),
        };
    };
}

freqz_extension!(
    FREQZ_INTEGER_COEFFICIENT_EXTENSION,
    "freqz-integer-coefficients",
    "freqz with typed-integer transfer-function coefficients is a RunMat extension",
    "RunMat:compatibility:FreqzIntegerCoefficientExtension"
);
freqz_extension!(
    FREQZ_LOGICAL_COEFFICIENT_EXTENSION,
    "freqz-logical-coefficients",
    "freqz with logical transfer-function coefficients is a RunMat extension",
    "RunMat:compatibility:FreqzLogicalCoefficientExtension"
);
freqz_extension!(FREQZ_INTEGER_N_EXTENSION, "freqz-integer-point-count", "freqz with a typed-integer point count is a provisional RunMat extension because the public core documentation does not publish its accepted classes", "RunMat:compatibility:FreqzIntegerPointCountExtension");
freqz_extension!(
    FREQZ_LOGICAL_N_EXTENSION,
    "freqz-logical-point-count",
    "freqz with a logical point count is a RunMat extension",
    "RunMat:compatibility:FreqzLogicalPointCountExtension"
);
freqz_extension!(
    FREQZ_SINGLE_N_EXTENSION,
    "freqz-single-point-count",
    "freqz with a single-precision point count is a provisional RunMat extension",
    "RunMat:compatibility:FreqzSinglePointCountExtension"
);
freqz_extension!(
    FREQZ_INTEGER_FS_EXTENSION,
    "freqz-integer-sample-rate",
    "freqz with a typed-integer sample rate is a RunMat extension",
    "RunMat:compatibility:FreqzIntegerSampleRateExtension"
);
freqz_extension!(
    FREQZ_LOGICAL_FS_EXTENSION,
    "freqz-logical-sample-rate",
    "freqz with a logical sample rate is a RunMat extension",
    "RunMat:compatibility:FreqzLogicalSampleRateExtension"
);
freqz_extension!(
    FREQZ_SINGLE_FS_EXTENSION,
    "freqz-single-sample-rate",
    "freqz with a single-precision sample rate is a RunMat extension",
    "RunMat:compatibility:FreqzSingleSampleRateExtension"
);
freqz_extension!(
    FREQZ_RESIDENT_INPUT_EXTENSION,
    "freqz-resident-input",
    "freqz with an interactive resident input is a RunMat extension",
    "RunMat:compatibility:FreqzResidentInputExtension"
);

pub const FREQZ_EXTENSIONS: [BuiltinExtensionDescriptor; 9] = [
    FREQZ_INTEGER_COEFFICIENT_EXTENSION,
    FREQZ_LOGICAL_COEFFICIENT_EXTENSION,
    FREQZ_INTEGER_N_EXTENSION,
    FREQZ_LOGICAL_N_EXTENSION,
    FREQZ_SINGLE_N_EXTENSION,
    FREQZ_INTEGER_FS_EXTENSION,
    FREQZ_LOGICAL_FS_EXTENSION,
    FREQZ_SINGLE_FS_EXTENSION,
    FREQZ_RESIDENT_INPUT_EXTENSION,
];

const FREQZ_INTEGER_COEFFICIENT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "b",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer numerator coefficients must be exactly representable at the binary64 response boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "a",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer denominator coefficients must be exactly representable at the binary64 response boundary.",
    },
];
const FREQZ_INTEGER_N_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Core freqz documents a positive integer scalar but publishes no datatype list; RunMat provisionally gates typed n and parses it exactly.",
    }];
const FREQZ_INTEGER_FS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fs",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented sample-rate class is double; admitted typed values require exact binary64 representation.",
    }];

pub const FREQZ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[h,w] = freqz(integer_b,integer_a,...)",
        inputs: &FREQZ_INTEGER_COEFFICIENT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Authoritative integer coefficients cross one checked binary64 boundary; integer-only responses are complex double and frequencies are double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[h,w] = freqz(b,a,integer_n,...)",
        inputs: &FREQZ_INTEGER_N_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Point count is decoded exactly in every supported integer class, must be at least two, and selects the response shape.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[h,f] = freqz(b,a,n,integer_fs)",
        inputs: &FREQZ_INTEGER_FS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Sample rate crosses one checked binary64 boundary; response precision follows documented coefficient precision and f remains host double.",
    },
];

pub const FREQZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FREQZ_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FREQZ_ERRORS,
};

fn freqz_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    freqz_error_with_message(error.message, error)
}

fn freqz_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    freqz_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn freqz_error_with_message(
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
    name = "freqz",
    category = "math/signal",
    summary = "Evaluate digital filter frequency response.",
    keywords = "freqz,frequency response,filter,FIR,IIR,signal processing",
    type_resolver(freqz_type),
    descriptor(crate::builtins::math::signal::freqz::FREQZ_DESCRIPTOR),
    extensions(crate::builtins::math::signal::freqz::FREQZ_EXTENSIONS),
    integer_capabilities(crate::builtins::math::signal::freqz::FREQZ_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::freqz"
)]
async fn freqz_builtin(b: Value, a: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(b, a, &rest).await
}

pub async fn evaluate(b: Value, a: Value, rest: &[Value]) -> BuiltinResult<Value> {
    if rest.len() > 2 {
        return Err(freqz_error(&FREQZ_ERROR_ARG_COUNT));
    }
    ensure_input_extensions(&b, &a, rest)?;
    let single_response = is_single_value(&b) || is_single_value(&a);
    let b = gather_freqz_value(b).await?;
    let a = gather_freqz_value(a).await?;
    let rest = gather_freqz_values(rest).await?;
    crate::builtins::common::validation::reject_typed_complex_integer(&b, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&a, BUILTIN_NAME)?;
    for value in &rest {
        crate::builtins::common::validation::reject_typed_complex_integer(value, BUILTIN_NAME)?;
    }
    ensure_exact_integer_boundary(&b, "numerator", &FREQZ_ERROR_INVALID_COEFFICIENTS)?;
    ensure_exact_integer_boundary(&a, "denominator", &FREQZ_ERROR_INVALID_COEFFICIENTS)?;
    let b = value_to_complex_vector(BUILTIN_NAME, "numerator", b)
        .await
        .map_err(|err| freqz_error_with_detail(&FREQZ_ERROR_INVALID_COEFFICIENTS, err.message()))?
        .data;
    let a = value_to_complex_vector(BUILTIN_NAME, "denominator", a)
        .await
        .map_err(|err| freqz_error_with_detail(&FREQZ_ERROR_INVALID_COEFFICIENTS, err.message()))?
        .data;
    if b.is_empty() || a.is_empty() {
        return Err(freqz_error_with_detail(
            &FREQZ_ERROR_INVALID_COEFFICIENTS,
            "coefficient vectors cannot be empty",
        ));
    }
    let n = if let Some(value) = rest.first() {
        if is_empty_value(value) {
            512
        } else {
            let parsed = parse_nonnegative_integer(BUILTIN_NAME, "n", value)
                .map_err(|err| freqz_error_with_detail(&FREQZ_ERROR_INVALID_N, err.message()))?;
            if parsed < 2 {
                return Err(freqz_error(&FREQZ_ERROR_INVALID_N));
            }
            parsed
        }
    } else {
        512
    };
    let fs = if let Some(value) = rest.get(1) {
        ensure_exact_integer_boundary(value, "fs", &FREQZ_ERROR_INVALID_FS)?;
        let fs = parse_scalar_f64(BUILTIN_NAME, "fs", value)
            .map_err(|err| freqz_error_with_detail(&FREQZ_ERROR_INVALID_FS, err.message()))?;
        if fs <= 0.0 {
            return Err(freqz_error(&FREQZ_ERROR_INVALID_FS));
        }
        Some(fs)
    } else {
        None
    };

    let eval = evaluate_response(&b, &a, n, fs, single_response)?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.h_value()?]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![eval.h_value()?, eval.w_value()?],
        ));
    }
    eval.h_value()
}

struct FreqzEvaluation {
    h: Vec<Complex<f64>>,
    w: Vec<f64>,
    single_response: bool,
}

impl FreqzEvaluation {
    fn h_value(&self) -> BuiltinResult<Value> {
        let storage = if self.single_response {
            ComplexStorage::F32(self.h.iter().map(|z| (z.re as f32, z.im as f32)).collect())
        } else {
            ComplexStorage::F64(self.h.iter().map(|z| (z.re, z.im)).collect())
        };
        ComplexTensor::from_complex_storage(storage, vec![self.h.len(), 1])
            .map(Value::ComplexTensor)
            .map_err(|e| freqz_error_with_detail(&FREQZ_ERROR_INTERNAL, e))
    }

    fn w_value(&self) -> BuiltinResult<Value> {
        Tensor::new(self.w.clone(), vec![self.w.len(), 1])
            .map(Value::Tensor)
            .map_err(|e| freqz_error_with_detail(&FREQZ_ERROR_INTERNAL, e))
    }
}

fn evaluate_response(
    b: &[Complex<f64>],
    a: &[Complex<f64>],
    n: usize,
    fs: Option<f64>,
    single_response: bool,
) -> BuiltinResult<FreqzEvaluation> {
    let mut h = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for idx in 0..n {
        let omega = std::f64::consts::PI * idx as f64 / n as f64;
        let point = Complex::new(omega.cos(), -omega.sin());
        let numerator = polynomial_in_z_inverse(b, point);
        let denominator = polynomial_in_z_inverse(a, point);
        h.push(numerator / denominator);
        w.push(match fs {
            Some(fs) => fs * idx as f64 / (2.0 * n as f64),
            None => omega,
        });
    }
    Ok(FreqzEvaluation {
        h,
        w,
        single_response,
    })
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn is_single_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn ensure_extension_if(
    condition: bool,
    extension: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if condition {
        crate::compatibility::ensure_builtin_extension_enabled(extension, BUILTIN_NAME)?;
    }
    Ok(())
}

fn ensure_input_extensions(b: &Value, a: &Value, rest: &[Value]) -> BuiltinResult<()> {
    for coefficient in [b, a] {
        ensure_extension_if(
            is_typed_integer_value(coefficient),
            &FREQZ_INTEGER_COEFFICIENT_EXTENSION,
        )?;
        ensure_extension_if(
            is_logical_value(coefficient),
            &FREQZ_LOGICAL_COEFFICIENT_EXTENSION,
        )?;
    }
    if let Some(n) = rest.first().filter(|value| !is_empty_value(value)) {
        ensure_extension_if(is_typed_integer_value(n), &FREQZ_INTEGER_N_EXTENSION)?;
        ensure_extension_if(is_logical_value(n), &FREQZ_LOGICAL_N_EXTENSION)?;
        ensure_extension_if(is_single_value(n), &FREQZ_SINGLE_N_EXTENSION)?;
    }
    if let Some(fs) = rest.get(1) {
        ensure_extension_if(is_typed_integer_value(fs), &FREQZ_INTEGER_FS_EXTENSION)?;
        ensure_extension_if(is_logical_value(fs), &FREQZ_LOGICAL_FS_EXTENSION)?;
        ensure_extension_if(is_single_value(fs), &FREQZ_SINGLE_FS_EXTENSION)?;
    }
    ensure_extension_if(
        std::iter::once(b)
            .chain(std::iter::once(a))
            .chain(rest.iter())
            .any(|value| matches!(value, Value::GpuTensor(_))),
        &FREQZ_RESIDENT_INPUT_EXTENSION,
    )
}

async fn gather_freqz_value(value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::builtins::common::gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|err| freqz_error_with_detail(&FREQZ_ERROR_INTERNAL, err.message()))
    } else {
        Ok(value)
    }
}

async fn gather_freqz_values(values: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(values.len());
    for value in values {
        gathered.push(gather_freqz_value(value.clone()).await?);
    }
    Ok(gathered)
}

fn is_empty_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.len() == 0)
}

fn ensure_exact_integer_boundary(
    value: &Value,
    role: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<()> {
    let exact = crate::builtins::math::trigonometry::cos::integer_is_exact_f64;
    let representable = match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        _ => true,
    };
    if representable {
        Ok(())
    } else {
        Err(freqz_error_with_detail(
            error,
            format!("integer {role} values must be exactly representable as double"),
        ))
    }
}

fn polynomial_in_z_inverse(coeffs: &[Complex<f64>], point: Complex<f64>) -> Complex<f64> {
    let mut acc = Complex::new(0.0, 0.0);
    let mut power = Complex::new(1.0, 0.0);
    for coeff in coeffs {
        acc += *coeff * power;
        power *= point;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, IntValue, IntegerStorage};

    fn call(b: Value, a: Value, rest: &[Value], outputs: Option<usize>) -> BuiltinResult<Value> {
        let _guard = outputs.map(|count| crate::output_count::push_output_count(Some(count)));
        block_on(evaluate(b, a, rest))
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("freqz builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| sig.label == "[H, w] = freqz(b, a, n, fs)"));
        assert_eq!(FREQZ_INTEGER_CAPABILITIES.len(), 3);
        assert!(FREQZ_EXTENSIONS
            .iter()
            .any(|extension| extension.id == "freqz-resident-input"));
    }

    #[test]
    fn simple_fir_response_matches_closed_form() {
        let h = call(
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
            &[Value::Num(4.0)],
            None,
        )
        .unwrap();
        let Value::ComplexTensor(h) = h else {
            panic!("expected complex response");
        };
        assert_eq!(h.shape, vec![4, 1]);
        assert!((h.materialize_f64()[0].0 - 2.0).abs() < 1e-12);
        assert!((h.materialize_f64()[2].0 - 1.0).abs() < 1e-12);
        assert!((h.materialize_f64()[2].1 + 1.0).abs() < 1e-12);
    }

    #[test]
    fn iir_response_and_frequency_outputs() {
        let out = call(
            Value::Num(0.2),
            Value::Tensor(Tensor::new(vec![1.0, -0.8], vec![1, 2]).unwrap()),
            &[Value::Num(8.0), Value::Num(1000.0)],
            Some(2),
        )
        .unwrap();
        let Value::OutputList(values) = out else {
            panic!("expected output list");
        };
        let Value::ComplexTensor(h) = &values[0] else {
            panic!("expected H");
        };
        let Value::Tensor(w) = &values[1] else {
            panic!("expected w");
        };
        assert!((h.materialize_f64()[0].0 - 1.0).abs() < 1e-12);
        assert_eq!(w.shape, vec![8, 1]);
        assert!((w.materialize_f64()[1] - 62.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_n_and_empty_coefficients() {
        assert!(call(Value::Num(1.0), Value::Num(1.0), &[Value::Num(1.0)], None).is_err());
        assert!(call(Value::Num(1.0), Value::Num(1.0), &[Value::Num(0.0)], None).is_err());
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        assert!(call(Value::Tensor(empty), Value::Num(1.0), &[], None).is_err());
    }

    #[test]
    fn empty_point_count_uses_default_response_length() {
        let empty = Tensor::new(Vec::new(), vec![0, 0]).expect("empty point count");
        let output = call(
            Value::Num(1.0),
            Value::Num(1.0),
            &[Value::Tensor(empty)],
            None,
        )
        .expect("empty n defaults");
        let Value::ComplexTensor(output) = output else {
            panic!("expected response");
        };
        assert_eq!(output.shape, vec![512, 1]);
    }

    #[test]
    fn single_coefficients_produce_single_complex_response() {
        let numerator = Tensor::from_f32(vec![1.0, 1.0], vec![1, 2]).expect("single b");
        let output = call(
            Value::Tensor(numerator),
            Value::Num(1.0),
            &[Value::Num(4.0)],
            None,
        )
        .expect("single response");
        let Value::ComplexTensor(output) = output else {
            panic!("expected complex response");
        };
        assert!(matches!(output.complex_storage(), ComplexStorage::F32(_)));
    }

    #[test]
    fn integer_coefficient_count_and_sample_rate_extensions_are_independently_gated() {
        let cases = [
            (
                Value::Int(IntValue::U8(1)),
                Value::Num(1.0),
                Vec::new(),
                "RunMat:compatibility:FreqzIntegerCoefficientExtension",
            ),
            (
                Value::Num(1.0),
                Value::Num(1.0),
                vec![Value::Int(IntValue::U16(4))],
                "RunMat:compatibility:FreqzIntegerPointCountExtension",
            ),
            (
                Value::Num(1.0),
                Value::Num(1.0),
                vec![Value::Num(4.0), Value::Int(IntValue::U32(1000))],
                "RunMat:compatibility:FreqzIntegerSampleRateExtension",
            ),
        ];
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (b, a, options, identifier) in cases {
            let error = call(b, a, &options, None).expect_err("integer role must be gated");
            assert_eq!(error.identifier(), Some(identifier));
        }
    }

    #[test]
    fn integer_coefficients_cross_one_checked_double_boundary() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let numerator = Tensor::new_integer(IntegerStorage::I16(vec![1, 1]), vec![1, 2])
            .expect("integer numerator");
        let output = call(
            Value::Tensor(numerator),
            Value::Int(IntValue::U8(1)),
            &[Value::Int(IntValue::U16(4))],
            None,
        )
        .expect("integer response");
        let Value::ComplexTensor(output) = output else {
            panic!("expected complex response");
        };
        assert!(matches!(output.complex_storage(), ComplexStorage::F64(_)));

        let wide =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("wide numerator");
        let error = call(
            Value::Tensor(wide),
            Value::Num(1.0),
            &[Value::Num(4.0)],
            None,
        )
        .expect_err("lossy coefficient conversion must reject");
        assert!(error.message().contains("exactly representable as double"));
    }

    #[test]
    fn logical_and_single_controls_have_distinct_compatibility_gates() {
        let single_n = Tensor::from_f32(vec![4.0], vec![1, 1]).expect("single n");
        let cases = [
            (
                vec![Value::Bool(true)],
                "RunMat:compatibility:FreqzLogicalPointCountExtension",
            ),
            (
                vec![Value::Tensor(single_n)],
                "RunMat:compatibility:FreqzSinglePointCountExtension",
            ),
            (
                vec![Value::Num(4.0), Value::Bool(true)],
                "RunMat:compatibility:FreqzLogicalSampleRateExtension",
            ),
        ];
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (options, identifier) in cases {
            let error = call(Value::Num(1.0), Value::Num(1.0), &options, None)
                .expect_err("control role must be gated");
            assert_eq!(error.identifier(), Some(identifier));
        }
    }

    #[test]
    fn resident_extension_rejects_before_provider_access() {
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
            descriptor: Default::default(),
        });
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call(resident, Value::Num(1.0), &[], None)
            .expect_err("resident call must reject before gather");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FreqzResidentInputExtension")
        );
    }
}
