//! Focused MATLAB-compatible `fir1` FIR-window design.

use num_complex::Complex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::signal::common::{
    keyword, parse_nonnegative_integer, real_vector_to_row_value, value_to_complex_vector,
};
use crate::builtins::math::signal::type_resolvers::fir1_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "fir1";
const EPS: f64 = 1.0e-12;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::fir1")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fir1",
    op_kind: GpuOpKind::Custom("fir-design"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The current implementation gathers resident numeric arguments and returns host double coefficients; restoring documented GPU output residency remains a tracked compatibility gap.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::fir1")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fir1",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "fir1 materialises coefficient vectors and is not fused.",
};

const FIR1_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "FIR numerator coefficient row vector.",
}];

const FIR1_INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized cutoff frequency or two-element band.",
    },
];

const FIR1_INPUTS_OPTIONS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized cutoff frequency or two-element band.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Filter type, window vector, or scale/noscale option.",
    },
];

const FIR1_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "b = fir1(n, Wn)",
        inputs: &FIR1_INPUTS_CORE,
        outputs: &FIR1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "b = fir1(n, Wn, ftype)",
        inputs: &FIR1_INPUTS_OPTIONS,
        outputs: &FIR1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "b = fir1(n, Wn, ftype, window, scaleopt)",
        inputs: &FIR1_INPUTS_OPTIONS,
        outputs: &FIR1_OUTPUT,
    },
];

const FIR1_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.ARG_COUNT",
    identifier: Some("RunMat:fir1:ArgCount"),
    when: "Fewer than two arguments are supplied.",
    message: "fir1: expected fir1(n, Wn, ...)",
};

const FIR1_ERROR_INVALID_ORDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INVALID_ORDER",
    identifier: Some("RunMat:fir1:InvalidOrder"),
    when: "The order is not a finite nonnegative integer scalar.",
    message: "fir1: filter order must be a nonnegative integer",
};

const FIR1_ERROR_INVALID_FREQUENCY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INVALID_FREQUENCY",
    identifier: Some("RunMat:fir1:InvalidFrequency"),
    when: "Wn is not a valid normalized scalar or increasing two-element band.",
    message: "fir1: invalid cutoff frequency",
};

const FIR1_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INVALID_OPTION",
    identifier: Some("RunMat:fir1:InvalidOption"),
    when: "A filter type, window vector, or scaling option is unsupported.",
    message: "fir1: invalid option",
};

const FIR1_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIR1.INTERNAL",
    identifier: Some("RunMat:fir1:Internal"),
    when: "Coefficient materialization fails internally.",
    message: "fir1: internal error",
};

const FIR1_ERRORS: [BuiltinErrorDescriptor; 5] = [
    FIR1_ERROR_ARG_COUNT,
    FIR1_ERROR_INVALID_ORDER,
    FIR1_ERROR_INVALID_FREQUENCY,
    FIR1_ERROR_INVALID_OPTION,
    FIR1_ERROR_INTERNAL,
];

macro_rules! fir1_extension {
    ($name:ident, $id:literal, $description:literal, $error:literal) => {
        const $name: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
            id: $id,
            mode: BuiltinExtensionMode::RunMatOnly,
            description: $description,
            error_identifier: Some($error),
        };
    };
}

fir1_extension!(
    FIR1_INTEGER_ORDER_EXTENSION,
    "fir1-integer-order",
    "fir1 with a typed-integer filter order is a RunMat extension",
    "RunMat:compatibility:Fir1IntegerOrderExtension"
);
fir1_extension!(
    FIR1_LOGICAL_ORDER_EXTENSION,
    "fir1-logical-order",
    "fir1 with a logical filter order is a RunMat extension",
    "RunMat:compatibility:Fir1LogicalOrderExtension"
);
fir1_extension!(
    FIR1_SINGLE_ORDER_EXTENSION,
    "fir1-single-order",
    "fir1 with a single-precision filter order is a RunMat extension",
    "RunMat:compatibility:Fir1SingleOrderExtension"
);
fir1_extension!(
    FIR1_INTEGER_CUTOFF_EXTENSION,
    "fir1-integer-cutoff",
    "fir1 with typed-integer cutoff frequencies is a RunMat extension",
    "RunMat:compatibility:Fir1IntegerCutoffExtension"
);
fir1_extension!(
    FIR1_LOGICAL_CUTOFF_EXTENSION,
    "fir1-logical-cutoff",
    "fir1 with logical cutoff frequencies is a RunMat extension",
    "RunMat:compatibility:Fir1LogicalCutoffExtension"
);
fir1_extension!(
    FIR1_SINGLE_CUTOFF_EXTENSION,
    "fir1-single-cutoff",
    "fir1 with single-precision cutoff frequencies is a RunMat extension",
    "RunMat:compatibility:Fir1SingleCutoffExtension"
);
fir1_extension!(
    FIR1_INTEGER_WINDOW_EXTENSION,
    "fir1-integer-window",
    "fir1 with a typed-integer window is a RunMat extension",
    "RunMat:compatibility:Fir1IntegerWindowExtension"
);
fir1_extension!(
    FIR1_LOGICAL_WINDOW_EXTENSION,
    "fir1-logical-window",
    "fir1 with a logical window is a RunMat extension",
    "RunMat:compatibility:Fir1LogicalWindowExtension"
);
fir1_extension!(
    FIR1_SINGLE_WINDOW_EXTENSION,
    "fir1-single-window",
    "fir1 with a single-precision window is a RunMat extension",
    "RunMat:compatibility:Fir1SingleWindowExtension"
);

pub const FIR1_EXTENSIONS: [BuiltinExtensionDescriptor; 9] = [
    FIR1_INTEGER_ORDER_EXTENSION,
    FIR1_LOGICAL_ORDER_EXTENSION,
    FIR1_SINGLE_ORDER_EXTENSION,
    FIR1_INTEGER_CUTOFF_EXTENSION,
    FIR1_LOGICAL_CUTOFF_EXTENSION,
    FIR1_SINGLE_CUTOFF_EXTENSION,
    FIR1_INTEGER_WINDOW_EXTENSION,
    FIR1_LOGICAL_WINDOW_EXTENSION,
    FIR1_SINGLE_WINDOW_EXTENSION,
];

const FIR1_INTEGER_ORDER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented order class is double; RunMat mode parses every typed-integer scalar exactly as a structural filter order.",
    }];
const FIR1_INTEGER_CUTOFF_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Wn",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed cutoff storage is admitted only in RunMat mode and must be exactly representable at the binary64 FIR-design boundary.",
    }];
const FIR1_INTEGER_WINDOW_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "window",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed window values remain authoritative through admission and must be exactly representable before binary64 window multiplication.",
    }];

pub const FIR1_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "b = fir1(integer_n,Wn,...)",
        inputs: &FIR1_INTEGER_ORDER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The order is decoded without an f64 mirror; resident order input gathers through its owner after compatibility admission, and coefficient output remains a host double row vector.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "b = fir1(n,integer_Wn,...)",
        inputs: &FIR1_INTEGER_CUTOFF_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer cutoff values cross one checked binary64 boundary before normalized-frequency validation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "b = fir1(n,Wn,integer_window,...)",
        inputs: &FIR1_INTEGER_WINDOW_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer window values cross one checked binary64 boundary before coefficient multiplication.",
    },
];

pub const FIR1_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIR1_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FIR1_ERRORS,
};

fn fir1_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fir1_error_with_message(error.message, error)
}

fn fir1_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fir1_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fir1_error_with_message(
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
    name = "fir1",
    category = "math/signal",
    summary = "Design windowed-sinc FIR filters.",
    keywords = "fir1,FIR,windowed sinc,lowpass,highpass,bandpass,bandstop,signal processing",
    type_resolver(fir1_type),
    descriptor(crate::builtins::math::signal::fir1::FIR1_DESCRIPTOR),
    extensions(crate::builtins::math::signal::fir1::FIR1_EXTENSIONS),
    integer_capabilities(crate::builtins::math::signal::fir1::FIR1_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::fir1"
)]
async fn fir1_builtin(n: Value, wn: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    evaluate(n, wn, &rest).await
}

pub async fn evaluate(n: Value, wn: Value, rest: &[Value]) -> BuiltinResult<Value> {
    ensure_input_extensions(&n, &wn, rest)?;
    let n = gather_signal_value(n).await?;
    let requested_order = parse_nonnegative_integer(BUILTIN_NAME, "order", &n)
        .map_err(|err| fir1_error_with_detail(&FIR1_ERROR_INVALID_ORDER, err.message()))?;
    let cutoff = parse_cutoff(wn).await?;
    let mut options = Fir1Options::default_for_cutoff(&cutoff);
    let mut window: Option<Value> = None;

    for arg in rest {
        if let Some(word) = keyword(arg) {
            match word.as_str() {
                "low" | "lowpass" => options.kind = FilterKind::Lowpass,
                "high" | "highpass" => options.kind = FilterKind::Highpass,
                "bandpass" | "pass" => options.kind = FilterKind::Bandpass,
                "stop" | "bandstop" | "bandreject" => options.kind = FilterKind::Bandstop,
                "scale" => options.scale = true,
                "noscale" => options.scale = false,
                _ => {
                    return Err(fir1_error_with_detail(
                        &FIR1_ERROR_INVALID_OPTION,
                        format!("unknown option '{word}'"),
                    ))
                }
            }
        } else {
            if window.is_some() {
                return Err(fir1_error_with_detail(
                    &FIR1_ERROR_INVALID_OPTION,
                    "multiple window vectors supplied",
                ));
            }
            window = Some(arg.clone());
        }
    }

    validate_kind_cutoff(options.kind, &cutoff)?;
    let order = adjusted_order(requested_order, options.kind);
    let window = if let Some(window) = window {
        parse_window(window, order + 1).await?
    } else {
        hamming_window(order + 1)
    };
    let mut coeffs = ideal_impulse_response(order, &cutoff, options.kind);
    for (coeff, win) in coeffs.iter_mut().zip(window.iter()) {
        *coeff *= *win;
    }
    if options.scale {
        scale_coefficients(&mut coeffs, &cutoff, options.kind)?;
    }
    real_vector_to_row_value(coeffs)
        .map_err(|err| fir1_error_with_detail(&FIR1_ERROR_INTERNAL, err.message()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FilterKind {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

#[derive(Clone, Copy)]
struct Fir1Options {
    kind: FilterKind,
    scale: bool,
}

impl Fir1Options {
    fn default_for_cutoff(cutoff: &[f64]) -> Self {
        Self {
            kind: if cutoff.len() == 1 {
                FilterKind::Lowpass
            } else {
                FilterKind::Bandpass
            },
            scale: true,
        }
    }
}

async fn parse_cutoff(value: Value) -> BuiltinResult<Vec<f64>> {
    let value = gather_signal_value(value).await?;
    ensure_exact_integer_boundary(&value, "Wn", &FIR1_ERROR_INVALID_FREQUENCY)?;
    let input = value_to_complex_vector(BUILTIN_NAME, "Wn", value)
        .await
        .map_err(|err| fir1_error_with_detail(&FIR1_ERROR_INVALID_FREQUENCY, err.message()))?;
    if input.data.is_empty() || input.data.len() > 2 || input.data.iter().any(|z| z.im.abs() > EPS)
    {
        return Err(fir1_error(&FIR1_ERROR_INVALID_FREQUENCY));
    }
    let cutoff = input.data.iter().map(|z| z.re).collect::<Vec<_>>();
    if cutoff
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0 || *value >= 1.0)
        || (cutoff.len() == 2 && cutoff[0] >= cutoff[1])
    {
        return Err(fir1_error(&FIR1_ERROR_INVALID_FREQUENCY));
    }
    Ok(cutoff)
}

async fn parse_window(value: Value, expected_len: usize) -> BuiltinResult<Vec<f64>> {
    let value = gather_signal_value(value).await?;
    ensure_exact_integer_boundary(&value, "window", &FIR1_ERROR_INVALID_OPTION)?;
    let input = value_to_complex_vector(BUILTIN_NAME, "window", value)
        .await
        .map_err(|err| fir1_error_with_detail(&FIR1_ERROR_INVALID_OPTION, err.message()))?;
    if input.data.len() != expected_len || input.data.iter().any(|z| z.im.abs() > EPS) {
        return Err(fir1_error_with_detail(
            &FIR1_ERROR_INVALID_OPTION,
            format!("window must be a real vector of length {expected_len}"),
        ));
    }
    Ok(input.data.iter().map(|z| z.re).collect())
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
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32))
}

fn ensure_role_extensions(
    value: &Value,
    integer: &'static BuiltinExtensionDescriptor,
    logical: &'static BuiltinExtensionDescriptor,
    single: &'static BuiltinExtensionDescriptor,
) -> BuiltinResult<()> {
    if is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(integer, BUILTIN_NAME)?;
    }
    if is_logical_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(logical, BUILTIN_NAME)?;
    }
    if is_single_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(single, BUILTIN_NAME)?;
    }
    Ok(())
}

fn ensure_input_extensions(n: &Value, wn: &Value, rest: &[Value]) -> BuiltinResult<()> {
    ensure_role_extensions(
        n,
        &FIR1_INTEGER_ORDER_EXTENSION,
        &FIR1_LOGICAL_ORDER_EXTENSION,
        &FIR1_SINGLE_ORDER_EXTENSION,
    )?;
    ensure_role_extensions(
        wn,
        &FIR1_INTEGER_CUTOFF_EXTENSION,
        &FIR1_LOGICAL_CUTOFF_EXTENSION,
        &FIR1_SINGLE_CUTOFF_EXTENSION,
    )?;
    for value in rest.iter().filter(|value| keyword(value).is_none()) {
        ensure_role_extensions(
            value,
            &FIR1_INTEGER_WINDOW_EXTENSION,
            &FIR1_LOGICAL_WINDOW_EXTENSION,
            &FIR1_SINGLE_WINDOW_EXTENSION,
        )?;
    }
    Ok(())
}

async fn gather_signal_value(value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::builtins::common::gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|err| fir1_error_with_detail(&FIR1_ERROR_INTERNAL, err.message()))
    } else {
        Ok(value)
    }
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
        Err(fir1_error_with_detail(
            error,
            format!("integer {role} values must be exactly representable as double"),
        ))
    }
}

fn validate_kind_cutoff(kind: FilterKind, cutoff: &[f64]) -> BuiltinResult<()> {
    match (kind, cutoff.len()) {
        (FilterKind::Lowpass | FilterKind::Highpass, 1)
        | (FilterKind::Bandpass | FilterKind::Bandstop, 2) => Ok(()),
        _ => Err(fir1_error_with_detail(
            &FIR1_ERROR_INVALID_OPTION,
            "filter type is incompatible with Wn shape",
        )),
    }
}

fn adjusted_order(order: usize, kind: FilterKind) -> usize {
    match kind {
        FilterKind::Highpass | FilterKind::Bandstop if order % 2 == 1 => order + 1,
        _ => order,
    }
}

fn hamming_window(len: usize) -> Vec<f64> {
    match len {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => (0..len)
            .map(|idx| {
                let phase = 2.0 * std::f64::consts::PI * idx as f64 / (len - 1) as f64;
                0.54 - 0.46 * phase.cos()
            })
            .collect(),
    }
}

fn ideal_impulse_response(order: usize, cutoff: &[f64], kind: FilterKind) -> Vec<f64> {
    let center = order as f64 / 2.0;
    (0..=order)
        .map(|idx| {
            let m = idx as f64 - center;
            match kind {
                FilterKind::Lowpass => lowpass_sample(cutoff[0], m),
                FilterKind::Highpass => delta(m) - lowpass_sample(cutoff[0], m),
                FilterKind::Bandpass => lowpass_sample(cutoff[1], m) - lowpass_sample(cutoff[0], m),
                FilterKind::Bandstop => {
                    delta(m) - (lowpass_sample(cutoff[1], m) - lowpass_sample(cutoff[0], m))
                }
            }
        })
        .collect()
}

fn lowpass_sample(cutoff: f64, m: f64) -> f64 {
    cutoff * sinc(cutoff * m)
}

fn sinc(x: f64) -> f64 {
    if x.abs() <= EPS {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

fn delta(m: f64) -> f64 {
    if m.abs() <= EPS {
        1.0
    } else {
        0.0
    }
}

fn scale_coefficients(coeffs: &mut [f64], cutoff: &[f64], kind: FilterKind) -> BuiltinResult<()> {
    let omega = match kind {
        FilterKind::Lowpass | FilterKind::Bandstop => 0.0,
        FilterKind::Highpass => std::f64::consts::PI,
        FilterKind::Bandpass => std::f64::consts::PI * (cutoff[0] + cutoff[1]) / 2.0,
    };
    let response = frequency_response(coeffs, omega).norm();
    if response <= EPS || !response.is_finite() {
        return Err(fir1_error_with_detail(
            &FIR1_ERROR_INVALID_OPTION,
            "cannot scale filter with near-zero passband response",
        ));
    }
    for coeff in coeffs {
        *coeff /= response;
    }
    Ok(())
}

fn frequency_response(coeffs: &[f64], omega: f64) -> Complex<f64> {
    coeffs
        .iter()
        .enumerate()
        .fold(Complex::new(0.0, 0.0), |acc, (idx, coeff)| {
            let phase = -omega * idx as f64;
            acc + Complex::new(phase.cos(), phase.sin()) * *coeff
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    fn call(n: Value, wn: Value, rest: &[Value]) -> BuiltinResult<Value> {
        block_on(evaluate(n, wn, rest))
    }

    fn tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("fir1 builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| sig.label == "b = fir1(n, Wn)"));
        assert_eq!(FIR1_INTEGER_CAPABILITIES.len(), 3);
        assert!(FIR1_EXTENSIONS
            .iter()
            .any(|extension| extension.id == "fir1-integer-window"));
    }

    #[test]
    fn lowpass_has_expected_shape_symmetry_and_dc_gain() {
        let out = tensor(call(Value::Num(10.0), Value::Num(0.25), &[]).unwrap());
        assert_eq!(out.shape, vec![1, 11]);
        for idx in 0..out.materialize_f64().len() {
            let mirror = out.materialize_f64().len() - 1 - idx;
            assert!((out.materialize_f64()[idx] - out.materialize_f64()[mirror]).abs() < 1e-12);
        }
        let sum: f64 = out.materialize_f64().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn highpass_scales_at_nyquist() {
        let out = tensor(call(Value::Num(10.0), Value::Num(0.35), &[Value::from("high")]).unwrap());
        let response = frequency_response(&out.materialize_f64(), std::f64::consts::PI).norm();
        assert!((response - 1.0).abs() < 1e-10);
    }

    #[test]
    fn odd_order_highpass_is_adjusted_to_even_order() {
        let out = tensor(call(Value::Num(5.0), Value::Num(0.35), &[Value::from("high")]).unwrap());
        assert_eq!(out.shape, vec![1, 7]);
        let response = frequency_response(&out.materialize_f64(), std::f64::consts::PI).norm();
        assert!((response - 1.0).abs() < 1e-10);
    }

    #[test]
    fn bandpass_and_stop_accept_two_element_cutoff() {
        let wn = Tensor::new(vec![0.2, 0.4], vec![1, 2]).unwrap();
        let bandpass = tensor(call(Value::Num(20.0), Value::Tensor(wn.clone()), &[]).unwrap());
        assert_eq!(bandpass.shape, vec![1, 21]);

        let stop =
            tensor(call(Value::Num(20.0), Value::Tensor(wn), &[Value::from("stop")]).unwrap());
        let dc: f64 = stop.materialize_f64().iter().sum();
        assert!((dc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn custom_window_and_noscale_are_supported() {
        let window = Tensor::new(vec![1.0; 5], vec![1, 5]).unwrap();
        let out = tensor(
            call(
                Value::Num(4.0),
                Value::Num(0.4),
                &[Value::Tensor(window), Value::from("noscale")],
            )
            .unwrap(),
        );
        assert_eq!(out.shape, vec![1, 5]);
        assert!((out.materialize_f64().iter().sum::<f64>() - 1.0).abs() > 1e-3);
    }

    #[test]
    fn rejects_invalid_cutoff_and_window() {
        assert!(call(Value::Num(4.0), Value::Num(1.2), &[]).is_err());
        let window = Tensor::new(vec![1.0; 4], vec![1, 4]).unwrap();
        assert!(call(Value::Num(4.0), Value::Num(0.4), &[Value::Tensor(window)]).is_err());
    }

    #[test]
    fn integer_order_cutoff_and_window_extensions_are_independently_gated() {
        let integer_window = || {
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![1; 5]), vec![1, 5])
                    .expect("integer window"),
            )
        };
        let cases = [
            (
                Value::Int(IntValue::U8(4)),
                Value::Num(0.4),
                Vec::new(),
                "RunMat:compatibility:Fir1IntegerOrderExtension",
            ),
            (
                Value::Num(4.0),
                Value::Int(IntValue::U8(0)),
                Vec::new(),
                "RunMat:compatibility:Fir1IntegerCutoffExtension",
            ),
            (
                Value::Num(4.0),
                Value::Num(0.4),
                vec![integer_window()],
                "RunMat:compatibility:Fir1IntegerWindowExtension",
            ),
        ];
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (order, cutoff, options, identifier) in cases {
            let error = call(order, cutoff, &options).expect_err("integer role must be gated");
            assert_eq!(error.identifier(), Some(identifier));
        }
    }

    #[test]
    fn integer_order_stays_structural_and_integer_window_has_checked_double_boundary() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for order in [
            IntValue::I8(4),
            IntValue::I16(4),
            IntValue::I32(4),
            IntValue::I64(4),
            IntValue::U8(4),
            IntValue::U16(4),
            IntValue::U32(4),
            IntValue::U64(4),
        ] {
            let output = tensor(
                call(Value::Int(order), Value::Num(0.4), &[]).expect("all integer order classes"),
            );
            assert_eq!(output.shape, vec![1, 5]);
        }
        let window = Tensor::new_integer(IntegerStorage::I16(vec![1; 5]), vec![1, 5])
            .expect("integer window");
        let output = tensor(
            call(
                Value::Int(IntValue::U32(4)),
                Value::Num(0.4),
                &[Value::Tensor(window), Value::from("noscale")],
            )
            .expect("integer roles"),
        );
        assert_eq!(output.shape, vec![1, 5]);
        assert_eq!(output.numeric_dtype(), NumericDType::F64);

        let wide =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("wide integer window");
        let error = call(
            Value::Num(0.0),
            Value::Num(0.4),
            &[Value::Tensor(wide), Value::from("noscale")],
        )
        .expect_err("lossy window conversion must reject");
        assert!(error.message().contains("exactly representable as double"));
    }

    #[test]
    fn logical_and_single_roles_have_distinct_compatibility_gates() {
        let single = Tensor::from_f32(vec![4.0], vec![1, 1]).expect("single order");
        let cases = [
            (
                Value::Bool(true),
                "RunMat:compatibility:Fir1LogicalOrderExtension",
            ),
            (
                Value::Tensor(single),
                "RunMat:compatibility:Fir1SingleOrderExtension",
            ),
        ];
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (order, identifier) in cases {
            let error = call(order, Value::Num(0.4), &[]).expect_err("role must be gated");
            assert_eq!(error.identifier(), Some(identifier));
        }
    }
}
