//! MATLAB-compatible `buttord` Butterworth order selection.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::signal::common::real_vector_to_row_value;
use crate::builtins::math::signal::order_selection::{
    checked_order, classify_kind, db_excess, db_excess_log_ratio, find_natural_frequency,
    option_text, parse_positive_db, real_edges, transform_edges, unwarp_frequency,
    validate_critical_frequencies, FilterKind, OrderFamily,
};
use crate::builtins::math::signal::type_resolvers::buttord_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "buttord";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::buttord")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "buttord",
    op_kind: GpuOpKind::Custom("butterworth-order-selection"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Butterworth order selection is scalar host-side analysis; GPU inputs are gathered automatically.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::buttord")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "buttord",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "buttord materialises scalar/vector design parameters and is not fused.",
};

const BUTTORD_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum Butterworth filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Natural cutoff frequency or frequency pair for butter.",
    },
];

const BUTTORD_INPUTS_DIGITAL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Wp",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Passband edge or two-element passband edge vector.",
    },
    BuiltinParamDescriptor {
        name: "Ws",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Stopband edge or two-element stopband edge vector.",
    },
    BuiltinParamDescriptor {
        name: "Rp",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum passband ripple in dB.",
    },
    BuiltinParamDescriptor {
        name: "Rs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum stopband attenuation in dB.",
    },
];

const BUTTORD_INPUTS_ANALOG: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "Wp",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog passband edge or edge pair in rad/s.",
    },
    BuiltinParamDescriptor {
        name: "Ws",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog stopband edge or edge pair in rad/s.",
    },
    BuiltinParamDescriptor {
        name: "Rp",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum passband ripple in dB.",
    },
    BuiltinParamDescriptor {
        name: "Rs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum stopband attenuation in dB.",
    },
    BuiltinParamDescriptor {
        name: "s",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog-design selector.",
    },
];

const BUTTORD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[n, Wn] = buttord(Wp, Ws, Rp, Rs)",
        inputs: &BUTTORD_INPUTS_DIGITAL,
        outputs: &BUTTORD_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[n, Wn] = buttord(Wp, Ws, Rp, Rs, 's')",
        inputs: &BUTTORD_INPUTS_ANALOG,
        outputs: &BUTTORD_OUTPUTS,
    },
];

const BUTTORD_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.ARG_COUNT",
    identifier: Some("RunMat:buttord:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "buttord: expected buttord(Wp, Ws, Rp, Rs [, 's'])",
};

const BUTTORD_ERROR_INVALID_FREQUENCY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INVALID_FREQUENCY",
    identifier: Some("RunMat:buttord:InvalidFrequency"),
    when: "Passband or stopband edges are invalid.",
    message: "buttord: invalid passband or stopband frequency",
};

const BUTTORD_ERROR_INVALID_ATTENUATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INVALID_ATTENUATION",
    identifier: Some("RunMat:buttord:InvalidAttenuation"),
    when: "Ripple or attenuation values are invalid.",
    message: "buttord: Rp and Rs must be positive finite scalars with Rs > Rp",
};

const BUTTORD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INVALID_OPTION",
    identifier: Some("RunMat:buttord:InvalidOption"),
    when: "The optional selector is not supported.",
    message: "buttord: optional selector must be 's'",
};

const BUTTORD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:buttord:TooManyOutputs"),
    when: "More than two output arguments are requested.",
    message: "buttord: too many output arguments",
};

const BUTTORD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INTERNAL",
    identifier: Some("RunMat:buttord:Internal"),
    when: "Output tensor construction fails internally.",
    message: "buttord: internal error",
};

const BUTTORD_ERRORS: [BuiltinErrorDescriptor; 6] = [
    BUTTORD_ERROR_ARG_COUNT,
    BUTTORD_ERROR_INVALID_FREQUENCY,
    BUTTORD_ERROR_INVALID_ATTENUATION,
    BUTTORD_ERROR_INVALID_OPTION,
    BUTTORD_ERROR_TOO_MANY_OUTPUTS,
    BUTTORD_ERROR_INTERNAL,
];

const BUTTORD_INTEGER_FREQUENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "buttord-integer-frequency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "buttord with typed-integer passband or stopband edges is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButtordIntegerFrequencyExtension"),
    };
const BUTTORD_LOGICAL_FREQUENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "buttord-logical-frequency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "buttord with logical passband or stopband edges is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButtordLogicalFrequencyExtension"),
    };
const BUTTORD_COMPLEX_FREQUENCY_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "buttord-complex-frequency",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "buttord with complex passband or stopband edges is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButtordComplexFrequencyExtension"),
    };
const BUTTORD_INTEGER_ATTENUATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "buttord-integer-attenuation",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "buttord with typed-integer ripple or attenuation is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButtordIntegerAttenuationExtension"),
    };
const BUTTORD_LOGICAL_ATTENUATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "buttord-logical-attenuation",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "buttord with logical ripple or attenuation is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ButtordLogicalAttenuationExtension"),
    };
const BUTTORD_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "buttord-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "buttord with resident GPU-array design arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ButtordGpuInputExtension"),
};

pub const BUTTORD_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    BUTTORD_INTEGER_FREQUENCY_EXTENSION,
    BUTTORD_LOGICAL_FREQUENCY_EXTENSION,
    BUTTORD_COMPLEX_FREQUENCY_EXTENSION,
    BUTTORD_INTEGER_ATTENUATION_EXTENSION,
    BUTTORD_LOGICAL_ATTENUATION_EXTENSION,
    BUTTORD_GPU_INPUT_EXTENSION,
];

const BUTTORD_INTEGER_FREQUENCY_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Wp",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented passband-edge datatypes are single and double; RunMat mode additionally accepts scalar or two-element typed-integer storage.",
    },
    BuiltinIntegerInputCapability {
        name: "Ws",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented stopband-edge datatypes are single and double; RunMat mode additionally accepts scalar or two-element typed-integer storage.",
    },
];
const BUTTORD_INTEGER_ATTENUATION_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Rp",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented passband-ripple datatypes are single and double; RunMat mode additionally accepts every scalar integer class.",
    },
    BuiltinIntegerInputCapability {
        name: "Rs",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented stopband-attenuation datatypes are single and double; RunMat mode additionally accepts every scalar integer class.",
    },
];

pub const BUTTORD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[n,Wn] = buttord(integer_Wp,integer_Ws,Rp,Rs,...)",
        inputs: &BUTTORD_INTEGER_FREQUENCY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer frequency edges cross the explicit binary64 Butterworth analysis boundary after compatibility and residency checks. Both n and Wn remain host double values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[n,Wn] = buttord(Wp,Ws,integer_Rp,integer_Rs,...)",
        inputs: &BUTTORD_INTEGER_ATTENUATION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer ripple and attenuation controls cross the explicit binary64 decibel-analysis boundary after compatibility and residency checks. Both n and Wn remain host double values.",
    },
];

pub const BUTTORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BUTTORD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BUTTORD_ERRORS,
};

#[derive(Clone, Debug)]
struct ButtordResult {
    order: usize,
    wn: Vec<f64>,
}

fn buttord_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    buttord_error_with_message(error.message, error)
}

fn buttord_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    buttord_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn buttord_error_with_message(
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
    name = "buttord",
    category = "math/signal",
    summary = "Select the minimum Butterworth filter order.",
    keywords = "buttord,butterworth,filter order,signal processing",
    type_resolver(buttord_type),
    descriptor(crate::builtins::math::signal::buttord::BUTTORD_DESCRIPTOR),
    extensions(crate::builtins::math::signal::buttord::BUTTORD_EXTENSIONS),
    integer_capabilities(crate::builtins::math::signal::buttord::BUTTORD_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::signal::buttord"
)]
async fn buttord_builtin(
    wp: Value,
    ws: Value,
    rp: Value,
    rs: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    evaluate(wp, ws, rp, rs, &rest).await
}

pub async fn evaluate(
    wp: Value,
    ws: Value,
    rp: Value,
    rs: Value,
    rest: &[Value],
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(buttord_error(&BUTTORD_ERROR_ARG_COUNT));
    }
    ensure_input_extensions(&wp, &ws, &rp, &rs)?;
    let analog = if let Some(option) = rest.first() {
        match option_text(option).as_deref() {
            Some("s") => true,
            Some(_) => return Err(buttord_error(&BUTTORD_ERROR_INVALID_OPTION)),
            None => return Err(buttord_error(&BUTTORD_ERROR_INVALID_OPTION)),
        }
    } else {
        false
    };
    let wp = real_edges(BUILTIN_NAME, "Wp", wp)
        .await
        .map_err(|detail| buttord_error_with_detail(&BUTTORD_ERROR_INVALID_FREQUENCY, detail))?;
    let ws = real_edges(BUILTIN_NAME, "Ws", ws)
        .await
        .map_err(|detail| buttord_error_with_detail(&BUTTORD_ERROR_INVALID_FREQUENCY, detail))?;
    let rp = gather_value(&rp).await?;
    let rs = gather_value(&rs).await?;
    let rp = parse_positive_db(BUILTIN_NAME, "Rp", &rp)
        .map_err(|detail| buttord_error_with_detail(&BUTTORD_ERROR_INVALID_ATTENUATION, detail))?;
    let rs = parse_positive_db(BUILTIN_NAME, "Rs", &rs)
        .map_err(|detail| buttord_error_with_detail(&BUTTORD_ERROR_INVALID_ATTENUATION, detail))?;
    if rs <= rp {
        return Err(buttord_error(&BUTTORD_ERROR_INVALID_ATTENUATION));
    }

    let result = compute_buttord(&wp, &ws, rp, rs, analog)?;
    output_result(result)
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

fn is_complex_value(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
}

fn ensure_input_extensions(wp: &Value, ws: &Value, rp: &Value, rs: &Value) -> BuiltinResult<()> {
    for value in [wp, ws] {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &BUTTORD_INTEGER_FREQUENCY_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if is_logical_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &BUTTORD_LOGICAL_FREQUENCY_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if is_complex_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &BUTTORD_COMPLEX_FREQUENCY_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    for value in [rp, rs] {
        if is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &BUTTORD_INTEGER_ATTENUATION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if is_logical_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &BUTTORD_LOGICAL_ATTENUATION_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
    }
    if [wp, ws, rp, rs]
        .into_iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &BUTTORD_GPU_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

async fn gather_value(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => {
            let tensor = crate::builtins::common::gpu_helpers::gather_tensor_async(handle)
                .await
                .map_err(|flow| {
                    crate::builtins::common::map_control_flow_with_builtin(flow, BUILTIN_NAME)
                })?;
            Ok(crate::builtins::common::tensor::tensor_into_value(tensor))
        }
        other => Ok(other.clone()),
    }
}

fn output_result(result: ButtordResult) -> BuiltinResult<Value> {
    let order = Value::Num(result.order as f64);
    let wn = if result.wn.len() == 1 {
        Value::Num(result.wn[0])
    } else {
        real_vector_to_row_value(result.wn)
            .map_err(|err| buttord_error_with_detail(&BUTTORD_ERROR_INTERNAL, err.message()))?
    };
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![order]));
        }
        if out_count == 2 {
            return Ok(Value::OutputList(vec![order, wn]));
        }
        return Err(buttord_error(&BUTTORD_ERROR_TOO_MANY_OUTPUTS));
    }
    Ok(order)
}

fn compute_buttord(
    wp: &[f64],
    ws: &[f64],
    rp: f64,
    rs: f64,
    analog: bool,
) -> BuiltinResult<ButtordResult> {
    if wp.len() != ws.len() {
        return Err(buttord_error_with_detail(
            &BUTTORD_ERROR_INVALID_FREQUENCY,
            "Wp and Ws must have the same length",
        ));
    }
    let kind = classify_kind(wp, ws)
        .map_err(|detail| buttord_error_with_detail(&BUTTORD_ERROR_INVALID_FREQUENCY, detail))?;
    crate::builtins::math::signal::order_selection::validate_domain(wp, analog, "Wp")
        .map_err(|detail| buttord_error_with_detail(&BUTTORD_ERROR_INVALID_FREQUENCY, detail))?;
    crate::builtins::math::signal::order_selection::validate_domain(ws, analog, "Ws")
        .map_err(|detail| buttord_error_with_detail(&BUTTORD_ERROR_INVALID_FREQUENCY, detail))?;

    let wp_work = transform_edges(wp, analog);
    let ws_work = transform_edges(ws, analog);
    let epsilon = db_excess(rp)
        .map_err(|detail| buttord_error_with_detail(&BUTTORD_ERROR_INVALID_ATTENUATION, detail))?
        .sqrt();
    let log_ratio = db_excess_log_ratio(rp, rs)
        .map_err(|detail| buttord_error_with_detail(&BUTTORD_ERROR_INVALID_ATTENUATION, detail))?;
    let (nat, passband) =
        find_natural_frequency(kind, &wp_work, &ws_work, rp, rs, OrderFamily::Butterworth)
            .map_err(|detail| {
                buttord_error_with_detail(&BUTTORD_ERROR_INVALID_FREQUENCY, detail)
            })?;
    let order = checked_order(log_ratio / (2.0 * nat.ln()))
        .map_err(|detail| buttord_error_with_detail(&BUTTORD_ERROR_INVALID_FREQUENCY, detail))?;
    let natural = natural_frequency(kind, &passband, epsilon, order)?;
    let wn: Vec<f64> = if analog {
        natural
    } else {
        natural.into_iter().map(unwarp_frequency).collect()
    };
    validate_critical_frequencies(&wn, analog)
        .map_err(|detail| buttord_error_with_detail(&BUTTORD_ERROR_INVALID_FREQUENCY, detail))?;
    Ok(ButtordResult { order, wn })
}

fn natural_frequency(
    kind: FilterKind,
    passband: &[f64],
    epsilon: f64,
    order: usize,
) -> BuiltinResult<Vec<f64>> {
    let cutoff_scale = epsilon.powf(-1.0 / order as f64);
    if !cutoff_scale.is_finite() || cutoff_scale <= 0.0 {
        return Err(buttord_error_with_detail(
            &BUTTORD_ERROR_INVALID_FREQUENCY,
            "computed cutoff scale is not finite and positive",
        ));
    }
    match kind {
        FilterKind::Lowpass => Ok(vec![passband[0] * cutoff_scale]),
        FilterKind::Highpass => Ok(vec![passband[0] / cutoff_scale]),
        FilterKind::Bandpass => {
            let bandwidth = passband[1] - passband[0];
            let center_sq = passband[0] * passband[1];
            let disc = (cutoff_scale * bandwidth).powi(2) + 4.0 * center_sq;
            let root = disc.sqrt();
            Ok(vec![
                (-cutoff_scale * bandwidth + root) / 2.0,
                (cutoff_scale * bandwidth + root) / 2.0,
            ])
        }
        FilterKind::Bandstop => {
            let bandwidth = passband[1] - passband[0];
            let center_sq = passband[0] * passband[1];
            let disc = bandwidth * bandwidth + 4.0 * cutoff_scale * cutoff_scale * center_sq;
            let root = disc.sqrt();
            Ok(vec![
                (-bandwidth + root) / (2.0 * cutoff_scale),
                (bandwidth + root) / (2.0 * cutoff_scale),
            ])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{
        builtin_function_by_name, IntValue, IntegerStorage, NumericDType, Tensor,
    };

    fn call(
        wp: Value,
        ws: Value,
        rp: Value,
        rs: Value,
        rest: &[Value],
        outputs: Option<usize>,
    ) -> BuiltinResult<Value> {
        let _guard = outputs.map(|count| crate::output_count::push_output_count(Some(count)));
        block_on(evaluate(wp, ws, rp, rs, rest))
    }

    fn outputs(value: Value) -> (f64, Vec<f64>) {
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        let order = match &values[0] {
            Value::Num(value) => *value,
            other => panic!("expected scalar order, got {other:?}"),
        };
        let wn = match &values[1] {
            Value::Num(value) => vec![*value],
            Value::Tensor(tensor) => tensor.materialize_f64().clone(),
            other => panic!("expected Wn, got {other:?}"),
        };
        (order, wn)
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("typed integer tensor")
    }

    #[cfg(feature = "wgpu")]
    fn register_wgpu_provider_available() -> bool {
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_ok()
            && runmat_accelerate_api::provider().is_some()
    }

    #[test]
    fn descriptor_is_registered() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("buttord builtin");
        let descriptor = builtin.descriptor.expect("descriptor");
        assert!(descriptor
            .signatures
            .iter()
            .any(|sig| sig.label == "[n, Wn] = buttord(Wp, Ws, Rp, Rs)"));
        assert_eq!(builtin.extensions.len(), 6);
        assert_eq!(builtin.integer_capabilities.len(), 2);
        assert!(builtin
            .integer_capabilities
            .iter()
            .all(|capability| capability.output_class == BuiltinIntegerOutputClassRule::Double));
    }

    #[test]
    fn digital_lowpass_order_and_cutoff() {
        let out = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 12.0);
        assert_eq!(wn.len(), 1);
        assert!((wn[0] - 0.2108).abs() < 5e-4);
    }

    #[test]
    fn digital_highpass_order_and_cutoff() {
        let out = call(
            Value::Num(0.5),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 8.0);
        assert_eq!(wn.len(), 1);
        assert!((wn[0] - 0.4732).abs() < 5e-4);
    }

    #[test]
    fn analog_lowpass_keeps_radian_cutoff() {
        let out = call(
            Value::Num(10.0),
            Value::Num(20.0),
            Value::Num(1.0),
            Value::Num(40.0),
            &[Value::from("s")],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 8.0);
        assert_eq!(wn.len(), 1);
        assert!(wn[0] > 10.0);
        assert!(wn[0] < 12.0);
    }

    #[test]
    fn analog_lowpass_reads_typed_integer_design_args_exactly() {
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let out = call(
            Value::Tensor(integer_tensor(IntegerStorage::I16(vec![10]), vec![1, 1])),
            Value::Tensor(integer_tensor(IntegerStorage::I16(vec![20]), vec![1, 1])),
            Value::Tensor(integer_tensor(IntegerStorage::U16(vec![1]), vec![1, 1])),
            Value::Tensor(integer_tensor(IntegerStorage::U16(vec![40]), vec![1, 1])),
            &[Value::from("s")],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert_eq!(n, 8.0);
        assert_eq!(wn.len(), 1);
        assert!(wn[0] > 10.0);
        assert!(wn[0] < 12.0);
    }

    #[test]
    fn buttord_accepts_all_integer_frequency_and_attenuation_classes_only_in_runmat_mode() {
        let integer_values = || {
            [
                Value::Int(IntValue::I8(1)),
                Value::Int(IntValue::I16(1)),
                Value::Int(IntValue::I32(1)),
                Value::Int(IntValue::I64(1)),
                Value::Int(IntValue::U8(1)),
                Value::Int(IntValue::U16(1)),
                Value::Int(IntValue::U32(1)),
                Value::Int(IntValue::U64(1)),
            ]
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = call(
                Value::Int(IntValue::U64(1)),
                Value::Num(2.0),
                Value::Num(1.0),
                Value::Num(40.0),
                &[Value::from("s")],
                Some(2),
            )
            .expect_err("integer frequency must be gated");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:ButtordIntegerFrequencyExtension")
            );
            let err = call(
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Int(IntValue::U64(1)),
                Value::Num(40.0),
                &[Value::from("s")],
                Some(2),
            )
            .expect_err("integer attenuation must be gated");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:ButtordIntegerAttenuationExtension")
            );
        }

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for wp in integer_values() {
            let out = call(
                wp,
                Value::Num(2.0),
                Value::Num(1.0),
                Value::Num(40.0),
                &[Value::from("s")],
                Some(2),
            )
            .expect("integer frequency");
            let Value::OutputList(values) = out else {
                panic!("expected output list");
            };
            assert!(values.iter().all(|value| matches!(value, Value::Num(_))));
        }
        for rp in integer_values() {
            let out = call(
                Value::Num(1.0),
                Value::Num(2.0),
                rp,
                Value::Num(40.0),
                &[Value::from("s")],
                Some(2),
            )
            .expect("integer attenuation");
            let Value::OutputList(values) = out else {
                panic!("expected output list");
            };
            assert!(values.iter().all(|value| matches!(value, Value::Num(_))));
        }
    }

    #[test]
    fn buttord_logical_and_complex_extensions_are_independently_gated() {
        let cases = [
            (
                Value::Bool(true),
                Value::Num(2.0),
                Value::Num(1.0),
                Value::Num(40.0),
                "RunMat:compatibility:ButtordLogicalFrequencyExtension",
            ),
            (
                Value::Complex(1.0, 0.0),
                Value::Num(2.0),
                Value::Num(1.0),
                Value::Num(40.0),
                "RunMat:compatibility:ButtordComplexFrequencyExtension",
            ),
            (
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Bool(true),
                Value::Num(40.0),
                "RunMat:compatibility:ButtordLogicalAttenuationExtension",
            ),
        ];
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (wp, ws, rp, rs, identifier) in cases {
            let err = call(wp, ws, rp, rs, &[Value::from("s")], Some(2))
                .expect_err("extension must be gated");
            assert_eq!(err.identifier(), Some(identifier));
        }
    }

    #[test]
    fn buttord_documented_single_inputs_return_double_outputs_in_compatibility_mode() {
        let scalar = |value| {
            Value::Tensor(Tensor::from_f32(vec![value], vec![1, 1]).expect("single scalar"))
        };
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let Value::OutputList(values) = call(
            scalar(1.0),
            scalar(2.0),
            scalar(1.0),
            scalar(40.0),
            &[Value::from("s")],
            Some(2),
        )
        .expect("documented single form") else {
            panic!("expected output list");
        };
        assert!(values.iter().all(|value| {
            matches!(value, Value::Num(_))
                || matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64)
        }));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn buttord_gpu_inputs_are_gated_before_gather_and_runmat_fallback_is_host_double() {
        let _lock = crate::builtins::common::test_support::accel_test_lock();
        if !register_wgpu_provider_available() {
            return;
        }
        let provider = runmat_accelerate_api::provider().expect("provider");
        let upload = |value| {
            let tensor = Tensor::new(vec![value], vec![1, 1]).expect("scalar");
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor).expect("upload")
        };
        let wp = upload(1.0);
        let ws = upload(2.0);
        let rp = upload(1.0);
        let rs = upload(40.0);
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = call(
                Value::GpuTensor(wp.clone()),
                Value::Num(2.0),
                Value::Num(1.0),
                Value::Num(40.0),
                &[Value::from("s")],
                Some(2),
            )
            .expect_err("gpu input must be gated");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:ButtordGpuInputExtension")
            );
        }
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let Value::OutputList(values) = call(
            Value::GpuTensor(wp),
            Value::GpuTensor(ws),
            Value::GpuTensor(rp),
            Value::GpuTensor(rs),
            &[Value::from("s")],
            Some(2),
        )
        .expect("host fallback") else {
            panic!("expected output list");
        };
        assert!(values.iter().all(|value| {
            matches!(value, Value::Num(_))
                || matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64)
        }));
    }

    #[test]
    fn bandpass_returns_two_cutoff_edges() {
        let wp = Tensor::new(vec![0.3, 0.5], vec![1, 2]).unwrap();
        let ws = Tensor::new(vec![0.2, 0.6], vec![1, 2]).unwrap();
        let out = call(
            Value::Tensor(wp),
            Value::Tensor(ws),
            Value::Num(1.0),
            Value::Num(30.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert!(n >= 1.0);
        assert_eq!(wn.len(), 2);
        assert!(wn[0] < 0.3);
        assert!(wn[1] > 0.5);
        assert!(wn[0] < wn[1]);
    }

    #[test]
    fn bandstop_uses_passband_edges_for_transform_basis() {
        let wp = Tensor::new(vec![0.2, 0.7], vec![1, 2]).unwrap();
        let ws = Tensor::new(vec![0.35, 0.5], vec![1, 2]).unwrap();
        let out = call(
            Value::Tensor(wp),
            Value::Tensor(ws),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(2),
        )
        .unwrap();
        let (n, wn) = outputs(out);
        assert!(n >= 1.0);
        assert_eq!(wn.len(), 2);
        assert!(wn[0] < 0.35);
        assert!(wn[1] > 0.5);
    }

    #[test]
    fn rejects_invalid_specs() {
        let err = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(1.0),
            Value::Num(40.0),
            &[],
            Some(3),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:buttord:TooManyOutputs"));

        let err = call(
            Value::Num(0.2),
            Value::Num(0.3),
            Value::Num(40.0),
            Value::Num(1.0),
            &[],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:buttord:InvalidAttenuation"));

        let err = call(
            Value::Num(1.2),
            Value::Num(1.4),
            Value::Num(1.0),
            Value::Num(20.0),
            &[],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:buttord:InvalidFrequency"));

        let err = call(
            Value::Num(0.2),
            Value::Num(0.2000001),
            Value::Num(1.0),
            Value::Num(1.0e6),
            &[],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:buttord:InvalidFrequency"));
    }
}
