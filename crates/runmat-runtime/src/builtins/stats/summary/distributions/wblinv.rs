use super::*;
use super::*;
normal_descriptor!("wblinv", WBLINV_SIGNATURES);

const INTEGER_P_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblinv-integer-probability",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblinv with typed-integer probabilities is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblinvIntegerProbabilityExtension"),
};
const INTEGER_A_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblinv-integer-scale",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblinv with typed-integer scale parameters is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblinvIntegerScaleExtension"),
};
const INTEGER_B_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblinv-integer-shape",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblinv with typed-integer shape parameters is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblinvIntegerShapeExtension"),
};
const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblinv-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblinv with logical numeric input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblinvLogicalInputExtension"),
};
const EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblinv-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblinv with explicit gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblinvExplicitGpuInputExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    INTEGER_P_EXTENSION,
    INTEGER_A_EXTENSION,
    INTEGER_B_EXTENSION,
    LOGICAL_INPUT_EXTENSION,
    EXPLICIT_GPU_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "p",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented probability datatype is single or double; typed integers are independently gated and must be exactly representable at the binary64 distribution boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "a",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented scale datatype is single or double; typed integers are independently gated and checked before conversion.",
    },
    BuiltinIntegerInputCapability {
        name: "b",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented shape datatype is single or double; typed integers are independently gated and checked before conversion.",
    },
];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "x = wblinv(integer_p, integer_a, integer_b)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each typed role is gated independently before provider access. Admitted values cross one checked binary64 Weibull boundary; single dominates output only when another documented input is single, and automatic fallback restores through a physically compatible owner.",
    }];

fn ensure_extensions(value: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if super::is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_P_EXTENSION, "wblinv")?;
    }
    if rest.first().is_some_and(super::is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_A_EXTENSION, "wblinv")?;
    }
    if rest.get(1).is_some_and(super::is_typed_integer_value) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_B_EXTENSION, "wblinv")?;
    }
    if std::iter::once(value)
        .chain(rest.iter())
        .any(super::is_logical_value)
    {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_INPUT_EXTENSION, "wblinv")?;
    }
    if std::iter::once(value).chain(rest.iter()).any(|value| {
        matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
    }) {
        crate::compatibility::ensure_builtin_extension_enabled(&EXPLICIT_GPU_EXTENSION, "wblinv")?;
    }
    Ok(())
}

#[runtime_builtin(
    name = "wblinv",
    category = "stats/summary",
    summary = "Evaluate the inverse Weibull cumulative distribution function.",
    keywords = "wblinv,weibull,inverse,cdf,statistics,distribution",
    type_resolver(super::normal_type),
    descriptor(self::DESCRIPTOR),
    extensions(self::EXTENSIONS),
    integer_capabilities(self::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::distributions::wblinv"
)]
pub(crate) async fn wblinv_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_extensions(&value, &rest)?;
    let output = super::NormalOutputPlan::inspect("wblinv", &value, &rest)?;
    let (a, b) = match rest.as_slice() {
        [] => (super::scalar_tensor(1.0), super::scalar_tensor(1.0)),
        [a, b] => (
            super::normal_value_to_tensor("wblinv", "a", a.clone()).await?,
            super::normal_value_to_tensor("wblinv", "b", b.clone()).await?,
        ),
        _ => {
            return Err(super::normal_error(
                "wblinv",
                "wblinv: expected one or three numeric arguments",
            ))
        }
    };
    let p = super::normal_value_to_tensor("wblinv", "p", value).await?;
    let (mut values, shape) = super::broadcast_tensors("wblinv", &[&p, &a, &b])?;
    let p = values.remove(0);
    let a = values.remove(0);
    let b = values.remove(0);
    let data = p
        .iter()
        .zip(a.iter())
        .zip(b.iter())
        .map(|((p, a), b)| super::wblinv_scalar(*p, *a, *b))
        .collect();
    output.finish("wblinv", shape, data)
}
