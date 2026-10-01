use super::*;
use super::*;
normal_descriptor!("tinv", T_INV_SIGNATURES);

const INTEGER_P_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tinv-integer-probability",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tinv with typed-integer probabilities is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TinvIntegerProbabilityExtension"),
};
const INTEGER_NU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tinv-integer-degrees-of-freedom",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tinv with typed-integer degrees of freedom is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TinvIntegerDegreesOfFreedomExtension"),
};
const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tinv-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tinv with logical numeric input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TinvLogicalInputExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    INTEGER_P_EXTENSION,
    INTEGER_NU_EXTENSION,
    LOGICAL_INPUT_EXTENSION,
];

const INTEGER_P_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public tinv probabilities are single or double. Typed-integer probabilities are independently gated and must cross the binary64 inverse-CDF boundary exactly.",
    }];
const INTEGER_NU_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nu",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public tinv degrees of freedom are single or double. Typed-integer values are independently gated and must cross the binary64 inverse-CDF boundary exactly.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "x = tinv(integer_p, nu)",
        inputs: &INTEGER_P_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "RunMat-only integer p is checked before binary64 evaluation. Documented single input selects single output, and resident fallback restores through the exact owning provider when possible.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "x = tinv(p, integer_nu)",
        inputs: &INTEGER_NU_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "RunMat-only integer nu is checked before binary64 evaluation. Documented single input selects single output, and resident fallback restores through the exact owning provider when possible.",
    },
];

#[runtime_builtin(
    name = "tinv",
    category = "stats/summary",
    summary = "Evaluate the inverse Student's t cumulative distribution function.",
    keywords = "tinv,student t,inverse,cdf,statistics,distribution",
    type_resolver(super::normal_type),
    descriptor(self::DESCRIPTOR),
    extensions(self::EXTENSIONS),
    integer_capabilities(self::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::distributions::tinv"
)]
pub(crate) async fn tinv_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() != 1 {
        return Err(super::normal_error("tinv", "tinv: expected p and nu"));
    }
    let nu_value = rest.into_iter().next().expect("one nu argument");
    ensure_extensions(&value, &nu_value)?;
    let output = NormalOutputPlan::inspect("tinv", &value, std::slice::from_ref(&nu_value))?;
    let p = super::normal_value_to_tensor("tinv", "p", value).await?;
    let nu = super::normal_value_to_tensor("tinv", "nu", nu_value).await?;
    let (p, nu, shape) = super::broadcast_pair("tinv", &p, &nu)?;
    let data = p
        .iter()
        .zip(nu.iter())
        .map(|(p, nu)| distribution_math::student_t_inv(*p, *nu))
        .collect();
    output.finish("tinv", shape, data)
}

fn ensure_extensions(p: &Value, nu: &Value) -> BuiltinResult<()> {
    if super::is_typed_integer_value(p) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_P_EXTENSION, "tinv")?;
    }
    if super::is_typed_integer_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_NU_EXTENSION, "tinv")?;
    }
    if super::is_logical_value(p) || super::is_logical_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_INPUT_EXTENSION, "tinv")?;
    }
    Ok(())
}
