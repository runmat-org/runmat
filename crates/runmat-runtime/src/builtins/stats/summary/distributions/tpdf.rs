use super::*;
use super::*;
normal_descriptor!("tpdf", T_PDF_SIGNATURES);

const INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tpdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tpdf accepts typed-integer evaluation points",
    error_identifier: Some("RunMat:compatibility:TpdfIntegerXExtension"),
};
const INTEGER_NU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tpdf-integer-degrees-of-freedom",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tpdf accepts typed-integer degrees of freedom",
    error_identifier: Some("RunMat:compatibility:TpdfIntegerDegreesOfFreedomExtension"),
};
const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tpdf-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tpdf accepts logical numeric input",
    error_identifier: Some("RunMat:compatibility:TpdfLogicalInputExtension"),
};
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    INTEGER_X_EXTENSION,
    INTEGER_NU_EXTENSION,
    LOGICAL_INPUT_EXTENSION,
];

const INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public tpdf evaluation points are single or double; native integers are independently gated and must cross the binary64 PDF boundary exactly.",
    }];
const INTEGER_NU_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nu",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public tpdf degrees of freedom are single or double; native integers are independently gated and must cross the binary64 PDF boundary exactly.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "y = tpdf(integer_x, nu)", inputs: &INTEGER_X_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "Admitted integer x values are checked before binary64 evaluation. Documented single input selects single output, and resident fallback restores through the exact owning provider." },
    BuiltinIntegerCapabilityDescriptor { form: "y = tpdf(x, integer_nu)", inputs: &INTEGER_NU_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "Admitted integer nu values are checked before binary64 evaluation. Documented single input selects single output, and resident fallback restores through the exact owning provider." },
];

#[runtime_builtin(
    name = "tpdf",
    category = "stats/summary",
    summary = "Evaluate the Student's t probability density function.",
    keywords = "tpdf,student t,pdf,statistics,distribution",
    type_resolver(super::normal_type),
    descriptor(self::DESCRIPTOR),
    extensions(self::EXTENSIONS),
    integer_capabilities(self::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::distributions::tpdf"
)]
pub(crate) async fn tpdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() != 1 {
        return Err(super::normal_error("tpdf", "tpdf: expected x and nu"));
    }
    let nu_value = rest.into_iter().next().expect("one nu argument");
    ensure_extensions(&value, &nu_value)?;
    let output = NormalOutputPlan::inspect("tpdf", &value, std::slice::from_ref(&nu_value))?;
    let x = super::normal_value_to_tensor("tpdf", "x", value).await?;
    let nu = super::normal_value_to_tensor("tpdf", "nu", nu_value).await?;
    let (x, nu, shape) = super::broadcast_pair("tpdf", &x, &nu)?;
    let data = x
        .iter()
        .zip(nu.iter())
        .map(|(x, nu)| distribution_math::student_t_pdf(*x, *nu))
        .collect();
    output.finish("tpdf", shape, data)
}

fn ensure_extensions(x: &Value, nu: &Value) -> BuiltinResult<()> {
    if super::is_typed_integer_value(x) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_X_EXTENSION, "tpdf")?;
    }
    if super::is_typed_integer_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_NU_EXTENSION, "tpdf")?;
    }
    if super::is_logical_value(x) || super::is_logical_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_INPUT_EXTENSION, "tpdf")?;
    }
    Ok(())
}
