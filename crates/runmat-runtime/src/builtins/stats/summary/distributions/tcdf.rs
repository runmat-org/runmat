use super::*;
use super::*;
normal_descriptor!("tcdf", T_CDF_SIGNATURES);

const INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcdf with typed-integer evaluation points is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcdfIntegerXExtension"),
};
const INTEGER_NU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcdf-integer-degrees-of-freedom",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcdf with typed-integer degrees of freedom is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcdfIntegerDegreesOfFreedomExtension"),
};
const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcdf-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcdf with logical numeric input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcdfLogicalInputExtension"),
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
        notes: "Public tcdf inputs are single or double. Typed-integer evaluation points are independently gated and must cross the binary64 CDF boundary exactly.",
    }];
const INTEGER_NU_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nu",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Public tcdf degrees of freedom are single or double. Typed-integer values are independently gated and must cross the binary64 CDF boundary exactly.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "p = tcdf(integer_x, nu [, \"upper\"])",
        inputs: &INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "RunMat-only integer x is checked before binary64 evaluation. Documented single input selects single output, and resident fallback restores through the exact owning provider when possible.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "p = tcdf(x, integer_nu [, \"upper\"])",
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
    name = "tcdf",
    category = "stats/summary",
    summary = "Evaluate the Student's t cumulative distribution function.",
    keywords = "tcdf,student t,cdf,upper tail,statistics,distribution",
    type_resolver(super::normal_type),
    descriptor(self::DESCRIPTOR),
    extensions(self::EXTENSIONS),
    integer_capabilities(self::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::distributions::tcdf"
)]
pub(crate) async fn tcdf_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = parse_args(value, rest).await?;
    let data = args
        .x
        .iter()
        .zip(args.nu.iter())
        .map(|(x, nu)| {
            if args.upper {
                distribution_math::student_t_cdf_upper(*x, *nu)
            } else {
                distribution_math::student_t_cdf(*x, *nu)
            }
        })
        .collect();
    args.output.finish("tcdf", args.shape, data)
}

struct Args {
    x: Vec<f64>,
    nu: Vec<f64>,
    shape: Vec<usize>,
    upper: bool,
    output: NormalOutputPlan,
}

async fn parse_args(value: Value, mut rest: Vec<Value>) -> BuiltinResult<Args> {
    let upper = rest
        .last()
        .and_then(crate::builtins::common::random_args::keyword_of)
        .is_some_and(|keyword| keyword.eq_ignore_ascii_case("upper"));
    if upper {
        rest.pop();
    }
    if rest.len() != 1 {
        return Err(super::normal_error(
            "tcdf",
            "tcdf: expected x and nu, optionally followed by 'upper'",
        ));
    }
    let nu_value = rest.pop().expect("one nu argument");
    ensure_extensions(&value, &nu_value)?;
    let output = NormalOutputPlan::inspect("tcdf", &value, std::slice::from_ref(&nu_value))?;
    let x = super::normal_value_to_tensor("tcdf", "x", value).await?;
    let nu = super::normal_value_to_tensor("tcdf", "nu", nu_value).await?;
    let (x, nu, shape) = super::broadcast_pair("tcdf", &x, &nu)?;
    Ok(Args {
        x,
        nu,
        shape,
        upper,
        output,
    })
}

fn ensure_extensions(x: &Value, nu: &Value) -> BuiltinResult<()> {
    if super::is_typed_integer_value(x) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_X_EXTENSION, "tcdf")?;
    }
    if super::is_typed_integer_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_NU_EXTENSION, "tcdf")?;
    }
    if super::is_logical_value(x) || super::is_logical_value(nu) {
        crate::compatibility::ensure_builtin_extension_enabled(&LOGICAL_INPUT_EXTENSION, "tcdf")?;
    }
    Ok(())
}
