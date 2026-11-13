use crate::*;

const FUN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Integrand callback.",
};
const LOWER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "xmin",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Lower integration bound.",
};
const UPPER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "xmax",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper integration bound.",
};
const CORE: [BuiltinParamDescriptor; 3] = [FUN, LOWER, UPPER];
const STRUCT: [BuiltinParamDescriptor; 4] = [
    FUN,
    LOWER,
    UPPER,
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Scalar options struct.",
    },
];
const PAIRS: [BuiltinParamDescriptor; 5] = [
    FUN,
    LOWER,
    UPPER,
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Option name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::PropertyValue,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value and further name/value pairs.",
    },
];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real or complex double scalar estimate, or a double array with the callback's shape when ArrayValued is true.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "q = integral(fun, xmin, xmax)",
        inputs: &CORE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "q = integral(fun, xmin, xmax, options)",
        inputs: &STRUCT,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "q = integral(fun, xmin, xmax, name, value, ...)",
        inputs: &PAIRS,
        outputs: &OUTPUT,
    },
];

pub const INTEGRAL_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTEGRAL.INVALID_ARGUMENT",
    identifier: Some("RunMat:integral:InvalidArgument"),
    when: "Option grammar/name-value parsing is invalid.",
    message: "integral: invalid argument",
};
pub const INTEGRAL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTEGRAL.INVALID_INPUT",
    identifier: Some("RunMat:integral:InvalidInput"),
    when: "Bounds/integrand/adaptive solver semantics are invalid.",
    message: "integral: invalid input",
};
pub const INTEGRAL_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTEGRAL.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:integral:TooManyOutputs"),
    when: "The caller requests more than the one supported numerical output.",
    message: "integral: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    INTEGRAL_ERROR_INVALID_ARGUMENT,
    INTEGRAL_ERROR_INVALID_INPUT,
    INTEGRAL_ERROR_TOO_MANY_OUTPUTS,
];
pub const INTEGRAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const INTEGRAL_INTEGER_BOUND_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "integral-integer-bound",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "integral with typed-integer bounds is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IntegralIntegerBoundExtension"),
    };
pub const INTEGRAL_LOGICAL_BOUND_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "integral-logical-bound",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "integral with logical bounds is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IntegralLogicalBoundExtension"),
    };
pub const INTEGRAL_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "integral-integer-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "integral with typed-integer numeric options is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IntegralIntegerOptionExtension"),
    };
pub const INTEGRAL_LOGICAL_NUMERIC_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "integral-logical-numeric-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "integral with logical tolerance or evaluation-count options is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:IntegralLogicalNumericOptionExtension"),
    };
pub const INTEGRAL_RENAMED_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "integral-renamed-tolerance-option",
        mode: BuiltinExtensionMode::AvailableSince(MatlabRelease {
            year: 2026,
            half: MatlabReleaseHalf::B,
        }),
        description: "integral renamed tolerance options require a newer compatibility release or RunMat extension mode",
        error_identifier: Some("RunMat:compatibility:IntegralRenamedToleranceExtension"),
    };
pub const INTEGRAL_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    INTEGRAL_INTEGER_BOUND_EXTENSION,
    INTEGRAL_LOGICAL_BOUND_EXTENSION,
    INTEGRAL_INTEGER_OPTION_EXTENSION,
    INTEGRAL_LOGICAL_NUMERIC_OPTION_EXTENSION,
    INTEGRAL_RENAMED_TOLERANCE_EXTENSION,
];

const INTEGER_BOUND_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "xmin or xmax",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed-integer bounds must be exactly representable in binary64.",
}];
const INTEGER_OPTION_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "AbsTol, RelTol, MaxFunEvals, or MaxIntervalCount value",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Floating tolerances cross a checked binary64 boundary; count controls are exact structural integers.",
}];
pub const INTEGRAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "q = integral(fun, integer_xmin, integer_xmax, ...)",
        inputs: &INTEGER_BOUND_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "RunMat-only integer bounds are classified before resident access and converted after exactness validation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "q = integral(fun, xmin, xmax, integer_option)",
        inputs: &INTEGER_OPTION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only typed-integer controls are gated independently from bounds and retain exact count parsing.",
    },
];
