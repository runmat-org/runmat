use crate::*;

pub const VPA_INTEGER_VALUE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every built-in integer class is documented input. RunMat's current scalar symbolic surface lifts the authoritative integer directly into an exact BigInt-backed rational before decimal rendering.",
    }];

pub const VPA_INTEGER_DIGITS_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "d",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The positive significant-digit count is decoded exactly from every integer class; floating controls use the documented nearest-integer rounding before range validation.",
    }];

pub const VPA_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "S = vpa(integer_y)",
        inputs: &VPA_INTEGER_VALUE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The supported scalar result is symbolic; no integer payload crosses binary floating point, including int64 and uint64 extrema.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "S = vpa(y, integer_d)",
        inputs: &VPA_INTEGER_DIGITS_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "d selects host arbitrary-precision evaluation state and is bounded to the supported precision range before allocation.",
    },
];

pub const VPA_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar symbolic variable-precision value or expression.",
}];

pub const VPA_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Host symbolic scalar, real numeric scalar from any class, scalar logical, or scalar text containing a finite number or exact integer rational; no symbolic arrays or complex input.",
    },
    BuiltinParamDescriptor {
        name: "d",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("digits"),
        description: "Host real numeric scalar precision rounded to the nearest integer in 1..=100000000. A scalar logical true selects one digit; text, logical arrays and resident controls reject.",
    },
];

pub const VPA_INPUTS_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host symbolic scalar, real numeric scalar from any class, scalar logical, or finite numeric/exact-rational scalar text; symbolic arrays and complex input reject.",
}];

pub const VPA_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "S = vpa(x)",
        inputs: &VPA_INPUTS_VALUE,
        outputs: &VPA_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = vpa(x, d)",
        inputs: &VPA_INPUTS,
        outputs: &VPA_OUTPUT,
    },
];

pub const VPA_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.VPA.ARG_COUNT",
        identifier: Some("RunMat:vpa:ArgCount"),
        when: "More than one optional precision argument is supplied.",
        message: "vpa: too many input arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.VPA.INVALID_INPUT",
        identifier: Some("RunMat:vpa:InvalidInput"),
        when: "Input cannot be represented as a scalar symbolic expression.",
        message: "vpa: expected a scalar symbolic, numeric, logical, or numeric text input",
    },
    BuiltinErrorDescriptor {
        code: "RM.VPA.INVALID_DIGITS",
        identifier: Some("RunMat:vpa:InvalidDigits"),
        when: "Precision is not a supported real host scalar, is nonfinite, or rounds outside 1..=100000000; text and logical-array controls are not supported.",
        message: "vpa: expected a positive integer digit count",
    },
    VPA_ERROR_TOO_MANY_OUTPUTS,
];

pub const VPA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &VPA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VPA_ERRORS,
};

pub const VPA_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VPA.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:vpa:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "vpa: too many output arguments",
};
