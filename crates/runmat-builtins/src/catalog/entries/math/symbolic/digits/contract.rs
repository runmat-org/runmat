use crate::*;

pub const DIGITS_DEFAULT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "digits-default-reset",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "digits('default') is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DigitsDefaultResetExtension"),
};

pub const DIGITS_NUMERIC_TEXT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "digits-numeric-text",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "digits with a numeric text scalar is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DigitsNumericTextExtension"),
};

pub const DIGITS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [DIGITS_DEFAULT_EXTENSION, DIGITS_NUMERIC_TEXT_EXTENSION];

pub const DIGITS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "d",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The positive integer precision control is read exactly from every host integer class; host double is rounded to the nearest integer before enforcing the documented 2..=100000000 range.",
    }];

pub const DIGITS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "old = digits(integer_d)",
        inputs: &DIGITS_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "d changes host symbolic precision state only; the returned previous precision is always a double scalar, and invalid controls leave state unchanged.",
    }];

pub const DIGITS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current or previous variable-precision digit count.",
}];

pub const DIGITS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Host real single/double or integer scalar, rounded to the nearest integer in 2..=100000000; numeric scalar text or 'default' reset require their named RunMat extensions. Logical, resident and array controls reject.",
}];

pub const DIGITS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "d = digits",
        inputs: &[],
        outputs: &DIGITS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "old = digits(d)",
        inputs: &DIGITS_INPUTS,
        outputs: &DIGITS_OUTPUT,
    },
];

pub const DIGITS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.DIGITS.ARG_COUNT",
        identifier: Some("RunMat:digits:ArgCount"),
        when: "More than one input argument is supplied.",
        message: "digits: too many input arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.DIGITS.INVALID_DIGITS",
        identifier: Some("RunMat:digits:InvalidDigits"),
        when: "The precision control is not a supported host scalar, is nonfinite, or its nearest-integer value is outside 2..=100000000.",
        message: "digits: expected a positive integer digit count",
    },
    DIGITS_ERROR_TOO_MANY_OUTPUTS,
];

pub const DIGITS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DIGITS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DIGITS_ERRORS,
};

pub const DIGITS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIGITS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:digits:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "digits: too many output arguments",
};
