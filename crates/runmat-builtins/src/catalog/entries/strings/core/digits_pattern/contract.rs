use crate::*;
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pat",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Unicode digit pattern.",
}];
const N: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exact nonnegative digit count.",
}];
const RANGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "minCharacters",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Inclusive nonnegative minimum.",
    },
    BuiltinParamDescriptor {
        name: "maxCharacters",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Inclusive nonnegative maximum, or positive infinity.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "pat = digitsPattern",
        inputs: &[],
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "pat = digitsPattern(N)",
        inputs: &N,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "pat = digitsPattern(minCharacters, maxCharacters)",
        inputs: &RANGE,
        outputs: &OUTPUTS,
    },
];
pub const DIGITS_PATTERN_ERROR_ARGUMENT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIGITS_PATTERN.ARGUMENT_COUNT",
    identifier: Some("RunMat:digitsPattern:ArgumentCount"),
    when: "More than two count arguments are supplied.",
    message: "digitsPattern: expected zero, one, or two input arguments",
};
pub const DIGITS_PATTERN_ERROR_INVALID_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: "RM.DIGITS_PATTERN.INVALID_COUNT", identifier: Some("RunMat:digitsPattern:InvalidCount"), when: "A count is not a host nonnegative numeric integer scalar or positive infinity in the maximum position.", message: "digitsPattern: invalid character count" };
pub const DIGITS_PATTERN_ERROR_INVALID_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIGITS_PATTERN.INVALID_RANGE",
    identifier: Some("RunMat:digitsPattern:InvalidRange"),
    when: "The minimum exceeds the maximum.",
    message: "digitsPattern: minCharacters must not exceed maxCharacters",
};
pub const DIGITS_PATTERN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIGITS_PATTERN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:digitsPattern:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "digitsPattern: too many output arguments",
};
pub const DIGITS_PATTERN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        DIGITS_PATTERN_ERROR_ARGUMENT_COUNT,
        DIGITS_PATTERN_ERROR_INVALID_COUNT,
        DIGITS_PATTERN_ERROR_INVALID_RANGE,
        DIGITS_PATTERN_ERROR_TOO_MANY_OUTPUTS,
    ],
};
const N_INTEGER: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "N",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes:
        "Every native integer class and integer-valued single/double scalar is accepted exactly.",
}];
const RANGE_INTEGER: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "minCharacters", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Inclusive minimum is decoded exactly." },
    BuiltinIntegerInputCapability { name: "maxCharacters", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Inclusive maximum is decoded exactly; positive host floating infinity separately means unbounded." },
];
pub const DIGITS_PATTERN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "pat = digitsPattern(integer_N)", inputs: &N_INTEGER, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Count controls only the host pattern repetition bound; resident controls reject before provider access." },
    BuiltinIntegerCapabilityDescriptor { form: "pat = digitsPattern(integer_minCharacters, integer_maxCharacters)", inputs: &RANGE_INTEGER, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "The inclusive bounds are exact and must be ordered; the resulting Unicode digit pattern is greedy." },
];
