use crate::*;
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pat",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host wildcard pattern.",
}];
const N: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exact nonnegative character count.",
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
        label: "pat = wildcardPattern",
        inputs: &[],
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "pat = wildcardPattern(N)",
        inputs: &N,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "pat = wildcardPattern(minCharacters, maxCharacters)",
        inputs: &RANGE,
        outputs: &OUTPUTS,
    },
];
pub const WILDCARD_PATTERN_ERROR_ARGUMENT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WILDCARD_PATTERN.ARGUMENT_COUNT",
    identifier: Some("RunMat:wildcardPattern:ArgumentCount"),
    when: "More than two count arguments are supplied.",
    message: "wildcardPattern: expected zero, one, or two count arguments",
};
pub const WILDCARD_PATTERN_ERROR_INVALID_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: "RM.WILDCARD_PATTERN.INVALID_COUNT", identifier: Some("RunMat:wildcardPattern:InvalidCount"), when: "A count is not a supported nonnegative integer-valued scalar, or an explicit GPU count is supplied.", message: "wildcardPattern: invalid character count" };
pub const WILDCARD_PATTERN_ERROR_INVALID_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WILDCARD_PATTERN.INVALID_RANGE",
    identifier: Some("RunMat:wildcardPattern:InvalidRange"),
    when: "The minimum exceeds the maximum.",
    message: "wildcardPattern: minCharacters must not exceed maxCharacters",
};
pub const WILDCARD_PATTERN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.WILDCARD_PATTERN.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:wildcardPattern:TooManyOutputs"),
        when: "More than one output is requested.",
        message: "wildcardPattern: too many output arguments",
    };
pub const WILDCARD_PATTERN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        WILDCARD_PATTERN_ERROR_ARGUMENT_COUNT,
        WILDCARD_PATTERN_ERROR_INVALID_COUNT,
        WILDCARD_PATTERN_ERROR_INVALID_RANGE,
        WILDCARD_PATTERN_ERROR_TOO_MANY_OUTPUTS,
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
pub const WILDCARD_PATTERN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "pat = wildcardPattern(integer_N)", inputs: &N_INTEGER, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Exact count controls the host pattern. Automatic resident controls may gather; explicit gpuArray controls reject." },
    BuiltinIntegerCapabilityDescriptor { form: "pat = wildcardPattern(integer_minCharacters, integer_maxCharacters)", inputs: &RANGE_INTEGER, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Inclusive bounds are exact and ordered; wildcard matching is lazy toward the minimum." },
];
