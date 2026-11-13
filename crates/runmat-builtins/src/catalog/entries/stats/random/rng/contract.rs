//! Typed declarations conserved from the Runtime owner.
use crate::*;

pub const RNG_WIDE_SEED_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rng-wide-seed",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rng seeds at or above 2^32 are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RngWideSeedExtension"),
};

pub const RNG_TYPED_STATE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rng-typed-state-fields",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "independently supplied native-integer Seed fields or legacy LCG State fields are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RngTypedStateFieldsExtension"),
};

pub const RNG_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [RNG_WIDE_SEED_EXTENSION, RNG_TYPED_STATE_EXTENSION];

pub const RNG_INTEGER_SEED_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "seed",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented seed is a nonnegative integer below 2^32; native integer scalars are read exactly before the range check.",
    }];

pub const RNG_WIDE_INTEGER_SEED_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "seed",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode extends the seed domain through u64; native typed values remain exact even above flintmax.",
    }];

pub const RNG_INTEGER_STATE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "State field",
        classes: &[crate::BuiltinIntegerClass::Uint32],
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "A structure returned by RunMat rng carries a 625-by-1 uint32 MT19937 state; its final word is RunMat's next-word index, not a claim of MATLAB byte-wire parity.",
    }];

pub const RNG_EXTENDED_STATE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "independently typed Seed or legacy LCG State field",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Independently supplied native-integer Seed fields and legacy LCG State fields require RunMat compatibility mode.",
    }];

pub const RNG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "s = rng(integer_seed[, generator]) where seed < 2^32",
        inputs: &RNG_INTEGER_SEED_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact seed configures RunMat's host MT19937 stream; the explicit device RNG is independent. MATLAB state-wire and normal-transform parity remain separate conformance obligations.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "s = rng(wide_integer_seed[, generator])",
        inputs: &RNG_WIDE_INTEGER_SEED_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "This RunMat-only form is gated before stream mutation and retains an exact typed Seed field when a binary64 field would round.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "s = rng(twister_state_returned_by_rng)",
        inputs: &RNG_INTEGER_STATE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The RunMat MT state is a 625-by-1 uint32 array with its final word storing the next-word index; restore validates all words before mutation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "s = rng(state_with_extended_integer_fields)",
        inputs: &RNG_EXTENDED_STATE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The independently gated extension validates native-integer Seed or legacy LCG State fields exactly.",
    },
];

pub const RNG_OUTPUT_S: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RNG state snapshot struct with fields Type, Seed, and State.",
}];

pub const RNG_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

pub const RNG_INPUTS_SEED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "seed",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Non-negative integer seed.",
}];

pub const RNG_INPUTS_OPTION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Option token: 'default' or 'shuffle'.",
}];

pub const RNG_INPUTS_STATE_STRUCT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "state",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State struct containing Type, optional Seed, and State fields.",
}];

pub const RNG_INPUTS_SEED_GENERATOR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "seed",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Non-negative integer seed.",
    },
    BuiltinParamDescriptor {
        name: "generator",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"twister\""),
        description: "Generator token: 'twister'/'default'; 'runmat-lcg-v1' is accepted only in a saved legacy state.",
    },
];

pub const RNG_INPUTS_OPTION_GENERATOR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option token: 'default' or 'shuffle'.",
    },
    BuiltinParamDescriptor {
        name: "generator",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"twister\""),
        description: "Generator token: 'twister'/'default'; 'runmat-lcg-v1' is accepted only in a saved legacy state.",
    },
];

pub const RNG_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "s = rng()",
        inputs: &RNG_INPUTS_NONE,
        outputs: &RNG_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "s = rng(seed)",
        inputs: &RNG_INPUTS_SEED,
        outputs: &RNG_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "s = rng(option)",
        inputs: &RNG_INPUTS_OPTION,
        outputs: &RNG_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "s = rng(state)",
        inputs: &RNG_INPUTS_STATE_STRUCT,
        outputs: &RNG_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "s = rng(seed, generator)",
        inputs: &RNG_INPUTS_SEED_GENERATOR,
        outputs: &RNG_OUTPUT_S,
    },
    BuiltinSignatureDescriptor {
        label: "s = rng(option, generator)",
        inputs: &RNG_INPUTS_OPTION_GENERATOR,
        outputs: &RNG_OUTPUT_S,
    },
];

pub const RNG_ERROR_SEED_NONNEGATIVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RNG.SEED_NONNEGATIVE",
    identifier: Some("RunMat:rng:SeedMustBeNonnegative"),
    when: "Seed value is negative.",
    message: "rng: seed must be non-negative",
};

pub const RNG_ERROR_GENERATOR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RNG.GENERATOR_UNSUPPORTED",
    identifier: Some("RunMat:rng:GeneratorUnsupported"),
    when: "Generator token is unsupported.",
    message: "rng: generator is not supported",
};

pub const RNG_ERROR_STATE_TYPE_FIELD_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RNG.STATE_TYPE_FIELD_MISSING",
    identifier: Some("RunMat:rng:StateTypeFieldMissing"),
    when: "State struct is missing the Type field.",
    message: "rng: state struct is missing the 'Type' field",
};

pub const RNG_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RNG.INVALID_ARGUMENT",
    identifier: Some("RunMat:rng:InvalidArgument"),
    when: "Arguments are missing, malformed, or incompatible with supported forms.",
    message: "rng: invalid argument",
};

pub const RNG_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RNG.INTERNAL",
    identifier: Some("RunMat:rng:Internal"),
    when: "Internal snapshot conversion/allocation/apply fails.",
    message: "rng: internal operation failed",
};

pub const RNG_ERROR_CONTEXT_REQUIRED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RNG.CONTEXT_REQUIRED",
    identifier: Some("RunMat:rng:ContextRequired"),
    when: "Shuffle is called without an invocation RuntimeContext.",
    message: "rng: shuffle requires an invocation context",
};

pub const RNG_ERRORS: [BuiltinErrorDescriptor; 6] = [
    RNG_ERROR_SEED_NONNEGATIVE,
    RNG_ERROR_GENERATOR_UNSUPPORTED,
    RNG_ERROR_STATE_TYPE_FIELD_MISSING,
    RNG_ERROR_INVALID_ARGUMENT,
    RNG_ERROR_INTERNAL,
    RNG_ERROR_CONTEXT_REQUIRED,
];

pub const RNG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RNG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RNG_ERRORS,
};
