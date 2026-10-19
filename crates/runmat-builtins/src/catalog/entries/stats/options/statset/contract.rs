use crate::*;

const OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Canonical statistics options structure.",
}];
const STATFUN: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "statfun",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Statistics function whose defaults are requested.",
}];
const PAIRS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value and subsequent name-value pairs.",
    },
];
const OLD_NEW: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "oldopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing statistics options structure.",
    },
    BuiltinParamDescriptor {
        name: "newopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonempty replacement fields override old options.",
    },
];
const OLD_PAIRS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "oldopts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Existing statistics options structure.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Option field name or replacement options structure.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value and subsequent name-value pairs.",
    },
];
pub const STATSET_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STATSET.INVALID_ARGUMENT",
    identifier: Some("RunMat:statset:InvalidArgument"),
    when: "The argument grammar or resident materialization fails.",
    message: "statset: invalid argument",
};
pub const STATSET_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STATSET.INVALID_OPTION",
    identifier: Some("RunMat:statset:InvalidOption"),
    when: "An option name or value is invalid.",
    message: "statset: invalid option",
};
pub const STATSET_ERROR_INVALID_STATFUN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STATSET.INVALID_STATFUN",
    identifier: Some("RunMat:statset:InvalidStatfun"),
    when: "The function-default selector is unsupported.",
    message: "statset: invalid statistics function",
};

pub const STATSET_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "options = statset()",
            inputs: &[],
            outputs: &OUT,
        },
        BuiltinSignatureDescriptor {
            label: "options = statset(statfun)",
            inputs: &STATFUN,
            outputs: &OUT,
        },
        BuiltinSignatureDescriptor {
            label: "options = statset(name, value, ...)",
            inputs: &PAIRS,
            outputs: &OUT,
        },
        BuiltinSignatureDescriptor {
            label: "options = statset(oldopts, newopts)",
            inputs: &OLD_NEW,
            outputs: &OUT,
        },
        BuiltinSignatureDescriptor {
            label: "options = statset(oldopts, name, value, ...)",
            inputs: &OLD_PAIRS,
            outputs: &OUT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        STATSET_ERROR_INVALID_ARGUMENT,
        STATSET_ERROR_INVALID_OPTION,
        STATSET_ERROR_INVALID_STATFUN,
    ],
};

pub const STATSET_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "statset-integer-option-value",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "statset with a native typed-integer option value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StatsetIntegerOptionValueExtension"),
    };
pub const STATSET_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [STATSET_INTEGER_OPTION_EXTENSION];
pub const STATSET_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "options = statset(..., name, integer_value, ...)",
    inputs: &[BuiltinIntegerInputCapability { name: "option values", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "All eight native integer classes are admitted only in RunMat mode after field-specific range checks." }],
    computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
    output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
    overflow: BuiltinIntegerOverflowRule::Error,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "Compatibility admission is recursive and precedes gather. Count fields become double metadata, boolean flags become logical, and DerivStep arrays retain native typed storage.",
}];
