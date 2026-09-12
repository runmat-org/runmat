use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

pub(super) const BUILTIN_NAME: &str = "duration";
pub(super) const DURATION_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::standard::DURATION;
pub(super) const DAYS_FIELD: &str = "__days";
pub(super) const FORMAT_FIELD: &str = "Format";
pub(crate) const DEFAULT_DURATION_FORMAT: &str = "hh:mm:ss";
pub(super) const SECONDS_PER_DAY: f64 = 86_400.0;

pub(super) const DURATION_SHORT_COMPONENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "duration-short-component-form",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "duration with one hour component or two hour/minute components is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DurationShortComponentFormExtension"),
    };
pub(super) const DURATION_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "duration-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "duration with resident numeric input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DurationGpuInputExtension"),
    };
pub(super) const DURATION_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DURATION_SHORT_COMPONENT_EXTENSION,
    DURATION_GPU_INPUT_EXTENSION,
];
pub(super) const DURATION_INTEGER_COMPONENT_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    BuiltinIntegerInputCapability {
        name: "H",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric hour arrays can be scalar-expanded; values enter the duration floating representation.",
    },
    BuiltinIntegerInputCapability {
        name: "MI",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric minute arrays can be scalar-expanded; nonscalars must match the other component sizes.",
    },
    BuiltinIntegerInputCapability {
        name: "S",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Numeric second arrays can be scalar-expanded; nonscalars must match the other component sizes.",
    },
    BuiltinIntegerInputCapability {
        name: "MS",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The optional fourth component contributes milliseconds and follows the same scalar-expansion rule.",
    },
];
pub(super) const DURATION_INTEGER_MATRIX_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "X must be a numeric matrix with exactly three columns ordered as hours, minutes, and seconds.",
    }];
pub const DURATION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "D = duration(integer_H, integer_MI, integer_S, integer_MS?)",
        inputs: &DURATION_INTEGER_COMPONENT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "All numeric classes share MATLAB scalar expansion and produce a host duration object backed by binary64 day counts; resident inputs are a separately gated RunMat extension and gather before construction.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "D = duration(integer_X)",
        inputs: &DURATION_INTEGER_MATRIX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each row of the N-by-3 matrix creates one duration; output is an N-by-1 host duration array.",
    },
];

pub(super) static DURATION_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(DURATION_CLASS);

pub(super) const DURATION_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DURATION.INVALID_ARGUMENT",
    identifier: Some("RunMat:duration:InvalidArgument"),
    when: "Arguments or option grammar do not match supported duration forms.",
    message: "duration: invalid argument",
};
pub(super) const DURATION_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DURATION.INVALID_INPUT",
    identifier: Some("RunMat:duration:InvalidInput"),
    when: "Input values cannot be converted/broadcast/formatted to a valid duration result.",
    message: "duration: invalid input",
};
pub(super) const DURATION_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DURATION.INTERNAL",
    identifier: Some("RunMat:duration:Internal"),
    when: "Internal duration state or indexing/evaluation failed unexpectedly.",
    message: "duration: internal operation failed",
};
pub(super) const DURATION_ERRORS: [BuiltinErrorDescriptor; 3] = [
    DURATION_ERROR_INVALID_ARGUMENT,
    DURATION_ERROR_INVALID_INPUT,
    DURATION_ERROR_INTERNAL,
];

pub(super) const OUT_DURATION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Duration object result.",
}];
pub(super) const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Method result.",
}];
pub(super) const DURATION_ARGS_ONLY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Duration constructor arguments.",
}];
pub(super) const DURATION_FOUR_COMPONENT_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "hours",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Hour component.",
    },
    BuiltinParamDescriptor {
        name: "minutes",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minute component.",
    },
    BuiltinParamDescriptor {
        name: "seconds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second component.",
    },
    BuiltinParamDescriptor {
        name: "milliseconds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Millisecond component.",
    },
];
pub(super) const DURATION_BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left duration operand.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right duration/datetime operand.",
    },
];
pub(super) const DURATION_SUBSREF_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Duration receiver object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct-compatible indexing path.",
    },
];
pub(super) const DURATION_SUBSASGN_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Duration receiver object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct-compatible indexing path.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];

pub(super) const DURATION_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "t = duration(X)",
        inputs: &[BuiltinParamDescriptor {
            name: "X",
            ty: BuiltinParamType::NumericArray,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "N-by-3 matrix of hour, minute, and second components.",
        }],
        outputs: &OUT_DURATION,
    },
    BuiltinSignatureDescriptor {
        label: "t = duration(hours, minutes)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "hours",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Hour component.",
            },
            BuiltinParamDescriptor {
                name: "minutes",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Minute component.",
            },
        ],
        outputs: &OUT_DURATION,
    },
    BuiltinSignatureDescriptor {
        label: "t = duration(hours, minutes, seconds)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "hours",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Hour component.",
            },
            BuiltinParamDescriptor {
                name: "minutes",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Minute component.",
            },
            BuiltinParamDescriptor {
                name: "seconds",
                ty: BuiltinParamType::NumericArray,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Second component.",
            },
        ],
        outputs: &OUT_DURATION,
    },
    BuiltinSignatureDescriptor {
        label: "t = duration(hours, minutes, seconds, milliseconds)",
        inputs: &DURATION_FOUR_COMPONENT_INPUTS,
        outputs: &OUT_DURATION,
    },
    BuiltinSignatureDescriptor {
        label: "t = duration(___, \"Format\", format)",
        inputs: &DURATION_ARGS_ONLY,
        outputs: &OUT_DURATION,
    },
    BuiltinSignatureDescriptor {
        label: "t = duration(___, Name, Value, ...)",
        inputs: &DURATION_ARGS_ONLY,
        outputs: &OUT_DURATION,
    },
];
pub(super) const DURATION_SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "out = duration.subsref(obj, S)",
        inputs: &DURATION_SUBSREF_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(super) const DURATION_SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "out = duration.subsasgn(obj, S, rhs)",
        inputs: &DURATION_SUBSASGN_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(super) const DURATION_BINARY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "out = duration.op(lhs, rhs)",
        inputs: &DURATION_BINARY_INPUTS,
        outputs: &OUT_ANY,
    }];

pub const DURATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DURATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DURATION_ERRORS,
};
pub const DURATION_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DURATION_SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DURATION_ERRORS,
};
pub const DURATION_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DURATION_SUBSASGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DURATION_ERRORS,
};
pub const DURATION_BINARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DURATION_BINARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &DURATION_ERRORS,
};
