use crate::*;
const CLF_OUTPUT_RESULT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cleared handle for single-target calls or count for multi/all clears.",
}];

const CLF_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const CLF_INPUTS_TARGETS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "targets",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Figure handle(s) or mode tokens ('all', 'reset').",
}];

const CLF_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "result = clf()",
        inputs: &CLF_INPUTS_NONE,
        outputs: &CLF_OUTPUT_RESULT,
    },
    BuiltinSignatureDescriptor {
        label: "result = clf(targets...)",
        inputs: &CLF_INPUTS_TARGETS,
        outputs: &CLF_OUTPUT_RESULT,
    },
];

pub const CLF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLF.INVALID_ARGUMENT",
    identifier: Some("RunMat:clf:InvalidArgument"),
    when: "One or more clear targets are invalid.",
    message: "clf: invalid argument",
};

pub const CLF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLF.INTERNAL",
    identifier: Some("RunMat:clf:Internal"),
    when: "Internal figure clear operation fails.",
    message: "clf: internal operation failed",
};

const CLF_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CLF_ERROR_INVALID_ARGUMENT,
    CLF_ERROR_INTERNAL,
    CLF_ERROR_TOO_MANY_OUTPUTS,
];

pub const CLF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLF_ERRORS,
};

pub const CLF_INTEGER_FIGURE_NUMBER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "clf-integer-figure-number",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "clf with a typed integer figure number is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ClfIntegerFigureNumberExtension"),
    };

pub const CLF_ALL_SELECTOR_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "clf-all-selector",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "clf('all') is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ClfAllSelectorExtension"),
};

pub const CLF_VARIADIC_TARGETS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "clf-variadic-targets",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "clf with separate variadic figure targets is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ClfVariadicTargetsExtension"),
};

pub const CLF_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    CLF_INTEGER_FIGURE_NUMBER_EXTENSION,
    CLF_ALL_SELECTOR_EXTENSION,
    CLF_VARIADIC_TARGETS_EXTENSION,
];

const CLF_INTEGER_TARGET_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "fig",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode interprets a scalar or array of any built-in integer class as figure numbers, not as numeric figure data. The public compatibility contract documents figure numbers but does not advertise typed integer classes.",
    }];

pub const CLF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "f = clf(integer_fig)",
        inputs: &CLF_INTEGER_TARGET_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Figure numbers are read from authoritative host integer storage and must be positive and representable as RunMat's u32 figure identifier. RunMat currently encodes a returned figure opaquely as f64 and returns a scalar count for multiple targets; those are explicit graphics-representation gaps, not integer numeric output. No numeric kernel or provider dispatch occurs.",
    }];

pub const CLF_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLF.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:clf:TooManyOutputs"),
    when: "More than 1 output arguments are requested.",
    message: "clf: too many output arguments",
};
