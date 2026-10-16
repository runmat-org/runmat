use crate::*;
pub const YLIM_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [AXES_CONTROL_SETTER_OUTPUT_EXTENSION];

const YLIM_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "limits",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are documented for the two-element limits vector.",
}];
pub const YLIM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "limits = ylim(integer_limits)", inputs: &YLIM_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat compares same-class integer endpoints exactly, then validates that they remain distinct in the graphics coordinate domain; queried limits are double." }];

const YLIM_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Displayed two-element vector [ymin ymax], including automatically selected bounds.",
}];
const YLIM_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current automatic or manual axis-limit mode.",
}];
const YLIM_OUTPUT_METHOD: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current tick-aligned, tight, or padded automatic limit method.",
}];

const YLIM_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const YLIM_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YLIM_INPUTS_LIMIT_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element numeric vector [ymin ymax].",
}];

const YLIM_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axis-limit text token: 'auto'/'manual' set mode; 'tight'/'padded'/'tickaligned' select an automatic method; 'mode'/'method' query state.",
}];

const YLIM_TARGET: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "axes",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existing axes handle; does not change the current axes.",
};
const YLIM_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [YLIM_TARGET];
const YLIM_INPUTS_TARGET_LIMITS: [BuiltinParamDescriptor; 2] = [
    YLIM_TARGET,
    BuiltinParamDescriptor {
        name: "limits",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element numeric vector [ymin ymax].",
    },
];
const YLIM_INPUTS_TARGET_MODE: [BuiltinParamDescriptor; 2] = [
    YLIM_TARGET,
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Limit mode or method token.",
    },
];

const YLIM_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "limits = ylim()",
        inputs: &YLIM_INPUTS_NONE,
        outputs: &YLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "ylim([ymin ymax])",
        inputs: &YLIM_INPUTS_LIMIT_VECTOR,
        outputs: &YLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "mode = ylim('mode')",
        inputs: &YLIM_INPUTS_MODE,
        outputs: &YLIM_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "method = ylim('method')",
        inputs: &YLIM_INPUTS_MODE,
        outputs: &YLIM_OUTPUT_METHOD,
    },
    BuiltinSignatureDescriptor {
        label: "ylim('auto'|'manual')",
        inputs: &YLIM_INPUTS_MODE,
        outputs: &YLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "ylim('tickaligned'|'tight'|'padded')",
        inputs: &YLIM_INPUTS_MODE,
        outputs: &YLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "limits = ylim(axes)",
        inputs: &YLIM_INPUTS_TARGET,
        outputs: &YLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "ylim(axes,[ymin ymax])",
        inputs: &YLIM_INPUTS_TARGET_LIMITS,
        outputs: &YLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "ylim(axes,mode)",
        inputs: &YLIM_INPUTS_TARGET_MODE,
        outputs: &YLIM_OUTPUTS_NONE,
    },
];

const YLIM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLIM.INVALID_ARGUMENT",
    identifier: Some("RunMat:ylim:InvalidArgument"),
    when: "Argument count, mode, or limit vector is invalid.",
    message: "ylim: invalid argument",
};

const YLIM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLIM.INTERNAL",
    identifier: Some("RunMat:ylim:Internal"),
    when: "Internal plotting state update fails.",
    message: "ylim: internal operation failed",
};

const YLIM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    YLIM_ERROR_INVALID_ARGUMENT,
    YLIM_ERROR_INTERNAL,
    YLIM_ERROR_TOO_MANY_OUTPUTS,
];

pub const YLIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &YLIM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &YLIM_ERRORS,
};
pub const YLIM_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YLIM.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ylim:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "ylim: at most one output is supported",
};
