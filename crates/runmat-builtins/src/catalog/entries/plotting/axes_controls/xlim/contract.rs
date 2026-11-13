use crate::*;
pub const XLIM_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [AXES_CONTROL_SETTER_OUTPUT_EXTENSION];

const XLIM_INTEGER_LIMITS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "limits",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented for the two-element limits vector.",
    }];
pub const XLIM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "limits = xlim(integer_limits)",
        inputs: &XLIM_INTEGER_LIMITS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat compares same-class integer endpoints exactly, then validates that they remain distinct when converted to the graphics coordinate domain; queried limits are double.",
    }];

const XLIM_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Displayed two-element vector [xmin xmax], including automatically selected bounds.",
}];
const XLIM_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current automatic or manual axis-limit mode.",
}];
const XLIM_OUTPUT_METHOD: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current tick-aligned, tight, or padded automatic limit method.",
}];

const XLIM_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const XLIM_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XLIM_LIMIT_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element numeric vector [xmin xmax].",
};
const XLIM_INPUTS_LIMIT_VECTOR: [BuiltinParamDescriptor; 1] = [XLIM_LIMIT_PARAM];

const XLIM_MODE_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axis-limit text token: 'auto'/'manual' set mode; 'tight'/'padded'/'tickaligned' select an automatic method; 'mode'/'method' query state.",
};
const XLIM_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [XLIM_MODE_PARAM];

const XLIM_AXES_TARGET: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "axes",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existing axes handle; does not change the current axes.",
};
const XLIM_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [XLIM_AXES_TARGET];
const XLIM_INPUTS_TARGET_LIMITS: [BuiltinParamDescriptor; 2] = [XLIM_AXES_TARGET, XLIM_LIMIT_PARAM];
const XLIM_INPUTS_TARGET_MODE: [BuiltinParamDescriptor; 2] = [XLIM_AXES_TARGET, XLIM_MODE_PARAM];

const XLIM_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "limits = xlim()",
        inputs: &XLIM_INPUTS_NONE,
        outputs: &XLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "xlim([xmin xmax])",
        inputs: &XLIM_INPUTS_LIMIT_VECTOR,
        outputs: &XLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "mode = xlim('mode')",
        inputs: &XLIM_INPUTS_MODE,
        outputs: &XLIM_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "method = xlim('method')",
        inputs: &XLIM_INPUTS_MODE,
        outputs: &XLIM_OUTPUT_METHOD,
    },
    BuiltinSignatureDescriptor {
        label: "xlim('auto'|'manual')",
        inputs: &XLIM_INPUTS_MODE,
        outputs: &XLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "xlim('tickaligned'|'tight'|'padded')",
        inputs: &XLIM_INPUTS_MODE,
        outputs: &XLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "limits = xlim(axes)",
        inputs: &XLIM_INPUTS_TARGET,
        outputs: &XLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "xlim(axes,[xmin xmax])",
        inputs: &XLIM_INPUTS_TARGET_LIMITS,
        outputs: &XLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "xlim(axes,mode)",
        inputs: &XLIM_INPUTS_TARGET_MODE,
        outputs: &XLIM_OUTPUTS_NONE,
    },
];

const XLIM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLIM.INVALID_ARGUMENT",
    identifier: Some("RunMat:xlim:InvalidArgument"),
    when: "Argument count, mode, or limit vector is invalid.",
    message: "xlim: invalid argument",
};

const XLIM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLIM.INTERNAL",
    identifier: Some("RunMat:xlim:Internal"),
    when: "Internal plotting state update fails.",
    message: "xlim: internal operation failed",
};

const XLIM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    XLIM_ERROR_INVALID_ARGUMENT,
    XLIM_ERROR_INTERNAL,
    XLIM_ERROR_TOO_MANY_OUTPUTS,
];

pub const XLIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XLIM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XLIM_ERRORS,
};
pub const XLIM_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XLIM.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:xlim:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "xlim: at most one output is supported",
};
