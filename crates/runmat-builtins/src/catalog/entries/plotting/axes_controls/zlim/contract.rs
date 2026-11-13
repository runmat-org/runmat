use crate::*;
pub const ZLIM_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [AXES_CONTROL_SETTER_OUTPUT_EXTENSION];

const ZLIM_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "limits",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are documented for the two-element limits vector.",
}];
pub const ZLIM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "limits = zlim(integer_limits)", inputs: &ZLIM_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "RunMat compares same-class integer endpoints exactly, then validates that they remain distinct in the graphics coordinate domain; queried limits are double." }];

const ZLIM_OUTPUT_LIMITS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Displayed two-element vector [zmin zmax], including automatically selected bounds.",
}];
const ZLIM_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current automatic or manual axis-limit mode.",
}];
const ZLIM_OUTPUT_METHOD: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current tick-aligned, tight, or padded automatic limit method.",
}];

const ZLIM_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const ZLIM_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const ZLIM_INPUTS_LIMIT_VECTOR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "limits",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element numeric vector [zmin zmax].",
}];

const ZLIM_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axis-limit text token: 'auto'/'manual' set mode; 'tight'/'padded'/'tickaligned' select an automatic method; 'mode'/'method' query state.",
}];

const ZLIM_TARGET: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "axes",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existing axes handle; does not change the current axes.",
};
const ZLIM_INPUTS_TARGET: [BuiltinParamDescriptor; 1] = [ZLIM_TARGET];
const ZLIM_INPUTS_TARGET_LIMITS: [BuiltinParamDescriptor; 2] = [
    ZLIM_TARGET,
    BuiltinParamDescriptor {
        name: "limits",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element numeric vector [zmin zmax].",
    },
];
const ZLIM_INPUTS_TARGET_MODE: [BuiltinParamDescriptor; 2] = [
    ZLIM_TARGET,
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Limit mode or method token.",
    },
];

const ZLIM_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "limits = zlim()",
        inputs: &ZLIM_INPUTS_NONE,
        outputs: &ZLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "zlim([zmin zmax])",
        inputs: &ZLIM_INPUTS_LIMIT_VECTOR,
        outputs: &ZLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "mode = zlim('mode')",
        inputs: &ZLIM_INPUTS_MODE,
        outputs: &ZLIM_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "method = zlim('method')",
        inputs: &ZLIM_INPUTS_MODE,
        outputs: &ZLIM_OUTPUT_METHOD,
    },
    BuiltinSignatureDescriptor {
        label: "zlim('auto'|'manual')",
        inputs: &ZLIM_INPUTS_MODE,
        outputs: &ZLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "zlim('tickaligned'|'tight'|'padded')",
        inputs: &ZLIM_INPUTS_MODE,
        outputs: &ZLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "limits = zlim(axes)",
        inputs: &ZLIM_INPUTS_TARGET,
        outputs: &ZLIM_OUTPUT_LIMITS,
    },
    BuiltinSignatureDescriptor {
        label: "zlim(axes,[zmin zmax])",
        inputs: &ZLIM_INPUTS_TARGET_LIMITS,
        outputs: &ZLIM_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "zlim(axes,mode)",
        inputs: &ZLIM_INPUTS_TARGET_MODE,
        outputs: &ZLIM_OUTPUTS_NONE,
    },
];

const ZLIM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZLIM.INVALID_ARGUMENT",
    identifier: Some("RunMat:zlim:InvalidArgument"),
    when: "Argument count, mode, or limit vector is invalid.",
    message: "zlim: invalid argument",
};

const ZLIM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZLIM.INTERNAL",
    identifier: Some("RunMat:zlim:Internal"),
    when: "Internal plotting state update fails.",
    message: "zlim: internal operation failed",
};

const ZLIM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ZLIM_ERROR_INVALID_ARGUMENT,
    ZLIM_ERROR_INTERNAL,
    ZLIM_ERROR_TOO_MANY_OUTPUTS,
];

pub const ZLIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZLIM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZLIM_ERRORS,
};
pub const ZLIM_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ZLIM.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:zlim:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "zlim: at most one output is supported",
};
