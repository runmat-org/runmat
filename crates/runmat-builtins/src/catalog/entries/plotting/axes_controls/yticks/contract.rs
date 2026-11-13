use crate::*;

pub const YTICKS_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "yticks-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a typed-integer alias for an encoded axes handle",
    error_identifier: Some("RunMat:compatibility:YticksIntegerAxesHandleExtension"),
};
pub const YTICKS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    YTICKS_INTEGER_AXES_EXTENSION,
    AXES_CONTROL_SETTER_OUTPUT_EXTENSION,
];
const YTICKS_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ticks",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are documented for increasing tick vectors.",
}];
const YTICKS_INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for encoded axes handles are separately gated.",
    }];
pub const YTICKS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "ticks = yticks(integer_ticks)", inputs: &YTICKS_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Integer ticks are checked in their native class for strict increase before crossing the graphics coordinate boundary; queried values are double." },
    BuiltinIntegerCapabilityDescriptor { form: "ticks = yticks(integer_ax, ticks)", inputs: &YTICKS_INTEGER_AXES_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Strict mode rejects the encoded-handle alias before graphics state access." },
];

const YTICKS_OUTPUT_TICKS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ticks",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis tick values.",
}];

const YTICKS_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current Y-axis tick mode, 'auto' or 'manual'.",
}];

const YTICKS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YTICKS_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];

const YTICKS_INPUTS_TICKS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ticks",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Increasing numeric vector of Y-axis tick values; [] removes tick marks.",
}];

const YTICKS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tick mode string: 'auto', 'manual', or 'mode'.",
}];

const YTICKS_INPUTS_AX_TICKS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "ticks",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Increasing numeric vector of Y-axis tick values.",
    },
];

const YTICKS_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const YTICKS_INPUTS_AX_MODE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target axes handle.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Tick mode token.",
    },
];

const YTICKS_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "ticks = yticks()",
        inputs: &YTICKS_INPUTS_NONE,
        outputs: &YTICKS_OUTPUT_TICKS,
    },
    BuiltinSignatureDescriptor {
        label: "ticks = yticks(ax)",
        inputs: &YTICKS_INPUTS_AX,
        outputs: &YTICKS_OUTPUT_TICKS,
    },
    BuiltinSignatureDescriptor {
        label: "yticks(ticks)",
        inputs: &YTICKS_INPUTS_TICKS,
        outputs: &YTICKS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "mode = yticks('mode')",
        inputs: &YTICKS_INPUTS_MODE,
        outputs: &YTICKS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "mode = yticks(ax, 'mode')",
        inputs: &YTICKS_INPUTS_AX_MODE,
        outputs: &YTICKS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "yticks('auto'|'manual')",
        inputs: &YTICKS_INPUTS_MODE,
        outputs: &YTICKS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "yticks(ax, 'auto'|'manual')",
        inputs: &YTICKS_INPUTS_AX_MODE,
        outputs: &YTICKS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "yticks(ax, ticks)",
        inputs: &YTICKS_INPUTS_AX_TICKS,
        outputs: &YTICKS_OUTPUTS_NONE,
    },
];

const YTICKS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKS.INVALID_ARGUMENT",
    identifier: Some("RunMat:yticks:InvalidArgument"),
    when: "Argument count, tick vector, mode, or axes handle is invalid.",
    message: "yticks: invalid argument",
};

const YTICKS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKS.INTERNAL",
    identifier: Some("RunMat:yticks:Internal"),
    when: "Internal plotting state update fails.",
    message: "yticks: internal operation failed",
};

const YTICKS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    YTICKS_ERROR_INVALID_ARGUMENT,
    YTICKS_ERROR_INTERNAL,
    YTICKS_ERROR_TOO_MANY_OUTPUTS,
];

pub const YTICKS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &YTICKS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &YTICKS_ERRORS,
};
pub const YTICKS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.YTICKS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:yticks:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "yticks: at most one output is supported",
};
