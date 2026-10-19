use crate::*;

pub const XTICKS_INTEGER_AXES_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "xticks-integer-axes-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Allow a typed-integer alias for an encoded axes handle",
    error_identifier: Some("RunMat:compatibility:XticksIntegerAxesHandleExtension"),
};
pub const XTICKS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    XTICKS_INTEGER_AXES_EXTENSION,
    AXES_CONTROL_SETTER_OUTPUT_EXTENSION,
];
const XTICKS_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ticks",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight integer classes are documented for increasing tick vectors.",
}];
const XTICKS_INTEGER_AXES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ax",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed-integer aliases for encoded axes handles are separately gated.",
    }];
pub const XTICKS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "ticks = xticks(integer_ticks)", inputs: &XTICKS_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Integer ticks are checked in their native class for strict increase before crossing the graphics coordinate boundary; queried values are double." },
    BuiltinIntegerCapabilityDescriptor { form: "ticks = xticks(integer_ax, ticks)", inputs: &XTICKS_INTEGER_AXES_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Strict mode rejects the encoded-handle alias before graphics state access." },
];

const XTICKS_OUTPUT_TICKS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ticks",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis tick values.",
}];

const XTICKS_OUTPUT_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current X-axis tick mode, 'auto' or 'manual'.",
}];

const XTICKS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XTICKS_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];

const XTICKS_INPUTS_TICKS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ticks",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Increasing numeric vector of X-axis tick values; [] removes tick marks.",
}];

const XTICKS_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tick mode string: 'auto', 'manual', or 'mode'.",
}];

const XTICKS_INPUTS_AX_TICKS: [BuiltinParamDescriptor; 2] = [
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
        description: "Increasing numeric vector of X-axis tick values.",
    },
];

const XTICKS_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const XTICKS_INPUTS_AX_MODE: [BuiltinParamDescriptor; 2] = [
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

const XTICKS_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "ticks = xticks()",
        inputs: &XTICKS_INPUTS_NONE,
        outputs: &XTICKS_OUTPUT_TICKS,
    },
    BuiltinSignatureDescriptor {
        label: "ticks = xticks(ax)",
        inputs: &XTICKS_INPUTS_AX,
        outputs: &XTICKS_OUTPUT_TICKS,
    },
    BuiltinSignatureDescriptor {
        label: "xticks(ticks)",
        inputs: &XTICKS_INPUTS_TICKS,
        outputs: &XTICKS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "mode = xticks('mode')",
        inputs: &XTICKS_INPUTS_MODE,
        outputs: &XTICKS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "mode = xticks(ax, 'mode')",
        inputs: &XTICKS_INPUTS_AX_MODE,
        outputs: &XTICKS_OUTPUT_MODE,
    },
    BuiltinSignatureDescriptor {
        label: "xticks('auto'|'manual')",
        inputs: &XTICKS_INPUTS_MODE,
        outputs: &XTICKS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "xticks(ax, 'auto'|'manual')",
        inputs: &XTICKS_INPUTS_AX_MODE,
        outputs: &XTICKS_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "xticks(ax, ticks)",
        inputs: &XTICKS_INPUTS_AX_TICKS,
        outputs: &XTICKS_OUTPUTS_NONE,
    },
];

const XTICKS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKS.INVALID_ARGUMENT",
    identifier: Some("RunMat:xticks:InvalidArgument"),
    when: "Argument count, tick vector, mode, or axes handle is invalid.",
    message: "xticks: invalid argument",
};

const XTICKS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKS.INTERNAL",
    identifier: Some("RunMat:xticks:Internal"),
    when: "Internal plotting state update fails.",
    message: "xticks: internal operation failed",
};

const XTICKS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    XTICKS_ERROR_INVALID_ARGUMENT,
    XTICKS_ERROR_INTERNAL,
    XTICKS_ERROR_TOO_MANY_OUTPUTS,
];

pub const XTICKS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &XTICKS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &XTICKS_ERRORS,
};
pub const XTICKS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.XTICKS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:xticks:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "xticks: at most one output is supported",
};
