use crate::*;
pub const LINKAXES_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor;1] = [BuiltinIntegerCapabilityDescriptor {
    form:"linkaxes(integer_handles, ...)",
    inputs:&[BuiltinIntegerInputCapability{name:"ax",classes:&ALL_INTEGER_CLASSES,availability:BuiltinIntegerInputAvailability::Documented,scalar_double:BuiltinIntegerScalarDoubleRule::Allowed,notes:"Numeric graphics handle representations accept exact integer scalar and dense array storage."}],
    computation_domain:BuiltinIntegerComputationDomain::Structural,
    output_class:BuiltinIntegerOutputClassRule::NotApplicable,
    overflow:BuiltinIntegerOverflowRule::Error,
    backend:BuiltinIntegerBackendRule::HostOnly,
    overload:BuiltinIntegerOverloadKind::StructuralParameter,
    notes:"Handle conversion requires exact representability, then validates existing axes before shared linked-limit state is changed. No numeric output or provider operation is produced."
}];
const LINKAXES_OUTPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const LINKAXES_INPUTS_AXES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axes handle or numeric array of axes handles to link.",
}];

const LINKAXES_INPUTS_AXES_OPTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Axes handle or numeric array of axes handles to link.",
    },
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Link option: 'x', 'y', 'xy', or 'off'.",
    },
];

const LINKAXES_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "linkaxes(ax)",
        inputs: &LINKAXES_INPUTS_AXES,
        outputs: &LINKAXES_OUTPUTS_NONE,
    },
    BuiltinSignatureDescriptor {
        label: "linkaxes(ax, option)",
        inputs: &LINKAXES_INPUTS_AXES_OPTION,
        outputs: &LINKAXES_OUTPUTS_NONE,
    },
];

const LINKAXES_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINKAXES.INVALID_ARGUMENT",
    identifier: Some("RunMat:linkaxes:InvalidArgument"),
    when: "Axes handles, option, or argument count is invalid.",
    message: "linkaxes: invalid argument",
};

const LINKAXES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINKAXES.INTERNAL",
    identifier: Some("RunMat:linkaxes:Internal"),
    when: "Internal plotting state update fails.",
    message: "linkaxes: internal operation failed",
};

const LINKAXES_ERRORS: [BuiltinErrorDescriptor; 3] = [
    LINKAXES_ERROR_INVALID_ARGUMENT,
    LINKAXES_ERROR_INTERNAL,
    LINKAXES_ERROR_TOO_MANY_OUTPUTS,
];

pub const LINKAXES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LINKAXES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LINKAXES_ERRORS,
};

pub const LINKAXES_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINKAXES.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:linkaxes:TooManyOutputs"),
    when: "Any output is explicitly requested.",
    message: "linkaxes: too many output arguments",
};
