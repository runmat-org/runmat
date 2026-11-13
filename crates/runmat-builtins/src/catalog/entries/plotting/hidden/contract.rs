use crate::*;
pub const HIDDEN_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
 name: "ax", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented,
 scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
 notes: "Local numeric axes identity representation accepts integer scalars; validation resolves a live registered axes. Integer arrays are not plotted data.",
}];
pub const HIDDEN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
 form: "hidden(ax, ...)", inputs: &HIDDEN_INTEGER_INPUTS,
 computation_domain: BuiltinIntegerComputationDomain::Structural,
 output_class: BuiltinIntegerOutputClassRule::NotApplicable,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostOnly,
 overload: BuiltinIntegerOverloadKind::StructuralParameter,
 notes: "The shared graphics handle resolver owns identity validation; values do not execute integer arithmetic, create handles from arbitrary integers, or transfer device ownership.",
}];
pub const HIDDEN_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hidden-line-removal state after command execution.",
}];
pub const HIDDEN_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
pub const HIDDEN_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Hidden-line-removal mode token ('on'|'off').",
}];
pub const HIDDEN_INPUTS_AX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Target axes handle.",
}];
pub const HIDDEN_INPUTS_AX_MODE: [BuiltinParamDescriptor; 2] = [
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
        arity: BuiltinParamArity::Optional,
        default: Some("\"toggle\""),
        description: "Hidden-line-removal mode token ('on'|'off').",
    },
];
pub const HIDDEN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "enabled = hidden()",
        inputs: &HIDDEN_INPUTS_NONE,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = hidden(mode)",
        inputs: &HIDDEN_INPUTS_MODE,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = hidden(ax)",
        inputs: &HIDDEN_INPUTS_AX,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = hidden(ax, mode)",
        inputs: &HIDDEN_INPUTS_AX_MODE,
        outputs: &HIDDEN_OUTPUT_ENABLED,
    },
];
pub const HIDDEN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HIDDEN.INVALID_ARGUMENT",
    identifier: Some("RunMat:hidden:InvalidArgument"),
    when: "Hidden-line-removal mode or axes target is unsupported.",
    message: "hidden: invalid argument",
};
pub const HIDDEN_ERRORS: [BuiltinErrorDescriptor; 2] =
    [HIDDEN_ERROR_INVALID_ARGUMENT, HIDDEN_ERROR_OUTPUTS];
pub const HIDDEN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HIDDEN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &HIDDEN_ERRORS,
};
pub const HIDDEN_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HIDDEN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:hidden:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "hidden: at most one output is supported",
};
pub const HIDDEN_STATE_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor { id: "hidden-state-output", mode: BuiltinExtensionMode::RunMatOnly, description: "The assigned logical state/success output is a RunMat extension; MATLAB command calls have no output.", error_identifier: Some("RunMat:compatibility:HIDDENStateOutputExtension") };
pub const HIDDEN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [HIDDEN_STATE_OUTPUT_EXTENSION];
