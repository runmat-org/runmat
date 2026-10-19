use crate::*;
pub const COLORBAR_LEGACY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor { id: "colorbar-toggle-state", mode: BuiltinExtensionMode::RunMatOnly, description: "RunMat profile preserves no-argument toggle and logical output; MATLAB profile enables and returns a registered ColorBar handle, while off deletes without output.", error_identifier: Some("RunMat:compatibility:ColorbarToggleStateExtension") };
pub const COLORBAR_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [COLORBAR_LEGACY_EXTENSION];
pub const COLORBAR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
 name: "ax", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented,
 scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
 notes: "Local numeric axes identity representation accepts integer scalars; validation resolves a live registered axes. Integer arrays are not plotted data.",
}];
pub const COLORBAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
 form: "colorbar(ax, ...)", inputs: &COLORBAR_INTEGER_INPUTS,
 computation_domain: BuiltinIntegerComputationDomain::Structural,
 output_class: BuiltinIntegerOutputClassRule::NotApplicable,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostOnly,
 overload: BuiltinIntegerOverloadKind::StructuralParameter,
 notes: "The shared graphics handle resolver owns identity validation; values do not execute integer arithmetic, create handles from arbitrary integers, or transfer device ownership.",
}];
pub const COLORBAR_OUTPUT_ENABLED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enabled",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Colorbar enabled state after command execution.",
}];
pub const COLORBAR_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const MODE_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Colorbar mode token ('on'|'off').",
};
pub const COLORBAR_INPUTS_MODE: [BuiltinParamDescriptor; 1] = [MODE_INPUT];
pub const COLORBAR_INPUTS_TARGET_MODE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "ax",
        ty: BuiltinParamType::AxesHandle,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Associated axes, without changing current selection.",
    },
    MODE_INPUT,
];
pub const COLORBAR_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Registered ColorBar identity in MATLAB profile, or logical state in RunMat profile.",
}];
pub const COLORBAR_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "c = colorbar(ax, mode)",
        inputs: &COLORBAR_INPUTS_TARGET_MODE,
        outputs: &COLORBAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "c = colorbar()",
        inputs: &COLORBAR_INPUTS_NONE,
        outputs: &COLORBAR_OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = colorbar()",
        inputs: &COLORBAR_INPUTS_NONE,
        outputs: &COLORBAR_OUTPUT_ENABLED,
    },
    BuiltinSignatureDescriptor {
        label: "enabled = colorbar(mode)",
        inputs: &COLORBAR_INPUTS_MODE,
        outputs: &COLORBAR_OUTPUT_ENABLED,
    },
];
pub const COLORBAR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORBAR.INVALID_ARGUMENT",
    identifier: Some("RunMat:colorbar:InvalidArgument"),
    when: "Colorbar mode argument is unsupported.",
    message: "colorbar: invalid argument",
};
pub const COLORBAR_ERRORS: [BuiltinErrorDescriptor; 2] =
    [COLORBAR_ERROR_INVALID_ARGUMENT, COLORBAR_ERROR_OUTPUTS];
pub const COLORBAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COLORBAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COLORBAR_ERRORS,
};
pub const COLORBAR_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLORBAR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:colorbar:TooManyOutputs"),
    when: "More than one output is requested, or off requests an output in MATLAB profile.",
    message: "colorbar: at most one output is supported",
};
