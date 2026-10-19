use crate::*;
pub const CLA_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
 name: "ax", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented,
 scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
 notes: "Local numeric axes identity representation accepts integer scalars; validation resolves a live registered axes. Integer arrays are not plotted data.",
}];
pub const CLA_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
 form: "cla(ax, ...)", inputs: &CLA_INTEGER_INPUTS,
 computation_domain: BuiltinIntegerComputationDomain::Structural,
 output_class: BuiltinIntegerOutputClassRule::NotApplicable,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostOnly,
 overload: BuiltinIntegerOverloadKind::StructuralParameter,
 notes: "The shared graphics handle resolver owns identity validation; values do not execute integer arithmetic, create handles from arbitrary integers, or transfer device ownership.",
}];
pub const CLA_OUTPUT_OK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ok",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when current axes are cleared.",
}];
pub const CLA_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const RESET_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "reset",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "The reset token resets appearance while preserving Position and Units.",
};
pub const CLA_INPUTS_RESET: [BuiltinParamDescriptor; 1] = [RESET_INPUT];
const AX_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ax",
    ty: BuiltinParamType::AxesHandle,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Axes to clear without selecting them.",
};
pub const CLA_INPUTS_AX: [BuiltinParamDescriptor; 1] = [AX_INPUT];
pub const CLA_INPUTS_AX_RESET: [BuiltinParamDescriptor; 2] = [AX_INPUT, RESET_INPUT];
pub const CLA_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "cla(ax)",
        inputs: &CLA_INPUTS_AX,
        outputs: &CLA_OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "cla(reset)",
        inputs: &CLA_INPUTS_RESET,
        outputs: &CLA_OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "cla(ax, reset)",
        inputs: &CLA_INPUTS_AX_RESET,
        outputs: &CLA_OUTPUT_OK,
    },
    BuiltinSignatureDescriptor {
        label: "ok = cla()",
        inputs: &CLA_INPUTS_NONE,
        outputs: &CLA_OUTPUT_OK,
    },
];
pub const CLA_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLA.INVALID_ARGUMENT",
    identifier: Some("RunMat:cla:InvalidArgument"),
    when: "Inputs are not an optional axes handle and reset token.",
    message: "cla: invalid argument",
};
pub const CLA_ERRORS: [BuiltinErrorDescriptor; 2] = [CLA_ERROR_INVALID_ARGUMENT, CLA_ERROR_OUTPUTS];
pub const CLA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLA_ERRORS,
};
pub const CLA_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLA.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:cla:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "cla: at most one output is supported",
};
pub const CLA_STATE_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor { id: "cla-state-output", mode: BuiltinExtensionMode::RunMatOnly, description: "The assigned logical state/success output is a RunMat extension; MATLAB command calls have no output.", error_identifier: Some("RunMat:compatibility:CLAStateOutputExtension") };
pub const CLA_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [CLA_STATE_OUTPUT_EXTENSION];
