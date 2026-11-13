use crate::*;

const INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value to display with a variable-name header when available.",
};
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "display(X)",
    inputs: &[INPUT],
    outputs: &[],
}];
pub const DISPLAY_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISPLAY.ARG_CONFIG",
    identifier: Some("RunMat:display:TooManyInputs"),
    when: "Too many input arguments are passed to display.",
    message: "display: too many input arguments",
};
pub const DISPLAY_ERROR_GATHER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISPLAY.GATHER",
    identifier: Some("RunMat:display:GatherFailed"),
    when: "Input value cannot be gathered onto the host for rendering.",
    message: "display: failed to gather value for display",
};
pub const DISPLAY_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISPLAY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:display:TooManyOutputs"),
    when: "An output is requested from display.",
    message: "display does not return output arguments",
};
pub const DISPLAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        DISPLAY_ERROR_ARG_CONFIG,
        DISPLAY_ERROR_GATHER,
        DISPLAY_ERROR_TOO_MANY_OUTPUTS,
    ],
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every integer class displays exactly unless class-specific display or disp dispatch handles the object.",
}];
pub const DISPLAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "display(X)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "No-output host rendering sink with class-method dispatch. Resident integers gather exactly through the provider before generic formatting.",
}];
