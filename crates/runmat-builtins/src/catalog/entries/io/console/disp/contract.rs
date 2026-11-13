use crate::*;

const INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value to display in the Command Window.",
};
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "disp(X)",
    inputs: &[INPUT],
    outputs: &[],
}];
pub const DISP_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISP.ARG_CONFIG",
    identifier: None,
    when: "Too many input arguments are passed to disp.",
    message: "disp: too many input arguments",
};
pub const DISP_ERROR_GATHER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISP.GATHER",
    identifier: None,
    when: "Input value cannot be gathered onto the host for rendering.",
    message: "disp: failed to gather value for display",
};
pub const DISP_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISP.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:disp:TooManyOutputs"),
    when: "An output is requested from disp.",
    message: "disp does not return output arguments",
};
pub const DISP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        DISP_ERROR_ARG_CONFIG,
        DISP_ERROR_GATHER,
        DISP_ERROR_TOO_MANY_OUTPUTS,
    ],
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every integer class is formatted from exact authoritative storage, including 64-bit extrema and integer members nested in supported containers.",
}];
pub const DISP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "disp(X)", inputs: &INTEGER_INPUTS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::GatherFallback,
    overload: BuiltinIntegerOverloadKind::FunctionSpecific,
    notes: "No-output host rendering sink. Resident integer values gather exactly through the provider before decimal formatting; no provider output is created.",
}];
