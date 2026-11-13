use crate::*;

const PZMAP_OUTPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Poles of the SISO tf or ss model as a column vector.",
};
const PZMAP_OUTPUT_Z: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zeros of the SISO tf or ss model as a column vector.",
};
const PZMAP_INPUT_SYS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf or ss model.",
};
const PZMAP_OUTPUTS_P: [BuiltinParamDescriptor; 1] = [PZMAP_OUTPUT_P];
const PZMAP_OUTPUTS_P_Z: [BuiltinParamDescriptor; 2] = [PZMAP_OUTPUT_P, PZMAP_OUTPUT_Z];
const PZMAP_INPUTS_SYS: [BuiltinParamDescriptor; 1] = [PZMAP_INPUT_SYS];
const PZMAP_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "pzmap(sys)",
        inputs: &PZMAP_INPUTS_SYS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "p = pzmap(sys)",
        inputs: &PZMAP_INPUTS_SYS,
        outputs: &PZMAP_OUTPUTS_P,
    },
    BuiltinSignatureDescriptor {
        label: "[p,z] = pzmap(sys)",
        inputs: &PZMAP_INPUTS_SYS,
        outputs: &PZMAP_OUTPUTS_P_Z,
    },
];
pub const PZMAP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PZMAP.INVALID_ARGUMENT",
    identifier: Some("RunMat:pzmap:InvalidArgument"),
    when: "Inputs do not match supported pzmap invocation forms.",
    message: "pzmap: invalid argument",
};
pub const PZMAP_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PZMAP.INVALID_MODEL",
    identifier: Some("RunMat:pzmap:InvalidModel"),
    when: "Input system is not a valid SISO tf or ss object.",
    message: "pzmap: invalid model",
};
pub const PZMAP_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PZMAP.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:pzmap:UnsupportedModel"),
    when: "Model form is not supported by the current implementation.",
    message: "pzmap: unsupported model",
};
pub const PZMAP_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PZMAP.PLOT_FAILED",
    identifier: Some("RunMat:pzmap:PlotFailed"),
    when: "Statement-form plotting failed for reasons other than known nonfatal setup conditions.",
    message: "pzmap: plotting failed",
};
pub const PZMAP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PZMAP.INTERNAL",
    identifier: Some("RunMat:pzmap:Internal"),
    when: "Pole-zero extraction or output construction failed.",
    message: "pzmap: internal error",
};
const PZMAP_ERRORS: [BuiltinErrorDescriptor; 5] = [
    PZMAP_ERROR_INVALID_ARGUMENT,
    PZMAP_ERROR_INVALID_MODEL,
    PZMAP_ERROR_UNSUPPORTED_MODEL,
    PZMAP_ERROR_PLOT_FAILED,
    PZMAP_ERROR_INTERNAL,
];
pub const PZMAP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PZMAP_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PZMAP_ERRORS,
};
const PZMAP_INTEGER_SYS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sys",
    classes: &[],
    availability: BuiltinIntegerInputAvailability::Rejected,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "pzmap accepts a dynamic-system object. Integer coefficient support belongs to the tf/ss constructors and does not make an integer array a valid sys input.",
}];
pub const PZMAP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[p,z] = pzmap(integer_sys)",
        inputs: &PZMAP_INTEGER_SYS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "A direct integer sys is not a dynamic-system object and rejects before provider access. Consumed typed integer properties of an existing tf/ss object have their own model-admission gates and exact numeric checks.",
    }];
