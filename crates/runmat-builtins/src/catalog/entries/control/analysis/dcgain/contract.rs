use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "gain",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real or complex steady-state gain at s=0 or z=1.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar SISO tf model.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "gain = dcgain(sys)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.DCGAIN.INVALID_ARGUMENT",
        identifier: Some("RunMat:dcgain:InvalidArgument"),
        when: "Input does not match supported invocation forms.",
        message: "dcgain: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.DCGAIN.INVALID_MODEL",
        identifier: Some("RunMat:dcgain:InvalidModel"),
        when: "Input system is not a valid SISO tf object.",
        message: "dcgain: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.DCGAIN.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:dcgain:UnsupportedModel"),
        when: "Model form is not supported by the current implementation.",
        message: "dcgain: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.DCGAIN.INTERNAL",
        identifier: Some("RunMat:dcgain:Internal"),
        when: "Gain evaluation failed internally.",
        message: "dcgain: internal error",
    },
];
pub const DCGAIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_SYS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "sys", classes: &[], availability: BuiltinIntegerInputAvailability::Rejected, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "dcgain accepts an LTI model object. Integer coefficient support belongs to model constructors and does not make an integer array a valid sys input." }];
pub const DCGAIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
    form: "k = dcgain(integer_sys)", inputs: &INTEGER_SYS,
    computation_domain: BuiltinIntegerComputationDomain::Structural,
    output_class: BuiltinIntegerOutputClassRule::NotApplicable,
    overflow: BuiltinIntegerOverflowRule::NotApplicable,
    backend: BuiltinIntegerBackendRule::HostOnly,
    overload: BuiltinIntegerOverloadKind::Multiple,
    notes: "A direct integer sys rejects at the model-object boundary before provider access. Typed integer properties of an existing tf object are separately gated and checked when this call reads them.",
}];
