use crate::*;
const GCF_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fig",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current figure handle.",
}];

const GCF_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const GCF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "fig = gcf()",
    inputs: &GCF_INPUTS_NONE,
    outputs: &GCF_OUTPUT_HANDLE,
}];

const GCF_ERRORS: [BuiltinErrorDescriptor; 1] = [GCF_ERROR_TOO_MANY_OUTPUTS];

pub const GCF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GCF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GCF_ERRORS,
};

pub const GCF_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCF.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:gcf:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "gcf: too many output arguments",
};
