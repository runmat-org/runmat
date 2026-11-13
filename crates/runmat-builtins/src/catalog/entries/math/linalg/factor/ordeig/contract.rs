use crate::*;

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Ordinary real quasitriangular or complex triangular Schur form.",
}];
const GENERALIZED_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "AA",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First retained generalized Schur factor.",
    },
    BuiltinParamDescriptor {
        name: "BB",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second retained generalized Schur factor.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "E",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Eigenvalues in Schur-block order.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "E = ordeig(T)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "E = ordeig(AA,BB)",
        inputs: &GENERALIZED_INPUTS,
        outputs: &OUTPUTS,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.ORDEIG.INVALID_INPUT",
    identifier: Some("RunMat:ordeig:InvalidInput"),
    when: "Input factors are not finite ordinary or generalized Schur forms.",
    message: "ordeig: expected Schur-form factors",
}];
pub const ORDEIG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
