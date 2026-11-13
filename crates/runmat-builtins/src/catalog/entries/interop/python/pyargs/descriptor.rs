use super::super::descriptor::LOCAL_ERRORS;
use crate::*;

const KEYWORD_INPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonempty keyword name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value retained in the keyword bundle.",
    },
];

const BUNDLE: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "bundle",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat.PythonArguments object with Names and Values cells.",
}];

const PYARGS_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "bundle = pyargs(name, value, ...)",
    inputs: KEYWORD_INPUTS,
    outputs: BUNDLE,
}];

pub const PYARGS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: PYARGS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: LOCAL_ERRORS,
};
