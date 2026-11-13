use super::super::descriptor::{FOREIGN_ERRORS, NONE, RESULT};
use crate::*;

const CODE_INPUT: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "code",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Python source for the persistent workspace.",
    },
    BuiltinParamDescriptor {
        name: "inputs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Named Python input pairs.",
    },
];

const RUN_INPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "code",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Python source.",
    },
    BuiltinParamDescriptor {
        name: "outputs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Python output name or names, required when an output is requested.",
    },
    BuiltinParamDescriptor {
        name: "inputs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Named Python input pairs.",
    },
];

const PYRUN_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "pyrun(code, name, value, ...)",
        inputs: CODE_INPUT,
        outputs: NONE,
    },
    BuiltinSignatureDescriptor {
        label: "result = pyrun(code, outputs, name, value, ...)",
        inputs: RUN_INPUTS,
        outputs: RESULT,
    },
];

pub const PYRUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: PYRUN_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: FOREIGN_ERRORS,
};
