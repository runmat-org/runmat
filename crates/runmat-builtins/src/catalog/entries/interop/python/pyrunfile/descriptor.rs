use super::super::descriptor::{FOREIGN_ERRORS, NONE, RESULT};
use crate::*;

const FILE_INPUT: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Python script file.",
    },
    BuiltinParamDescriptor {
        name: "inputs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Named inputs and optional Arguments cell of script arguments.",
    },
];

const RUNFILE_INPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "file",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Python script file.",
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
        description: "Named inputs and optional Arguments cell of script arguments.",
    },
];

const PYRUNFILE_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "pyrunfile(file, name, value, ...)",
        inputs: FILE_INPUT,
        outputs: NONE,
    },
    BuiltinSignatureDescriptor {
        label: "result = pyrunfile(file, outputs, name, value, ...)",
        inputs: RUNFILE_INPUTS,
        outputs: RESULT,
    },
];

pub const PYRUNFILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: PYRUNFILE_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: FOREIGN_ERRORS,
};
