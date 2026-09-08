use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

use super::errors::ERRORS;

const OUTPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mapped callback output.",
}];

const FUNC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "func",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle, function name, or supported cellfun shorthand.",
};

const CELLS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "One or more equal-sized cell arrays, followed by constant callback arguments.",
};

const UNIFORM_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "UniformOutput",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"UniformOutput\""),
    description: "Name-value key selecting uniform result collection.",
};

const UNIFORM_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("true"),
    description: "Logical scalar, or legacy double zero or one.",
};

const ERROR_NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ErrorHandler",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"ErrorHandler\""),
    description: "Name-value key selecting a callback error handler.",
};

const ERROR_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "errfunc",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Error-handler function handle or name.",
};

const BASIC: &[BuiltinParamDescriptor] = &[FUNC, CELLS];
const UNIFORM: &[BuiltinParamDescriptor] = &[FUNC, CELLS, UNIFORM_NAME, UNIFORM_VALUE];
const ERROR_HANDLER: &[BuiltinParamDescriptor] = &[FUNC, CELLS, ERROR_NAME, ERROR_VALUE];
const BOTH: &[BuiltinParamDescriptor] = &[
    FUNC,
    CELLS,
    UNIFORM_NAME,
    UNIFORM_VALUE,
    ERROR_NAME,
    ERROR_VALUE,
];

const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "Y = cellfun(func, C)",
        inputs: BASIC,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Y = cellfun(func, C1, C2, ...)",
        inputs: BASIC,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Y = cellfun(func, C..., \"UniformOutput\", tf)",
        inputs: UNIFORM,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Y = cellfun(func, C..., \"ErrorHandler\", errfunc)",
        inputs: ERROR_HANDLER,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Y = cellfun(func, C..., \"UniformOutput\", tf, \"ErrorHandler\", errfunc)",
        inputs: BOTH,
        outputs: OUTPUTS,
    },
];

pub const CELLFUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
