use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

use super::errors::ERRORS;

const OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One collected value for each requested callback output.",
}];
const FUNC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "func",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle or function name.",
};
const STRUCT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar struct whose fields are mapped in field order.",
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
    description: "Logical-compatible scalar selecting uniform output.",
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
const BASIC: &[BuiltinParamDescriptor] = &[FUNC, STRUCT];
const UNIFORM: &[BuiltinParamDescriptor] = &[FUNC, STRUCT, UNIFORM_NAME, UNIFORM_VALUE];
const ERROR_HANDLER: &[BuiltinParamDescriptor] = &[FUNC, STRUCT, ERROR_NAME, ERROR_VALUE];
const BOTH: &[BuiltinParamDescriptor] = &[
    FUNC,
    STRUCT,
    UNIFORM_NAME,
    UNIFORM_VALUE,
    ERROR_NAME,
    ERROR_VALUE,
];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "A = structfun(func, S)",
        inputs: BASIC,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = structfun(func, S, \"UniformOutput\", tf)",
        inputs: UNIFORM,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = structfun(func, S, \"ErrorHandler\", errfunc)",
        inputs: ERROR_HANDLER,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = structfun(func, S, \"UniformOutput\", tf, \"ErrorHandler\", errfunc)",
        inputs: BOTH,
        outputs: OUTPUT,
    },
];

pub const STRUCTFUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
