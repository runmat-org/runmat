use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

use super::errors::ERRORS;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Uniform scalar results as an array, or nonuniform results as a cell array.",
}];
const BASE_INPUTS: [BuiltinParamDescriptor; 3] = [
    callable(),
    first_array(),
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional element-wise array inputs.",
    },
];
const OPTION_INPUTS: [BuiltinParamDescriptor; 4] = [
    callable(),
    first_array(),
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional element-wise array inputs.",
    },
    BuiltinParamDescriptor {
        name: "nameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "`UniformOutput` and `ErrorHandler` name-value pairs.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = arrayfun(func, A1, An...)",
        inputs: &BASE_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = arrayfun(func, A1, An..., nameValue...)",
        inputs: &OPTION_INPUTS,
        outputs: &OUTPUT,
    },
];

const fn callable() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar callback function handle.",
    }
}

const fn first_array() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First array input.",
    }
}

pub const ARRAYFUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
