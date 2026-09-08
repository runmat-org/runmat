use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

use super::errors::ERRORS;

const OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "S2",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Structure or represented structure array with the selected fields removed.",
}];
const INPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input structure or represented structure array.",
    },
    BuiltinParamDescriptor {
        name: "fields",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Character row, string value, string array, or cell collection of field names.",
    },
];
const VARIADIC_INPUTS: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input structure or represented structure array.",
    },
    BuiltinParamDescriptor {
        name: "fields",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First field-name value or collection.",
    },
    BuiltinParamDescriptor {
        name: "more_fields",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional RunMat-only field-name arguments.",
    },
];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "S2 = rmfield(S, fields)",
        inputs: INPUTS,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S2 = rmfield(S, fields, ...)",
        inputs: VARIADIC_INPUTS,
        outputs: OUTPUT,
    },
];

pub const RMFIELD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
