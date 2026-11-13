use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

use super::errors::SETFIELD_ERRORS;

const OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated structure or supported RunMat object-family value.",
}];
const DIRECT: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Structure or supported RunMat object-family target.",
    },
    BuiltinParamDescriptor {
        name: "field",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Field name.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];
const PATH: &[BuiltinParamDescriptor] = &[
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Structure or supported RunMat object-family target.",
    },
    BuiltinParamDescriptor {
        name: "path",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Field names with optional cell-array index selectors.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "S = setfield(S, field, value)",
        inputs: DIRECT,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = setfield(S, field1, ..., fieldN, value)",
        inputs: PATH,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = setfield(S, idx, field1, ..., fieldN, value)",
        inputs: PATH,
        outputs: OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = setfield(S, idx, field1, idx1, ..., fieldN, idxN, value)",
        inputs: PATH,
        outputs: OUTPUT,
    },
];
pub const SETFIELD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: SETFIELD_ERRORS,
};
