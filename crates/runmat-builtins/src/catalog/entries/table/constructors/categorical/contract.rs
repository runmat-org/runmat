use crate::*;
const OUTPUT:[BuiltinParamDescriptor;1]=[BuiltinParamDescriptor{name:"C",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,description:"Categorical object with double Codes preserving observation geometry, Categories and logical flags."}];
const INPUT_0: [BuiltinParamDescriptor; 0] = [];
const INPUT_1: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text, real numeric, logical, cell labels, or a categorical source.",
}];
const INPUT_2: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Source observations.",
    },
    BuiltinParamDescriptor {
        name: "valueset",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unique same-class source values in category order.",
    },
];
const INPUT_3: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Source observations.",
    },
    BuiltinParamDescriptor {
        name: "valueset",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unique source values.",
    },
    BuiltinParamDescriptor {
        name: "catnames",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonmissing category names; repeats merge source levels.",
    },
];
const INPUT_4: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Positional source/valueset/names followed by Ordinal and Protected scalar flags.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "C = categorical()",
        inputs: &INPUT_0,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = categorical(A)",
        inputs: &INPUT_1,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = categorical(A, valueset)",
        inputs: &INPUT_2,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = categorical(A, valueset, catnames)",
        inputs: &INPUT_3,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = categorical(..., Name=Value)",
        inputs: &INPUT_4,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    TABLE_ERROR_INVALID_ARGUMENT,
    TABLE_ERROR_INVALID_VARIABLE,
    TABLE_ERROR_INVALID_INDEX,
];
pub const CATEGORICAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
