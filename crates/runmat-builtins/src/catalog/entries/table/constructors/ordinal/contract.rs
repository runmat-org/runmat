use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Categorical object with meaningful category order and protected ordinal metadata.",
}];
const INPUT_0: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Nonmissing observations determine sorted ordinal levels.",
}];
const INPUT_1: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Source observations.",
    },
    BuiltinParamDescriptor {
        name: "labels",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One text label for each inferred level.",
    },
];
const INPUT_2: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Source observations.",
    },
    BuiltinParamDescriptor {
        name: "labels",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One text label for each explicit level.",
    },
    BuiltinParamDescriptor {
        name: "levels",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unique comparable level vector.",
    },
];
const INPUT_3: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real numeric observations.",
    },
    BuiltinParamDescriptor {
        name: "labels",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One text label for each bin.",
    },
    BuiltinParamDescriptor {
        name: "levels",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Empty levels placeholder.",
    },
    BuiltinParamDescriptor {
        name: "edges",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Strictly increasing real numeric vector of length numel(labels)+1.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "O = ordinal(X)",
        inputs: &INPUT_0,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "O = ordinal(X, labels)",
        inputs: &INPUT_1,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "O = ordinal(X, labels, levels)",
        inputs: &INPUT_2,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "O = ordinal(X, labels, [], edges)",
        inputs: &INPUT_3,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    TABLE_ERROR_INVALID_ARGUMENT,
    TABLE_ERROR_INVALID_VARIABLE,
    TABLE_ERROR_INVALID_INDEX,
];
pub const ORDINAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
