use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "d",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar dictionary object with configured Keys/Values entry cells.",
}];
const INPUT_0: [BuiltinParamDescriptor; 0] = [];
const INPUT_1: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "keys",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Uniform or cell-wrapped dictionary keys.",
    },
    BuiltinParamDescriptor {
        name: "values",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Same-size values, or scalar values expanded to each key.",
    },
];
const INPUT_2: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "keyValuePair1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First key/value argument pair.",
    },
    BuiltinParamDescriptor {
        name: "keyValuePairN",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional key/value argument pairs.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "d = dictionary()",
        inputs: &INPUT_0,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "d = dictionary(keys, values)",
        inputs: &INPUT_1,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "d = dictionary(k1, v1, ..., kN, vN)",
        inputs: &INPUT_2,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    TABLE_ERROR_INVALID_ARGUMENT,
    TABLE_ERROR_INVALID_VARIABLE,
    TABLE_ERROR_INVALID_INDEX,
];
pub const DICTIONARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
