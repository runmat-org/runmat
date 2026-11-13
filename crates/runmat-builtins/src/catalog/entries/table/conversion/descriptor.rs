use super::super::shared::{
    TABLE_ERRORS, TABLE_ERROR_INVALID_ARGUMENT, TABLE_ERROR_INVALID_INDEX,
    TABLE_ERROR_INVALID_VARIABLE,
};
use crate::*;

const TABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output table.",
}];
const ANY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Converted output value.",
}];
const INPUT_WITH_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array, cell array, or structure.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Conversion options.",
    },
];
const ARRAY_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Homogeneous array to split by column.",
}];
const CELL_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-dimensional cell array.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "VariableNames and RowNames options.",
    },
];
const STRUCT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar structure or structure array.",
    },
    BuiltinParamDescriptor {
        name: "Name,Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "AsArray, VariableNames, and RowNames options.",
    },
];
const TABLE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input table or timetable.",
}];
const TABLE2STRUCT_INPUT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input table or timetable.",
    },
    BuiltinParamDescriptor {
        name: "ToScalar",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Name of the scalar-structure option.",
    },
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::PropertyValue,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical selector for a scalar structure of complete variables.",
    },
];
const ARRAY2TABLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "T = array2table(A)",
        inputs: &ARRAY_INPUT,
        outputs: &TABLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = array2table(A, Name, Value, ...)",
        inputs: &INPUT_WITH_OPTIONS,
        outputs: &TABLE_OUTPUT,
    },
];
const CELL2TABLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "T = cell2table(C, Name, Value, ...)",
    inputs: &CELL_INPUTS,
    outputs: &TABLE_OUTPUT,
}];
const STRUCT2TABLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "T = struct2table(S, Name, Value, ...)",
    inputs: &STRUCT_INPUTS,
    outputs: &TABLE_OUTPUT,
}];
const TABLE2ARRAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "A = table2array(T)",
    inputs: &TABLE_INPUT,
    outputs: &ANY_OUTPUT,
}];
const TABLE2CELL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "C = table2cell(T)",
    inputs: &TABLE_INPUT,
    outputs: &ANY_OUTPUT,
}];
const TABLE2STRUCT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "S = table2struct(T)",
        inputs: &TABLE_INPUT,
        outputs: &ANY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = table2struct(T, 'ToScalar', tf)",
        inputs: &TABLE2STRUCT_INPUT,
        outputs: &ANY_OUTPUT,
    },
];
const CONVERSION_ERRORS: [BuiltinErrorDescriptor; 3] = [
    TABLE_ERROR_INVALID_ARGUMENT,
    TABLE_ERROR_INVALID_INDEX,
    TABLE_ERROR_INVALID_VARIABLE,
];
pub const ARRAY2TABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ARRAY2TABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const CELL2TABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CELL2TABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONVERSION_ERRORS,
};
pub const STRUCT2TABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRUCT2TABLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TABLE_ERRORS,
};
pub const TABLE2ARRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TABLE2ARRAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONVERSION_ERRORS,
};
pub const TABLE2CELL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TABLE2CELL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONVERSION_ERRORS,
};
pub const TABLE2STRUCT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TABLE2STRUCT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONVERSION_ERRORS,
};
