use crate::*;

const TABLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "G",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grouped table summary.",
}];

const GROUPSUMMARY_TABLE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input table.",
    },
    BuiltinParamDescriptor {
        name: "groupvars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping variable name or names.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summary method name or names.",
    },
    BuiltinParamDescriptor {
        name: "datavars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Data variable name or names.",
    },
];
const GROUPSUMMARY_TABLE_BIN_INPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input table.",
    },
    BuiltinParamDescriptor {
        name: "groupvars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping variable name or names.",
    },
    BuiltinParamDescriptor {
        name: "groupbins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping bin specification.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summary method name or names.",
    },
    BuiltinParamDescriptor {
        name: "datavars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Data variable name or names, optionally followed by name-value pairs.",
    },
];
const GROUPSUMMARY_ARRAY_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric array or cell array of numeric arrays.",
    },
    BuiltinParamDescriptor {
        name: "groupvars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping array or cell array of grouping arrays.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summary method name or names.",
    },
];
const GROUPSUMMARY_ARRAY_BIN_INPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric array or cell array of numeric arrays.",
    },
    BuiltinParamDescriptor {
        name: "groupvars",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping array or cell array of grouping arrays.",
    },
    BuiltinParamDescriptor {
        name: "groupbins",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping bin specification.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summary method name or names.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Grouping options such as IncludedEdge and IncludeEmptyGroups.",
    },
];
const GROUPSUMMARY_ARRAY_OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Summary values.",
    },
    BuiltinParamDescriptor {
        name: "BG",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Grouping values.",
    },
    BuiltinParamDescriptor {
        name: "BC",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Group counts.",
    },
];

const GROUPSUMMARY_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "G = groupsummary(T, groupvars, method, datavars)",
        inputs: &GROUPSUMMARY_TABLE_INPUTS,
        outputs: &TABLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "G = groupsummary(T, groupvars, groupbins, method, datavars)",
        inputs: &GROUPSUMMARY_TABLE_BIN_INPUTS,
        outputs: &TABLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = groupsummary(A, groupvars, method)",
        inputs: &GROUPSUMMARY_ARRAY_INPUTS,
        outputs: &GROUPSUMMARY_ARRAY_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[B, BG, BC] = groupsummary(A, groupvars, groupbins, method)",
        inputs: &GROUPSUMMARY_ARRAY_BIN_INPUTS,
        outputs: &GROUPSUMMARY_ARRAY_OUTPUTS,
    },
];

pub const GROUPSUMMARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GROUPSUMMARY_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::super::super::shared::TABLE_ERRORS,
};
