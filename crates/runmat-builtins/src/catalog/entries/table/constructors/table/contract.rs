use crate::*;
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar table object retaining named variables and public Properties.",
}];
const INPUT_0: [BuiltinParamDescriptor; 0] = [];
const INPUT_1: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "variables",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Arrays with a common row count; each retains its native class and trailing shape.",
}];
const INPUT_2: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "variables",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Data variables followed by VariableNames or RowNames controls.",
}];
const INPUT_3: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Size",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element exact nonnegative numeric [rows variables] vector.",
    },
    BuiltinParamDescriptor {
        name: "VariableTypes",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One numeric class name for every requested variable.",
    },
    BuiltinParamDescriptor {
        name: "VariableNames",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional explicit variable names.",
    },
    BuiltinParamDescriptor {
        name: "RowNames",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional row labels.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "T = table()",
        inputs: &INPUT_0,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = table(variables...)",
        inputs: &INPUT_1,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = table(variables..., Name=Value)",
        inputs: &INPUT_2,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "T = table(Size=sz, VariableTypes=varTypes, Name=Value...)",
        inputs: &INPUT_3,
        outputs: &OUTPUT,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    TABLE_ERROR_INVALID_ARGUMENT,
    TABLE_ERROR_INVALID_VARIABLE,
    TABLE_ERROR_INVALID_INDEX,
];
pub const TABLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
