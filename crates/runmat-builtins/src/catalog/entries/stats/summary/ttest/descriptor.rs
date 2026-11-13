use crate::*;
pub const TTEST_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TTEST.INVALID_ARGUMENT",
    identifier: Some("RunMat:ttest:InvalidArgument"),
    when: "Sample class, paired geometry, option grammar/value, device residency intent, or output count is unsupported.",
    message: "ttest: invalid argument",
};
const X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real single or double sample array.",
};
const REFERENCE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "reference",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Scalar population mean or same-sized paired sample.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name_value_pairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Alpha, Dim, and Tail name/value pairs.",
};
const OUTPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "h",
        ty: BuiltinParamType::LogicalArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reject when p is less than Alpha.",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Student-t tail probability.",
    },
    BuiltinParamDescriptor {
        name: "ci",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Confidence limits for the population mean or paired mean difference.",
    },
    BuiltinParamDescriptor {
        name: "stats",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Scalar structure containing tstat, df, and sd arrays.",
    },
];
pub const TTEST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "[h,p,ci,stats] = ttest(x,Name,Value,...)",
            inputs: &[X, OPTIONS],
            outputs: &OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[h,p,ci,stats] = ttest(x,reference,Name,Value,...)",
            inputs: &[X, REFERENCE, OPTIONS],
            outputs: &OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[TTEST_ERROR_INVALID_ARGUMENT],
};
