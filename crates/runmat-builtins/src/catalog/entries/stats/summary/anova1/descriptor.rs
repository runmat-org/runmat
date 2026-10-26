use crate::*;
pub const ANOVA1_ERROR_INVALID_ARGUMENT:BuiltinErrorDescriptor=BuiltinErrorDescriptor {
    code:"RM.ANOVA1.INVALID_ARGUMENT",identifier:Some("RunMat:anova1:InvalidArgument"),
    when:"Sample/group class, geometry, observed-group count, device residency intent, display or output count is unsupported.",message:"anova1: invalid argument",
};
const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real single or double vector or matrix of observations.",
    },
    BuiltinParamDescriptor {
        name: "group",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description:
            "Numeric/logical, text, cellstr or categorical grouping labels; required for vectors.",
    },
    BuiltinParamDescriptor {
        name: "display",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("'on'"),
        description: "'on' prints the table and renders a boxplot; 'off' suppresses presentation.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper-tail F-test probability.",
    },
    BuiltinParamDescriptor {
        name: "tbl",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "4-by-6 ANOVA cell array with Source, SS, df, MS, F, and Prob>F columns.",
    },
    BuiltinParamDescriptor {
        name: "stats",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Scalar structure with gnames, n, source, means, df and s.",
    },
];
pub const ANOVA1_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "[p,tbl,stats] = anova1(y,group,display)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ANOVA1_ERROR_INVALID_ARGUMENT],
};
