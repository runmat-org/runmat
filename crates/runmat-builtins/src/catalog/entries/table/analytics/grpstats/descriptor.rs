use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "stats",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "A numeric summary or grouped summary table.",
}];

const GRPSTATS_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input matrix or table.",
    },
    BuiltinParamDescriptor {
        name: "group",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grouping variables, variable selectors, or empty grouping.",
    },
    BuiltinParamDescriptor {
        name: "whichstats",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("\"mean\""),
        description: "Summary statistic name or names.",
    },
    BuiltinParamDescriptor {
        name: "nameValuePairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Alpha, DataVars, and VarNames options.",
    },
];

const GRPSTATS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "tblstats = grpstats(tbl, groupvars)",
        inputs: &GRPSTATS_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "stats = grpstats(X, group)",
        inputs: &GRPSTATS_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[stats1, ..., statsN] = grpstats(X, group, whichstats)",
        inputs: &GRPSTATS_INPUTS,
        outputs: &OUTPUT,
    },
];

pub const GRPSTATS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GRPSTATS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &super::super::super::shared::TABLE_ERRORS,
};
