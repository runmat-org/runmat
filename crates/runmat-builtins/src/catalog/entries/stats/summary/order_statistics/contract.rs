use crate::*;

const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real observations to order along a dimension.",
};
const INPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Requested probabilities or percentages.",
};
const INPUT_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Dimension or the all-elements option.",
};
const INPUT_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "NaN flag or Method name-value option.",
};
const OUTPUT_Q: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double-precision quantiles or percentiles.",
}];
const OUTPUT_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double-precision average ranks for ties.",
};
const OUTPUT_TIEADJ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tieadj",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description:
        "Tie corrections: one value for ordinary or Ansari ranks, three for Kendall ranks.",
};
const INPUT_KENDALL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "kendall",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Scalar one requests Kendall tie corrections.",
};
const INPUT_ANSARI: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ansari",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "With kendall zero, scalar one requests symmetric Ansari scores.",
};
const INPUTS_X: [BuiltinParamDescriptor; 1] = [INPUT_X];
const INPUTS_X_KENDALL: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_KENDALL];
const INPUTS_X_ANSARI: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_KENDALL, INPUT_ANSARI];
const INPUTS_X_P: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_P];
const INPUTS_X_P_DIM: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_P, INPUT_DIM];
const INPUTS_X_P_OPTIONS: [BuiltinParamDescriptor; 4] =
    [INPUT_X, INPUT_P, INPUT_DIM, INPUT_OPTIONS];
const OUTPUT_R_ONLY: [BuiltinParamDescriptor; 1] = [OUTPUT_R];
const OUTPUT_R_TIEADJ: [BuiltinParamDescriptor; 2] = [OUTPUT_R, OUTPUT_TIEADJ];

const QUANTILE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Q = quantile(X, p)",
        inputs: &INPUTS_X_P,
        outputs: &OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "Q = quantile(X, p, dim)",
        inputs: &INPUTS_X_P_DIM,
        outputs: &OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "Q = quantile(X, p, 'all')",
        inputs: &INPUTS_X_P_DIM,
        outputs: &OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "Q = quantile(X, p, dim, nanflag, 'Method', method)",
        inputs: &INPUTS_X_P_OPTIONS,
        outputs: &OUTPUT_Q,
    },
];
const PRCTILE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Y = prctile(X, p)",
        inputs: &INPUTS_X_P,
        outputs: &OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "Y = prctile(X, p, dim)",
        inputs: &INPUTS_X_P_DIM,
        outputs: &OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "Y = prctile(X, p, 'all')",
        inputs: &INPUTS_X_P_DIM,
        outputs: &OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "Y = prctile(X, p, dim, nanflag, 'Method', method)",
        inputs: &INPUTS_X_P_OPTIONS,
        outputs: &OUTPUT_Q,
    },
];
const TIEDRANK_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "R = tiedrank(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_R_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "[R, tieadj] = tiedrank(X)",
        inputs: &INPUTS_X,
        outputs: &OUTPUT_R_TIEADJ,
    },
    BuiltinSignatureDescriptor {
        label: "[R, tieadj] = tiedrank(X, 1)",
        inputs: &INPUTS_X_KENDALL,
        outputs: &OUTPUT_R_TIEADJ,
    },
    BuiltinSignatureDescriptor {
        label: "[R, tieadj] = tiedrank(X, 0, 1)",
        inputs: &INPUTS_X_ANSARI,
        outputs: &OUTPUT_R_TIEADJ,
    },
];

const fn error(
    code: &'static str,
    identifier: &'static str,
    when: &'static str,
    message: &'static str,
) -> BuiltinErrorDescriptor {
    BuiltinErrorDescriptor {
        code,
        identifier: Some(identifier),
        when,
        message,
    }
}
const QUANTILE_ERRORS: [BuiltinErrorDescriptor; 2] = [
    error(
        "RM.quantile.INVALID_ARGUMENT",
        "RunMat:quantile:InvalidArgument",
        "Input, probability, dimension, method, or NaN option is malformed.",
        "quantile: invalid argument",
    ),
    error(
        "RM.quantile.INTERNAL",
        "RunMat:quantile:Internal",
        "Gather, tensor conversion, or output allocation fails.",
        "quantile: internal error",
    ),
];
const PRCTILE_ERRORS: [BuiltinErrorDescriptor; 2] = [
    error(
        "RM.prctile.INVALID_ARGUMENT",
        "RunMat:prctile:InvalidArgument",
        "Input, percentage, dimension, method, or NaN option is malformed.",
        "prctile: invalid argument",
    ),
    error(
        "RM.prctile.INTERNAL",
        "RunMat:prctile:Internal",
        "Gather, tensor conversion, or output allocation fails.",
        "prctile: internal error",
    ),
];
const TIEDRANK_ERRORS: [BuiltinErrorDescriptor; 2] = [
    error(
        "RM.tiedrank.INVALID_ARGUMENT",
        "RunMat:tiedrank:InvalidArgument",
        "Input is not a supported real numeric array, or flags are malformed.",
        "tiedrank: invalid argument",
    ),
    error(
        "RM.tiedrank.INTERNAL",
        "RunMat:tiedrank:Internal",
        "Gather, tensor conversion, or output allocation fails.",
        "tiedrank: internal error",
    ),
];

pub const QUANTILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &QUANTILE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &QUANTILE_ERRORS,
};
pub const PRCTILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PRCTILE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PRCTILE_ERRORS,
};
pub const TIEDRANK_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TIEDRANK_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TIEDRANK_ERRORS,
};
