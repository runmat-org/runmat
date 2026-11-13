use crate::*;

pub(super) const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Distribution function value.",
}];

pub(super) const INPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Evaluation point.",
};

pub(super) const INPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Probability value.",
};

pub(super) const INPUT_DIST: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Distribution name.",
};

pub(super) const INPUT_MU: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mu",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Mean parameter.",
};

pub(super) const INPUT_SIGMA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sigma",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Standard deviation parameter.",
};

pub(super) const INPUT_NU: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nu",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Degrees of freedom parameter.",
};

pub(super) const INPUT_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of trials.",
};

pub(super) const INPUT_PROB: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "prob",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Success probability parameter.",
};

pub(super) const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Scale parameter.",
};

pub(super) const INPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1"),
    description: "Shape parameter.",
};

pub(super) const INPUT_SHAPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "shape",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Shape parameter.",
};

pub(super) const INPUT_SCALE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "scale",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scale parameter.",
};

pub(super) const INPUT_LAMBDA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "lambda",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Rate or mean parameter.",
};

pub(super) const INPUT_LOWER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Lower endpoint.",
};

pub(super) const INPUT_UPPER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper endpoint.",
};

pub(super) const INPUT_DF1: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "v1",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numerator degrees of freedom.",
};

pub(super) const INPUT_DF2: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "v2",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Denominator degrees of freedom.",
};

pub(super) const INPUT_ALPHA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "a",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "First shape parameter.",
};

pub(super) const INPUT_BETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "b",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second shape parameter.",
};

pub(super) const INPUT_MEAN_REQUIRED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mu",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mean parameter.",
};

pub(super) const INPUTS_X: [BuiltinParamDescriptor; 1] = [INPUT_X];
pub(super) const INPUTS_X_MU: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_MU];
pub(super) const INPUTS_X_MU_SIGMA: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_MU, INPUT_SIGMA];
pub(super) const INPUTS_P: [BuiltinParamDescriptor; 1] = [INPUT_P];
pub(super) const INPUTS_DIST_P: [BuiltinParamDescriptor; 2] = [INPUT_DIST, INPUT_P];
pub(super) const INPUTS_DIST_P_PARAM: [BuiltinParamDescriptor; 3] = [INPUT_DIST, INPUT_P, INPUT_NU];
pub(super) const INPUTS_DIST_P_MU_SIGMA: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_MU, INPUT_SIGMA];
pub(super) const INPUTS_DIST_P_MU: [BuiltinParamDescriptor; 3] =
    [INPUT_DIST, INPUT_P, INPUT_MEAN_REQUIRED];
pub(super) const INPUTS_DIST_P_A_B: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_A, INPUT_B];
pub(super) const INPUTS_DIST_P_SHAPE_SCALE: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_SHAPE, INPUT_SCALE];
pub(super) const INPUTS_DIST_P_N_PROB: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_N, INPUT_PROB];
pub(super) const INPUTS_DIST_P_LAMBDA: [BuiltinParamDescriptor; 3] =
    [INPUT_DIST, INPUT_P, INPUT_LAMBDA];
pub(super) const INPUTS_DIST_P_BOUNDS: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_LOWER, INPUT_UPPER];
pub(super) const INPUTS_DIST_P_DF: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_DF1, INPUT_DF2];
pub(super) const INPUTS_DIST_P_ALPHA_BETA: [BuiltinParamDescriptor; 4] =
    [INPUT_DIST, INPUT_P, INPUT_ALPHA, INPUT_BETA];
pub(super) const INPUTS_P_MU: [BuiltinParamDescriptor; 2] = [INPUT_P, INPUT_MU];
pub(super) const INPUTS_P_MU_SIGMA: [BuiltinParamDescriptor; 3] = [INPUT_P, INPUT_MU, INPUT_SIGMA];
pub(super) const INPUTS_X_NU: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_NU];
pub(super) const INPUT_TAIL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "tail",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper-tail selector (`\"upper\"`).",
};
pub(super) const INPUTS_X_MU_SIGMA_UPPER: [BuiltinParamDescriptor; 4] =
    [INPUT_X, INPUT_MU, INPUT_SIGMA, INPUT_TAIL];
pub(super) const INPUTS_X_UPPER: [BuiltinParamDescriptor; 2] = [INPUT_X, INPUT_TAIL];
pub(super) const INPUTS_X_MU_UPPER: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_MU, INPUT_TAIL];
pub(super) const INPUTS_X_NU_UPPER: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_NU, INPUT_TAIL];
pub(super) const INPUTS_P_NU: [BuiltinParamDescriptor; 2] = [INPUT_P, INPUT_NU];
pub(super) const INPUTS_X_N_P: [BuiltinParamDescriptor; 3] = [INPUT_X, INPUT_N, INPUT_PROB];
pub(super) const INPUTS_X_N_P_UPPER: [BuiltinParamDescriptor; 4] =
    [INPUT_X, INPUT_N, INPUT_PROB, INPUT_TAIL];
pub(super) const INPUTS_P_A_B: [BuiltinParamDescriptor; 3] = [INPUT_P, INPUT_A, INPUT_B];
