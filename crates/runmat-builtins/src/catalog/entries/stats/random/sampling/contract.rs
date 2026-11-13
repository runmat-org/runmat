//! Shared sampling parameter and result vocabulary. Identity declarations live beside each builtin.
pub use super::bootstrp::contract::*;
pub use super::datasample::contract::*;
pub use super::dividerand::contract::*;
pub use super::randsample::contract::*;
pub use super::unidrnd::contract::*;
use crate::*;

pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLING.INVALID_ARGUMENT",
    identifier: None,
    when: "Inputs, dimensions, replacement flags, or weights are malformed.",
    message: "sampling: invalid argument",
};

pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLING.INTERNAL",
    identifier: None,
    when: "Internal conversion or allocation fails.",
    message: "sampling: internal error",
};

pub const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample.",
}];

pub const OUTPUT_Y_IDX: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Random sample.",
    },
    BuiltinParamDescriptor {
        name: "idx",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based sampled indices along the sampled dimension.",
    },
];

pub const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Discrete uniform random sample.",
}];

pub const OUTPUT_DIVIDERAND: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "trainInd",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "One-based training-set indices.",
    },
    BuiltinParamDescriptor {
        name: "valInd",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based validation-set indices.",
    },
    BuiltinParamDescriptor {
        name: "testInd",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based test-set indices.",
    },
];

pub const OUTPUT_BOOTSTAT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bootstat",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bootstrap statistics, one bootstrap replicate per row.",
}];

pub const OUTPUT_BOOTSTAT_BOOTSAM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bootstat",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bootstrap statistics, one bootstrap replicate per row.",
    },
    BuiltinParamDescriptor {
        name: "bootsam",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "One-based bootstrap sample indices.",
    },
];

pub const PARAM_DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Population to sample from.",
};

pub const PARAM_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Population size or upper discrete uniform bound.",
};

pub const PARAM_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of samples.",
};

pub const PARAM_Q: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of targets to divide.",
};

pub const PARAM_TRAIN_RATIO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "trainRatio",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.7"),
    description: "Training-set allocation ratio.",
};

pub const PARAM_VAL_RATIO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "valRatio",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.15"),
    description: "Validation-set allocation ratio.",
};

pub const PARAM_TEST_RATIO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "testRatio",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.15"),
    description: "Test-set allocation ratio.",
};

pub const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension, replacement, and weights options.",
};

pub const PARAM_SZ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output dimensions.",
};

pub const PARAM_NBOOT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nboot",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of bootstrap samples.",
};

pub const PARAM_BOOTFUN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "bootfun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle applied to each bootstrap sample.",
};

pub const PARAM_BOOT_DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Data arrays and name-value options.",
};

pub const INPUTS_DATA_K: [BuiltinParamDescriptor; 2] = [PARAM_DATA, PARAM_K];

pub const INPUTS_DATA_K_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_DATA, PARAM_K, PARAM_OPTIONS];

pub const INPUTS_N_K: [BuiltinParamDescriptor; 2] = [PARAM_N, PARAM_K];

pub const INPUTS_N_K_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_N, PARAM_K, PARAM_OPTIONS];

pub const INPUTS_N: [BuiltinParamDescriptor; 1] = [PARAM_N];

pub const INPUTS_N_SZ: [BuiltinParamDescriptor; 2] = [PARAM_N, PARAM_SZ];

pub const INPUTS_BOOTSTRP: [BuiltinParamDescriptor; 3] =
    [PARAM_NBOOT, PARAM_BOOTFUN, PARAM_BOOT_DATA];

pub const INPUTS_DIVIDERAND_Q: [BuiltinParamDescriptor; 1] = [PARAM_Q];

pub const INPUTS_DIVIDERAND_RATIOS: [BuiltinParamDescriptor; 4] = [
    PARAM_Q,
    PARAM_TRAIN_RATIO,
    PARAM_VAL_RATIO,
    PARAM_TEST_RATIO,
];
