//! Sampling utilities for Statistics and Machine Learning Toolbox compatibility.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, IntValue, LogicalArray, NumericStorage, ResolveContext, StringArray,
    Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const MAX_DIVIDERAND_Q: usize = 10_000_000;

const DIVIDERAND_RESIDENT_ARGUMENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "dividerand-resident-argument",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "resident dividerand arguments are gathered as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DividerandResidentArgumentExtension"),
    };
const DIVIDERAND_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [DIVIDERAND_RESIDENT_ARGUMENT_EXTENSION];
const DIVIDERAND_INTEGER_Q_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Q",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Q is a nonnegative scalar target count and typed integer scalars are decoded from authoritative storage.",
    }];
pub const DIVIDERAND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[trainInd,valInd,testInd] = dividerand(integer_Q,___)",
        inputs: &DIVIDERAND_INTEGER_Q_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Q must fit usize and RunMat's bounded allocation policy; all returned index vectors are double row vectors.",
    }];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLING.INVALID_ARGUMENT",
    identifier: None,
    when: "Inputs, dimensions, replacement flags, or weights are malformed.",
    message: "sampling: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAMPLING.INTERNAL",
    identifier: None,
    when: "Internal conversion or allocation fails.",
    message: "sampling: internal error",
};

macro_rules! sampling_descriptor {
    ($name:literal, $signatures:expr, $output_mode:expr) => {
        const ERRORS: [BuiltinErrorDescriptor; 2] = [
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
                identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
                when: ERROR_INVALID_ARGUMENT.when,
                message: ERROR_INVALID_ARGUMENT.message,
            },
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INTERNAL"),
                identifier: Some(concat!("RunMat:", $name, ":Internal")),
                when: ERROR_INTERNAL.when,
                message: ERROR_INTERNAL.message,
            },
        ];

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: $output_mode,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

const OUTPUT_Y: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample.",
}];

const OUTPUT_Y_IDX: [BuiltinParamDescriptor; 2] = [
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

const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Discrete uniform random sample.",
}];

const OUTPUT_DIVIDERAND: [BuiltinParamDescriptor; 3] = [
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

const OUTPUT_BOOTSTAT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bootstat",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bootstrap statistics, one bootstrap replicate per row.",
}];

const OUTPUT_BOOTSTAT_BOOTSAM: [BuiltinParamDescriptor; 2] = [
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

const PARAM_DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Population to sample from.",
};

const PARAM_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Population size or upper discrete uniform bound.",
};

const PARAM_K: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "k",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of samples.",
};

const PARAM_Q: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of targets to divide.",
};

const PARAM_TRAIN_RATIO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "trainRatio",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.7"),
    description: "Training-set allocation ratio.",
};

const PARAM_VAL_RATIO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "valRatio",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.15"),
    description: "Validation-set allocation ratio.",
};

const PARAM_TEST_RATIO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "testRatio",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.15"),
    description: "Test-set allocation ratio.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension, replacement, and weights options.",
};

const PARAM_SZ: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output dimensions.",
};

const PARAM_NBOOT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nboot",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of bootstrap samples.",
};

const PARAM_BOOTFUN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "bootfun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle applied to each bootstrap sample.",
};

const PARAM_BOOT_DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Data arrays and name-value options.",
};

const INPUTS_DATA_K: [BuiltinParamDescriptor; 2] = [PARAM_DATA, PARAM_K];
const INPUTS_DATA_K_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_DATA, PARAM_K, PARAM_OPTIONS];
const INPUTS_N_K: [BuiltinParamDescriptor; 2] = [PARAM_N, PARAM_K];
const INPUTS_N_K_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_N, PARAM_K, PARAM_OPTIONS];
const INPUTS_N: [BuiltinParamDescriptor; 1] = [PARAM_N];
const INPUTS_N_SZ: [BuiltinParamDescriptor; 2] = [PARAM_N, PARAM_SZ];
const INPUTS_BOOTSTRP: [BuiltinParamDescriptor; 3] = [PARAM_NBOOT, PARAM_BOOTFUN, PARAM_BOOT_DATA];
const INPUTS_DIVIDERAND_Q: [BuiltinParamDescriptor; 1] = [PARAM_Q];
const INPUTS_DIVIDERAND_RATIOS: [BuiltinParamDescriptor; 4] = [
    PARAM_Q,
    PARAM_TRAIN_RATIO,
    PARAM_VAL_RATIO,
    PARAM_TEST_RATIO,
];

const DATASAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "y = datasample(data, k)",
        inputs: &INPUTS_DATA_K,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = datasample(data, k, dim)",
        inputs: &INPUTS_DATA_K_OPTIONS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = datasample(___, Name, Value)",
        inputs: &INPUTS_DATA_K_OPTIONS,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "[y, idx] = datasample(___)",
        inputs: &INPUTS_DATA_K_OPTIONS,
        outputs: &OUTPUT_Y_IDX,
    },
];

pub const DATASAMPLE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with typed-integer population data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleIntegerDataExtension"),
    };
pub const DATASAMPLE_INTEGER_K_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "datasample-integer-k",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "datasample with a typed-integer sample count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DatasampleIntegerKExtension"),
};
pub const DATASAMPLE_INTEGER_DIM_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-integer-dim",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with a typed-integer dimension is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleIntegerDimExtension"),
    };
pub const DATASAMPLE_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-integer-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with typed-integer weights is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleIntegerWeightsExtension"),
    };
pub const DATASAMPLE_LOGICAL_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-logical-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with logical weights is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleLogicalWeightsExtension"),
    };
pub const DATASAMPLE_NUMERIC_REPLACE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-numeric-replace",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with a numeric Replace value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleNumericReplaceExtension"),
    };
pub const DATASAMPLE_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "datasample-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "datasample with resident accelerator input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DatasampleResidentInputExtension"),
    };
pub const DATASAMPLE_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    DATASAMPLE_INTEGER_DATA_EXTENSION,
    DATASAMPLE_INTEGER_K_EXTENSION,
    DATASAMPLE_INTEGER_DIM_EXTENSION,
    DATASAMPLE_INTEGER_WEIGHTS_EXTENSION,
    DATASAMPLE_LOGICAL_WEIGHTS_EXTENSION,
    DATASAMPLE_NUMERIC_REPLACE_EXTENSION,
    DATASAMPLE_RESIDENT_INPUT_EXTENSION,
];

const DATASAMPLE_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "data",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are sampled by exact storage indexing; the documented ordinary numeric population classes are single and double.",
    }];
const DATASAMPLE_INTEGER_K_INPUTS: [BuiltinIntegerInputCapability; 1] = [
    BuiltinIntegerInputCapability {
        name: "k",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes:
            "Typed k is independently gated and read exactly as a positive structural sample count.",
    },
];
const DATASAMPLE_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed dim is independently gated and read exactly as a positive one-based structural dimension.",
    }];
const DATASAMPLE_INTEGER_WEIGHTS_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Weights",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed nonnegative weights are independently gated and converted only at the floating probability boundary.",
    }];
pub const DATASAMPLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = datasample(integer_data,k,___)",
        inputs: &DATASAMPLE_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Sampling preserves the exact selected integer values and native integer class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = datasample(data,integer_k,___)",
        inputs: &DATASAMPLE_INTEGER_K_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed k controls output extent and never determines the sampled value class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = datasample(data,k,integer_dim,___)",
        inputs: &DATASAMPLE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed dim selects an axis and never determines the sampled value class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = datasample(data,k,Weights=integer_weights,___)",
        inputs: &DATASAMPLE_INTEGER_WEIGHTS_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes:
            "Weights affect selection probabilities only; sampled data retains its ordinary class.",
    },
];

const RANDSAMPLE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "y = randsample(n, k)",
        inputs: &INPUTS_N_K,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = randsample(population, k)",
        inputs: &INPUTS_DATA_K,
        outputs: &OUTPUT_Y,
    },
    BuiltinSignatureDescriptor {
        label: "y = randsample(___, replacement, w)",
        inputs: &INPUTS_N_K_OPTIONS,
        outputs: &OUTPUT_Y,
    },
];

const RANDSAMPLE_INTEGER_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "randsample-integer-range",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "randsample with a typed-integer range limit is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandsampleIntegerRangeExtension"),
};
const RANDSAMPLE_INTEGER_POPULATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "randsample-integer-population",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "randsample with a typed-integer population is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RandsampleIntegerPopulationExtension"),
    };
const RANDSAMPLE_INTEGER_K_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "randsample-integer-count",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "randsample with a typed-integer sample count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandsampleIntegerCountExtension"),
};
const RANDSAMPLE_INTEGER_REPLACEMENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "randsample-integer-replacement",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "randsample with a typed-integer replacement flag is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RandsampleIntegerReplacementExtension"),
    };
const RANDSAMPLE_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "randsample-integer-weights",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "randsample with typed-integer sampling weights is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RandsampleIntegerWeightsExtension"),
    };
pub const RANDSAMPLE_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    RANDSAMPLE_INTEGER_N_EXTENSION,
    RANDSAMPLE_INTEGER_POPULATION_EXTENSION,
    RANDSAMPLE_INTEGER_K_EXTENSION,
    RANDSAMPLE_INTEGER_REPLACEMENT_EXTENSION,
    RANDSAMPLE_INTEGER_WEIGHTS_EXTENSION,
];
const RANDSAMPLE_INTEGER_RANGE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents single/double n; RunMat typed range limits are exact structural controls and return double sampled indices.",
    }];
const RANDSAMPLE_INTEGER_POPULATION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "population",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target lists single/double/logical/text/categorical populations; RunMat preserves exact typed-integer population values, orientation, class, and supported residency.",
    }];
const RANDSAMPLE_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "k/replacement/w",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed count, replacement, and weight roles are independently gated; only weights cross a checked binary64 probability boundary.",
    }];
pub const RANDSAMPLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = randsample(integer_n,k,___)",
        inputs: &RANDSAMPLE_INTEGER_RANGE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The range form samples 1:n and emits a double column/scalar result.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = randsample(integer_population,k,___)",
        inputs: &RANDSAMPLE_INTEGER_POPULATION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Sampling copies authoritative values by index and restores resident population output through the exact owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = randsample(...,integer_k,integer_replacement,integer_w)",
        inputs: &RANDSAMPLE_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Counts and flags remain structural; weights require exact representability before floating probability normalization.",
    },
];

const UNIDRND_INTEGER_LIMIT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "unidrnd-integer-limit",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "unidrnd with a typed-integer upper limit is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UnidrndIntegerLimitExtension"),
};
const UNIDRND_INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "unidrnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "unidrnd with typed-integer size arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UnidrndIntegerSizeExtension"),
};
const UNIDRND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    UNIDRND_INTEGER_LIMIT_EXTENSION,
    UNIDRND_INTEGER_SIZE_EXTENSION,
];
const UNIDRND_INTEGER_LIMIT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double upper limits; RunMat mode accepts typed integers only when the sampling boundary can represent them exactly as binary64.",
    }];
const UNIDRND_INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sz, sz1, ...",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double size controls; RunMat mode decodes typed integer extents exactly as structural values.",
    }];
pub const UNIDRND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "r = unidrnd(integer_n, ___)",
        inputs: &UNIDRND_INTEGER_LIMIT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The upper limit is extension-gated before gather and must cross the binary64 random-sampling boundary without rounding.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = unidrnd(n, integer_sz)",
        inputs: &UNIDRND_INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed extents are extension-gated before provider access and parsed exactly as output shape.",
    },
];

const UNIDRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = unidrnd(n)",
        inputs: &INPUTS_N,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = unidrnd(n, sz)",
        inputs: &INPUTS_N_SZ,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = unidrnd(n, sz1, sz2, ...)",
        inputs: &INPUTS_N_SZ,
        outputs: &OUTPUT_R,
    },
];

const BOOTSTRP_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "bootstat = bootstrp(nboot, bootfun, d)",
        inputs: &INPUTS_BOOTSTRP,
        outputs: &OUTPUT_BOOTSTAT,
    },
    BuiltinSignatureDescriptor {
        label: "bootstat = bootstrp(nboot, bootfun, d1, ..., dN)",
        inputs: &INPUTS_BOOTSTRP,
        outputs: &OUTPUT_BOOTSTAT,
    },
    BuiltinSignatureDescriptor {
        label: "[bootstat, bootsam] = bootstrp(___)",
        inputs: &INPUTS_BOOTSTRP,
        outputs: &OUTPUT_BOOTSTAT_BOOTSAM,
    },
];

const BOOTSTRP_INTEGER_NBOOT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bootstrp-integer-nboot",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bootstrp with an integer nboot control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BootstrpIntegerNbootExtension"),
};

const BOOTSTRP_LOGICAL_NBOOT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bootstrp-logical-nboot",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bootstrp with a logical nboot control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BootstrpLogicalNbootExtension"),
};

const BOOTSTRP_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bootstrp-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bootstrp with integer sample data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BootstrpIntegerDataExtension"),
};

const BOOTSTRP_INTEGER_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bootstrp-integer-weights",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bootstrp with integer observation weights is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BootstrpIntegerWeightsExtension"),
};

const BOOTSTRP_LOGICAL_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bootstrp-logical-weights",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bootstrp with logical observation weights is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BootstrpLogicalWeightsExtension"),
};

const BOOTSTRP_TEXT_CALLABLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bootstrp-text-callable",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bootstrp with a text callback name is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BootstrpTextCallableExtension"),
};

const BOOTSTRP_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bootstrp-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bootstrp with gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BootstrpGpuInputExtension"),
};

const BOOTSTRP_EXTENSIONS: [BuiltinExtensionDescriptor; 7] = [
    BOOTSTRP_INTEGER_NBOOT_EXTENSION,
    BOOTSTRP_LOGICAL_NBOOT_EXTENSION,
    BOOTSTRP_INTEGER_DATA_EXTENSION,
    BOOTSTRP_INTEGER_WEIGHTS_EXTENSION,
    BOOTSTRP_LOGICAL_WEIGHTS_EXTENSION,
    BOOTSTRP_TEXT_CALLABLE_EXTENSION,
    BOOTSTRP_GPU_INPUT_EXTENSION,
];

const BOOTSTRP_INTEGER_NBOOT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "nboot",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat accepts every integer class as an exact positive scalar bootstrap count; the public control domain is single or double.",
    }];

const BOOTSTRP_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "d1,...,dN",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat samples every integer data class by exact row or vector indexing; public nongrouping numeric data is single or double.",
    }];

const BOOTSTRP_INTEGER_WEIGHT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Weights",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat accepts every nonnegative integer weight class and converts it once at the floating multinomial-probability boundary; the public weight domain is single or double.",
    }];

pub const BOOTSTRP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "bootstat = bootstrp(nboot,bootfun,d1,...,dN) with integer nboot",
        inputs: &BOOTSTRP_INTEGER_NBOOT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only exact structural count; callback output class determines bootstat and bootsam remains double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "bootstat = bootstrp(nboot,bootfun,d1,...,dN) with integer data",
        inputs: &BOOTSTRP_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only resampling preserves authoritative integer storage into each callback; homogeneous numeric or logical callback output determines bootstat class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "bootstat = bootstrp(___,Weights=w) with integer weights",
        inputs: &BOOTSTRP_INTEGER_WEIGHT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only integer weights cross one deliberate binary64 probability boundary after exact nonnegative validation; sampled data and callback output retain their own classes.",
    },
];

const DIVIDERAND_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[trainInd, valInd, testInd] = dividerand(Q)",
        inputs: &INPUTS_DIVIDERAND_Q,
        outputs: &OUTPUT_DIVIDERAND,
    },
    BuiltinSignatureDescriptor {
        label: "[trainInd, valInd, testInd] = dividerand(Q, trainRatio, valRatio, testRatio)",
        inputs: &INPUTS_DIVIDERAND_RATIOS,
        outputs: &OUTPUT_DIVIDERAND,
    },
];

fn sampling_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn numeric_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn sampling_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

async fn gathered(value: Value, name: &str) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))
}

fn parse_positive_usize(name: &str, value: &Value, label: &str) -> BuiltinResult<usize> {
    if let Some(value) = scalar_integer_value(value) {
        return value
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                sampling_error(name, format!("{name}: {label} must be a positive integer"))
            });
    }
    let raw = match value {
        Value::Num(v) => *v,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::Bool(v) => {
            if *v {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(sampling_error(
                name,
                format!("{name}: {label} must be a positive integer, got {other:?}"),
            ));
        }
    };
    if !raw.is_finite()
        || raw < 1.0
        || raw.fract() != 0.0
        || raw > usize::MAX as f64
        || (usize::BITS == 64 && raw == usize::MAX as f64)
    {
        return Err(sampling_error(
            name,
            format!("{name}: {label} must be a positive integer"),
        ));
    }
    Ok(raw as usize)
}

fn scalar_integer_value(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor.integer_storage().map(|storage| {
                storage
                    .value_at(0)
                    .expect("scalar integer tensor has one storage value")
            })
        }
        _ => None,
    }
}

fn parse_bool(name: &str, value: &Value, label: &str) -> BuiltinResult<bool> {
    if let Some(value) = scalar_integer_value(value) {
        return match value.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(sampling_error(
                name,
                format!("{name}: {label} must be logical true or false, got {value:?}"),
            )),
        };
    }
    match value {
        Value::Bool(v) => Ok(*v),
        Value::Num(v) if *v == 0.0 || *v == 1.0 => Ok(*v != 0.0),
        other => Err(sampling_error(
            name,
            format!("{name}: {label} must be logical true or false, got {other:?}"),
        )),
    }
}

fn first_non_singleton(shape: &[usize]) -> usize {
    shape.iter().position(|dim| *dim > 1).unwrap_or(0)
}

fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape = vec![1, 1];
    } else if shape.len() == 1 {
        shape.push(1);
    }
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn parse_weights(name: &str, value: Value, expected: usize) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(name, value)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))?;
    let weights = tensor::tensor_into_values_f64(tensor);
    if weights.len() != expected {
        return Err(sampling_error(
            name,
            format!("{name}: weights length must match the sampled dimension"),
        ));
    }
    if weights
        .iter()
        .any(|weight| weight.is_nan() || *weight < 0.0)
    {
        return Err(sampling_error(
            name,
            format!("{name}: weights must be nonnegative and cannot contain NaN"),
        ));
    }
    if weights.iter().sum::<f64>() <= 0.0 {
        return Err(sampling_error(
            name,
            format!("{name}: weights must contain at least one positive value"),
        ));
    }
    Ok(weights)
}

fn sample_indices(
    name: &str,
    population_len: usize,
    k: usize,
    replacement: bool,
    weights: Option<&[f64]>,
) -> BuiltinResult<Vec<usize>> {
    if population_len == 0 {
        return Err(sampling_error(name, format!("{name}: population is empty")));
    }
    if !replacement && k > population_len {
        return Err(sampling_error(
            name,
            format!("{name}: k cannot exceed population size without replacement"),
        ));
    }
    match (replacement, weights) {
        (true, Some(weights)) => weighted_with_replacement(name, k, weights),
        (false, Some(weights)) => weighted_without_replacement(name, k, weights),
        (true, None) => {
            let uniforms = random::generate_uniform(k, name)?;
            Ok(uniforms
                .into_iter()
                .map(|u| ((u * population_len as f64).floor() as usize).min(population_len - 1))
                .collect())
        }
        (false, None) => unweighted_without_replacement(name, population_len, k),
    }
}

fn unweighted_without_replacement(
    name: &str,
    population_len: usize,
    k: usize,
) -> BuiltinResult<Vec<usize>> {
    let uniforms = random::generate_uniform(k, name)?;
    let mut pool = (0..population_len).collect::<Vec<_>>();
    let mut out = Vec::with_capacity(k);
    for (draw, u) in uniforms.into_iter().enumerate() {
        let span = population_len - draw;
        let offset = ((u * span as f64).floor() as usize).min(span - 1);
        out.push(pool.swap_remove(offset));
    }
    Ok(out)
}

fn weighted_with_replacement(name: &str, k: usize, weights: &[f64]) -> BuiltinResult<Vec<usize>> {
    let total = weights.iter().sum::<f64>();
    let uniforms = random::generate_uniform(k, name)?;
    Ok(uniforms
        .into_iter()
        .map(|u| choose_weighted(weights, total, u))
        .collect())
}

fn weighted_without_replacement(
    name: &str,
    k: usize,
    weights: &[f64],
) -> BuiltinResult<Vec<usize>> {
    let uniforms = random::generate_uniform(k, name)?;
    let mut weights = weights.to_vec();
    let mut out = Vec::with_capacity(k);
    for u in uniforms {
        let total = weights.iter().sum::<f64>();
        if total <= 0.0 {
            return Err(sampling_error(
                name,
                format!("{name}: not enough positive weights to sample without replacement"),
            ));
        }
        let idx = choose_weighted(&weights, total, u);
        weights[idx] = 0.0;
        out.push(idx);
    }
    Ok(out)
}

fn choose_weighted(weights: &[f64], total: f64, u: f64) -> usize {
    let mut threshold = u * total;
    for (idx, weight) in weights.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        if threshold < *weight {
            return idx;
        }
        threshold -= *weight;
    }
    weights
        .iter()
        .rposition(|weight| *weight > 0.0)
        .unwrap_or(0)
}

fn indices_value(indices: &[usize]) -> BuiltinResult<Value> {
    Tensor::new(
        indices.iter().map(|idx| (idx + 1) as f64).collect(),
        vec![indices.len(), 1],
    )
    .map(tensor::tensor_into_value)
    .map_err(|err| sampling_error("datasample", format!("datasample: {err}")))
}

fn sample_tensor_axis(
    tensor: Tensor,
    shape: &[usize],
    axis: usize,
    indices: &[usize],
    name: &str,
) -> BuiltinResult<Value> {
    let mut out_shape = shape.to_vec();
    out_shape[axis] = indices.len();
    let out_len = tensor::element_count(&out_shape);
    let mut source_indices = vec![0; out_len];
    let pre: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let post: usize = shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            for (dst_axis, src_axis) in indices.iter().enumerate() {
                if *src_axis >= axis_len {
                    return Err(sampling_error(
                        name,
                        format!("{name}: sample index out of range"),
                    ));
                }
                let src = prefix + src_axis * pre + suffix * pre * axis_len;
                let dst = prefix + dst_axis * pre + suffix * pre * indices.len();
                source_indices[dst] = src;
            }
        }
    }
    let storage = tensor
        .into_numeric_storage()
        .and_then(|storage| storage.gather(&source_indices))
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))?;
    Tensor::from_numeric_storage(storage, out_shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))
}

fn sample_logical_axis(
    data: &[u8],
    shape: &[usize],
    axis: usize,
    indices: &[usize],
    name: &str,
) -> BuiltinResult<Value> {
    let mut out_shape = shape.to_vec();
    out_shape[axis] = indices.len();
    let out_len = tensor::element_count(&out_shape);
    let mut out = vec![0u8; out_len];
    let pre: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let post: usize = shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            for (dst_axis, src_axis) in indices.iter().enumerate() {
                let src = prefix + src_axis * pre + suffix * pre * axis_len;
                let dst = prefix + dst_axis * pre + suffix * pre * indices.len();
                out[dst] = data[src];
            }
        }
    }
    LogicalArray::new(out, out_shape)
        .map(Value::LogicalArray)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))
}

fn sample_string_axis(
    array: &StringArray,
    axis: usize,
    indices: &[usize],
    _name: &str,
) -> BuiltinResult<Value> {
    let shape = normalize_shape(array.shape.clone());
    let mut out_shape = shape.clone();
    out_shape[axis] = indices.len();
    let out_len = tensor::element_count(&out_shape);
    let mut out = vec![String::new(); out_len];
    let pre: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let post: usize = shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            for (dst_axis, src_axis) in indices.iter().enumerate() {
                let src = prefix + src_axis * pre + suffix * pre * axis_len;
                let dst = prefix + dst_axis * pre + suffix * pre * indices.len();
                out[dst] = array.data[src].clone();
            }
        }
    }
    let rows = *out_shape.first().unwrap_or(&1);
    let cols = *out_shape.get(1).unwrap_or(&1);
    Ok(Value::StringArray(StringArray {
        data: out,
        shape: out_shape,
        rows,
        cols,
    }))
}

fn sample_char_axis(
    array: &CharArray,
    axis: usize,
    indices: &[usize],
    name: &str,
) -> BuiltinResult<Value> {
    let shape = vec![array.rows, array.cols];
    let mut out_shape = shape.clone();
    out_shape[axis] = indices.len();
    let out_len = tensor::element_count(&out_shape);
    let mut out = vec![' '; out_len];
    let pre: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let post: usize = shape[axis + 1..].iter().product();
    for prefix in 0..pre {
        for suffix in 0..post {
            for (dst_axis, src_axis) in indices.iter().enumerate() {
                let src = prefix + src_axis * pre + suffix * pre * axis_len;
                let dst = prefix + dst_axis * pre + suffix * pre * indices.len();
                out[dst] = array.data[src];
            }
        }
    }
    CharArray::new(out, out_shape[0], out_shape[1])
        .map(Value::CharArray)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))
}

fn sample_cell_axis(
    array: &CellArray,
    axis: usize,
    indices: &[usize],
    name: &str,
) -> BuiltinResult<Value> {
    let shape = normalize_shape(array.shape.clone());
    let mut out_shape = shape.clone();
    out_shape[axis] = indices.len();
    let out_len = tensor::element_count(&out_shape);
    let mut out = vec![Value::Num(0.0); out_len];
    let source_strides = row_major_strides(&shape);
    let output_strides = row_major_strides(&out_shape);
    for output_linear in 0..out_len {
        let mut rem = output_linear;
        let mut coords = vec![0usize; out_shape.len()];
        for (dim, stride) in output_strides.iter().enumerate() {
            coords[dim] = rem / *stride;
            rem %= *stride;
        }
        let source_axis = indices[coords[axis]];
        if source_axis >= shape[axis] {
            return Err(sampling_error(
                name,
                format!("{name}: sample index out of range"),
            ));
        }
        coords[axis] = source_axis;
        let source_linear = coords
            .iter()
            .zip(source_strides.iter())
            .map(|(coord, stride)| coord * stride)
            .sum::<usize>();
        out[output_linear] = array.data[source_linear].clone();
    }
    CellArray::new_with_shape(out, out_shape)
        .map(Value::Cell)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    let mut acc = 1usize;
    for idx in (0..shape.len()).rev() {
        strides[idx] = acc;
        acc = acc.saturating_mul(shape[idx]);
    }
    strides
}

fn sample_value_axis(
    data: Value,
    axis: usize,
    indices: &[usize],
    name: &str,
) -> BuiltinResult<Value> {
    match data {
        Value::Tensor(t) => {
            let shape = normalize_shape(t.shape.clone());
            sample_tensor_axis(t, &shape, axis, indices, name)
        }
        Value::Num(value) => {
            let tensor = Tensor::new(vec![value], vec![1, 1])
                .map_err(|err| sampling_error(name, format!("{name}: {err}")))?;
            sample_tensor_axis(tensor, &[1, 1], axis, indices, name)
        }
        Value::Int(value) => {
            let tensor = Tensor::new_integer(
                runmat_builtins::IntegerStorage::from_scalar(value),
                vec![1, 1],
            )
            .map_err(|err| sampling_error(name, format!("{name}: {err}")))?;
            sample_tensor_axis(tensor, &[1, 1], axis, indices, name)
        }
        Value::Bool(value) => {
            let byte = if value { 1 } else { 0 };
            sample_logical_axis(&[byte], &[1, 1], axis, indices, name)
        }
        Value::LogicalArray(array) => sample_logical_axis(
            &array.data,
            &normalize_shape(array.shape),
            axis,
            indices,
            name,
        ),
        Value::String(value) => sample_string_axis(
            &StringArray {
                data: vec![value],
                shape: vec![1, 1],
                rows: 1,
                cols: 1,
            },
            axis,
            indices,
            name,
        ),
        Value::StringArray(array) => sample_string_axis(&array, axis, indices, name),
        Value::CharArray(array) => sample_char_axis(&array, axis, indices, name),
        Value::Cell(array) => sample_cell_axis(&array, axis, indices, name),
        other => Err(sampling_error(
            name,
            format!("{name}: unsupported population type {other:?}"),
        )),
    }
}

fn shape_of_sampled_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => Ok(normalize_shape(t.shape.clone())),
        Value::LogicalArray(a) => Ok(normalize_shape(a.shape.clone())),
        Value::StringArray(a) => Ok(normalize_shape(a.shape.clone())),
        Value::CharArray(a) => Ok(vec![a.rows, a.cols]),
        Value::Cell(a) => Ok(normalize_shape(a.shape.clone())),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => Ok(vec![1, 1]),
        other => Err(sampling_error(
            "datasample",
            format!("datasample: unsupported population type {other:?}"),
        )),
    }
}

#[derive(Clone)]
struct DatasampleArgs {
    data: Value,
    k: usize,
    dim: Option<usize>,
    replacement: bool,
    weights: Option<Vec<f64>>,
}

fn datasample_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(value_tensor) if value_tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn datasample_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn datasample_numeric_replace(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_))
        || matches!(value, Value::GpuTensor(handle) if !runmat_accelerate_api::handle_is_logical(handle))
}

fn ensure_datasample_extension(extension: &BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, "datasample")
}

fn ensure_datasample_extensions(data: &Value, rest: &[Value]) -> BuiltinResult<()> {
    if datasample_integer_value(data) {
        ensure_datasample_extension(&DATASAMPLE_INTEGER_DATA_EXTENSION)?;
    }
    if matches!(data, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        ensure_datasample_extension(&DATASAMPLE_RESIDENT_INPUT_EXTENSION)?;
    }
    if let Some(k) = rest.first() {
        if datasample_integer_value(k) {
            ensure_datasample_extension(&DATASAMPLE_INTEGER_K_EXTENSION)?;
        }
    }
    let mut index = 1usize;
    while index < rest.len() {
        if let Some(keyword) = keyword_of(&rest[index]) {
            let Some(value) = rest.get(index + 1) else {
                break;
            };
            match keyword.as_str() {
                "replace" if datasample_numeric_replace(value) => {
                    ensure_datasample_extension(&DATASAMPLE_NUMERIC_REPLACE_EXTENSION)?;
                }
                "weights" if datasample_integer_value(value) => {
                    ensure_datasample_extension(&DATASAMPLE_INTEGER_WEIGHTS_EXTENSION)?;
                }
                "weights" if datasample_logical_value(value) => {
                    ensure_datasample_extension(&DATASAMPLE_LOGICAL_WEIGHTS_EXTENSION)?;
                }
                _ => {}
            }
            index += 2;
        } else {
            if datasample_integer_value(&rest[index]) {
                ensure_datasample_extension(&DATASAMPLE_INTEGER_DIM_EXTENSION)?;
            }
            index += 1;
        }
    }
    Ok(())
}

async fn parse_datasample_args(data: Value, rest: Vec<Value>) -> BuiltinResult<DatasampleArgs> {
    if rest.is_empty() {
        return Err(sampling_error("datasample", "datasample: k is required"));
    }
    let data = gathered(data, "datasample").await?;
    let k = parse_positive_usize("datasample", &rest[0], "k")?;
    let mut dim = None;
    let mut replacement = true;
    let mut weight_value = None;
    let mut idx = 1usize;
    while idx < rest.len() {
        if let Some(keyword) = keyword_of(&rest[idx]) {
            match keyword.as_str() {
                "replace" => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(sampling_error(
                            "datasample",
                            "datasample: Replace requires a value",
                        ));
                    };
                    replacement = parse_bool("datasample", value, "Replace")?;
                    idx += 2;
                    continue;
                }
                "weights" => {
                    let Some(value) = rest.get(idx + 1) else {
                        return Err(sampling_error(
                            "datasample",
                            "datasample: Weights requires a value",
                        ));
                    };
                    weight_value = Some(gathered(value.clone(), "datasample").await?);
                    idx += 2;
                    continue;
                }
                other => {
                    return Err(sampling_error(
                        "datasample",
                        format!("datasample: unsupported option '{other}'"),
                    ));
                }
            }
        }
        if dim.is_some() {
            return Err(sampling_error(
                "datasample",
                "datasample: dimension can only be specified once",
            ));
        }
        dim = Some(parse_positive_usize("datasample", &rest[idx], "dim")?);
        idx += 1;
    }
    let shape = shape_of_sampled_value(&data)?;
    let axis = dim
        .map(|value| value - 1)
        .unwrap_or_else(|| first_non_singleton(&shape));
    if axis >= shape.len() {
        return Err(sampling_error(
            "datasample",
            "datasample: dimension exceeds input rank",
        ));
    }
    let weights = match weight_value {
        Some(value) => Some(parse_weights("datasample", value, shape[axis])?),
        None => None,
    };
    Ok(DatasampleArgs {
        data,
        k,
        dim: Some(axis),
        replacement,
        weights,
    })
}

fn datasample_compute(args: DatasampleArgs) -> BuiltinResult<(Value, Value)> {
    let shape = shape_of_sampled_value(&args.data)?;
    let axis = args.dim.unwrap_or_else(|| first_non_singleton(&shape));
    let indices = sample_indices(
        "datasample",
        shape[axis],
        args.k,
        args.replacement,
        args.weights.as_deref(),
    )?;
    let idx_value = indices_value(&indices)?;
    let sample = sample_value_axis(args.data, axis, &indices, "datasample")?;
    Ok((sample, idx_value))
}

pub mod datasample {
    use super::*;
    sampling_descriptor!(
        "datasample",
        DATASAMPLE_SIGNATURES,
        BuiltinOutputMode::ByRequestedOutputCount
    );

    #[runtime_builtin(
        name = "datasample",
        category = "stats/random",
        summary = "Randomly sample from data with or without replacement.",
        keywords = "datasample,random,sample,replacement,weights,statistics",
        type_resolver(super::sampling_type),
        descriptor(self::DESCRIPTOR),
        extensions(super::DATASAMPLE_EXTENSIONS),
        integer_capabilities(super::DATASAMPLE_INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::random::sampling::datasample"
    )]
    pub(crate) async fn datasample_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        super::ensure_datasample_extensions(&value, &rest)?;
        let value = super::gathered(value, "datasample").await?;
        let mut gathered_rest = Vec::with_capacity(rest.len());
        for argument in rest {
            gathered_rest.push(super::gathered(argument, "datasample").await?);
        }
        let args = super::parse_datasample_args(value, gathered_rest).await?;
        let (sample, idx) = super::datasample_compute(args)?;
        match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::OutputList(Vec::new())),
            Some(1) => Ok(Value::OutputList(vec![sample])),
            Some(out_count) => Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![sample, idx],
            )),
            None => Ok(sample),
        }
    }
}

enum RandsamplePopulation {
    Range(usize),
    Values(Value, Vec<usize>, usize),
}

struct RandsampleArgs {
    population: RandsamplePopulation,
    k: usize,
    replacement: bool,
    weights: Option<Vec<f64>>,
}

async fn parse_randsample_args(args: Vec<Value>) -> BuiltinResult<RandsampleArgs> {
    if args.len() < 2 {
        return Err(sampling_error(
            "randsample",
            "randsample: population and k are required",
        ));
    }
    let first = gathered(args[0].clone(), "randsample").await?;
    let k = parse_positive_usize("randsample", &args[1], "k")?;
    let mut replacement = false;
    let mut weights_value = None;
    match args.len() {
        2 => {}
        3 => replacement = parse_bool("randsample", &args[2], "replacement")?,
        4 => {
            replacement = parse_bool("randsample", &args[2], "replacement")?;
            weights_value = Some(gathered(args[3].clone(), "randsample").await?);
        }
        _ => {
            return Err(sampling_error(
                "randsample",
                "randsample: too many arguments",
            ))
        }
    }
    let population = match &first {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            RandsamplePopulation::Range(parse_positive_usize("randsample", &first, "n")?)
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            RandsamplePopulation::Range(parse_positive_usize("randsample", &first, "n")?)
        }
        _ => {
            let shape = shape_of_sampled_value(&first).map_err(|err| {
                sampling_error("randsample", format!("randsample: {}", err.message()))
            })?;
            if shape.iter().filter(|dim| **dim > 1).count() > 1 {
                return Err(sampling_error(
                    "randsample",
                    "randsample: population must be a vector",
                ));
            }
            let axis = first_non_singleton(&shape);
            RandsamplePopulation::Values(first, shape, axis)
        }
    };
    let pop_len = match &population {
        RandsamplePopulation::Range(n) => *n,
        RandsamplePopulation::Values(_, shape, axis) => shape[*axis],
    };
    let weights = match weights_value {
        Some(value) => {
            if !replacement {
                return Err(sampling_error(
                    "randsample",
                    "randsample: weights require sampling with replacement",
                ));
            }
            Some(parse_weights("randsample", value, pop_len)?)
        }
        None => None,
    };
    Ok(RandsampleArgs {
        population,
        k,
        replacement,
        weights,
    })
}

fn randsample_compute(args: RandsampleArgs) -> BuiltinResult<Value> {
    match args.population {
        RandsamplePopulation::Range(n) => {
            let indices = sample_indices(
                "randsample",
                n,
                args.k,
                args.replacement,
                args.weights.as_deref(),
            )?;
            Tensor::new(
                indices.into_iter().map(|idx| (idx + 1) as f64).collect(),
                if args.k == 1 {
                    vec![1, 1]
                } else {
                    vec![args.k, 1]
                },
            )
            .map(tensor::tensor_into_value)
            .map_err(|err| sampling_error("randsample", format!("randsample: {err}")))
        }
        RandsamplePopulation::Values(value, shape, axis) => {
            let indices = sample_indices(
                "randsample",
                shape[axis],
                args.k,
                args.replacement,
                args.weights.as_deref(),
            )?;
            sample_value_axis(value, axis, &indices, "randsample")
        }
    }
}

pub mod randsample {
    use super::*;
    sampling_descriptor!(
        "randsample",
        RANDSAMPLE_SIGNATURES,
        BuiltinOutputMode::Fixed
    );

    #[runtime_builtin(
        name = "randsample",
        category = "stats/random",
        summary = "Randomly sample from a range or population vector.",
        keywords = "randsample,random,sample,replacement,weights,statistics",
        type_resolver(super::sampling_type),
        descriptor(self::DESCRIPTOR),
        extensions(super::RANDSAMPLE_EXTENSIONS),
        integer_capabilities(super::RANDSAMPLE_INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::random::sampling::randsample"
    )]
    pub(crate) async fn randsample_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        ensure_randsample_extensions(&args).await?;
        let resident_population = args.first().and_then(|value| match value {
            Value::GpuTensor(handle) if tensor::element_count(&handle.shape) > 1 => {
                Some(handle.clone())
            }
            _ => None,
        });
        let args = super::parse_randsample_args(args).await?;
        let output = super::randsample_compute(args)?;
        if let Some(source) = resident_population {
            let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(&source)
                .ok_or_else(|| {
                    sampling_error(
                        "randsample",
                        "randsample: resident population owner is unavailable",
                    )
                })?;
            return crate::builtins::math::trigonometry::inverse_helpers::upload_value_like(
                provider,
                output,
                "randsample",
                &source,
            );
        }
        Ok(output)
    }
}

async fn ensure_randsample_extensions(args: &[Value]) -> BuiltinResult<()> {
    if args.len() < 2 {
        return Ok(());
    }
    let first = &args[0];
    if crate::builtins::common::validation::value_has_native_integer_class(first) {
        let scalar = match first {
            Value::Int(_) => true,
            Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
            Value::GpuTensor(handle) => tensor::element_count(&handle.shape) == 1,
            _ => false,
        };
        crate::compatibility::ensure_builtin_extension_enabled(
            if scalar {
                &RANDSAMPLE_INTEGER_N_EXTENSION
            } else {
                &RANDSAMPLE_INTEGER_POPULATION_EXTENSION
            },
            "randsample",
        )?;
    }
    if crate::builtins::common::validation::value_has_native_integer_class(&args[1]) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RANDSAMPLE_INTEGER_K_EXTENSION,
            "randsample",
        )?;
    }
    if let Some(replacement) = args.get(2) {
        if crate::builtins::common::validation::value_has_native_integer_class(replacement) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &RANDSAMPLE_INTEGER_REPLACEMENT_EXTENSION,
                "randsample",
            )?;
        }
    }
    if let Some(weights) = args.get(3) {
        crate::builtins::common::validation::reject_typed_complex_integer(weights, "randsample")?;
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            weights,
            &RANDSAMPLE_INTEGER_WEIGHTS_EXTENSION,
            "randsample",
            "weights",
        )
        .await?;
    }
    Ok(())
}

async fn parse_shape_args(name: &str, rest: &[Value]) -> BuiltinResult<Vec<usize>> {
    if rest.is_empty() {
        return Ok(vec![1, 1]);
    }
    let mut dims = Vec::new();
    for arg in rest {
        match crate::builtins::common::random_args::extract_dims(arg, name).await {
            Ok(Some(values)) => dims.extend(values),
            Ok(None) => {
                return Err(sampling_error(
                    name,
                    format!("{name}: invalid size argument {arg:?}"),
                ));
            }
            Err(err) => return Err(sampling_error(name, err)),
        }
    }
    if dims.is_empty() {
        Ok(vec![0, 0])
    } else if dims.len() == 1 {
        Ok(vec![dims[0], dims[0]])
    } else {
        while dims.len() > 2 && dims.last() == Some(&1) {
            dims.pop();
        }
        Ok(dims)
    }
}

async fn parse_unidrnd_args(args: Vec<Value>) -> BuiltinResult<(Tensor, Vec<usize>)> {
    if args.is_empty() {
        return Err(sampling_error("unidrnd", "unidrnd: n is required"));
    }
    let n = tensor::value_into_tensor_for("unidrnd", gathered(args[0].clone(), "unidrnd").await?)
        .map_err(|err| sampling_error("unidrnd", format!("unidrnd: {err}")))?;
    let n_values = tensor::tensor_values_f64(&n);
    if n_values
        .iter()
        .any(|value| !value.is_finite() || *value < 1.0 || value.fract() != 0.0)
    {
        return Err(sampling_error(
            "unidrnd",
            "unidrnd: n must contain positive integers",
        ));
    }
    let shape = if args.len() == 1 {
        normalize_shape(n.shape.clone())
    } else {
        parse_shape_args("unidrnd", &args[1..]).await?
    };
    if !tensor::is_scalar_tensor(&n) && normalize_shape(n.shape.clone()) != shape {
        return Err(sampling_error(
            "unidrnd",
            "unidrnd: requested size must match non-scalar n",
        ));
    }
    Ok((n, shape))
}

pub mod unidrnd {
    use super::*;
    sampling_descriptor!("unidrnd", UNIDRND_SIGNATURES, BuiltinOutputMode::Fixed);

    #[runtime_builtin(
        name = "unidrnd",
        category = "stats/random",
        summary = "Generate random integers from a discrete uniform distribution.",
        keywords = "unidrnd,uniform,discrete,random,integer,statistics",
        type_resolver(super::numeric_type),
        descriptor(self::DESCRIPTOR),
        extensions(UNIDRND_EXTENSIONS),
        integer_capabilities(UNIDRND_INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::random::sampling::unidrnd"
    )]
    pub(crate) async fn unidrnd_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        if let Some(value) = args.first() {
            crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                value,
                &UNIDRND_INTEGER_LIMIT_EXTENSION,
                "unidrnd",
                "upper-limit",
            )
            .await?;
        }
        if args
            .iter()
            .skip(1)
            .any(crate::builtins::common::validation::value_has_native_integer_class)
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &UNIDRND_INTEGER_SIZE_EXTENSION,
                "unidrnd",
            )?;
        }
        let (n, shape) = super::parse_unidrnd_args(args).await?;
        let len = tensor::element_count(&shape);
        let uniforms = random::generate_uniform(len, "unidrnd")?;
        let data = uniforms
            .into_iter()
            .enumerate()
            .map(|(idx, u)| {
                let upper = if tensor::is_scalar_tensor(&n) {
                    tensor::tensor_value_f64(&n, 0)
                } else {
                    tensor::tensor_value_f64(&n, idx)
                };
                (u * upper).floor() + 1.0
            })
            .collect();
        Tensor::new(data, shape)
            .map(tensor::tensor_into_value)
            .map_err(|err| sampling_error("unidrnd", format!("unidrnd: {err}")))
    }
}

#[derive(Clone, Copy)]
struct DividerandArgs {
    q: usize,
    ratios: [f64; 3],
}

async fn parse_dividerand_args(args: Vec<Value>) -> BuiltinResult<DividerandArgs> {
    if args.len() != 1 && args.len() != 4 {
        return Err(sampling_error(
            "dividerand",
            "dividerand: expected Q or Q, trainRatio, valRatio, testRatio",
        ));
    }
    let q = parse_nonnegative_usize(
        "dividerand",
        gathered(args[0].clone(), "dividerand").await?,
        "Q",
    )?;
    let ratios = if args.len() == 1 {
        [0.7, 0.15, 0.15]
    } else {
        [
            parse_nonnegative_scalar_ratio(
                gathered(args[1].clone(), "dividerand").await?,
                "trainRatio",
            )?,
            parse_nonnegative_scalar_ratio(
                gathered(args[2].clone(), "dividerand").await?,
                "valRatio",
            )?,
            parse_nonnegative_scalar_ratio(
                gathered(args[3].clone(), "dividerand").await?,
                "testRatio",
            )?,
        ]
    };
    if ratios.iter().sum::<f64>() <= 0.0 {
        return Err(sampling_error(
            "dividerand",
            "dividerand: at least one ratio must be positive",
        ));
    }
    Ok(DividerandArgs { q, ratios })
}

fn parse_nonnegative_usize(name: &str, value: Value, label: &str) -> BuiltinResult<usize> {
    if let Some(value) = scalar_integer_value(&value) {
        return value.try_to_usize().ok_or_else(|| {
            sampling_error(
                name,
                format!("{name}: {label} must be a nonnegative integer"),
            )
        });
    }
    let tensor = tensor::value_into_tensor_for(name, value)
        .map_err(|err| sampling_error(name, format!("{name}: {err}")))?;
    if !tensor::is_scalar_tensor(&tensor) {
        return Err(sampling_error(
            name,
            format!("{name}: {label} must be a scalar"),
        ));
    }
    let raw = tensor::tensor_value_f64(&tensor, 0);
    if !raw.is_finite()
        || raw < 0.0
        || raw.fract() != 0.0
        || raw > usize::MAX as f64
        || (usize::BITS == 64 && raw == usize::MAX as f64)
    {
        return Err(sampling_error(
            name,
            format!("{name}: {label} must be a nonnegative integer"),
        ));
    }
    Ok(raw as usize)
}

fn parse_nonnegative_scalar_ratio(value: Value, label: &str) -> BuiltinResult<f64> {
    let tensor = tensor::value_into_tensor_for("dividerand", value)
        .map_err(|err| sampling_error("dividerand", format!("dividerand: {err}")))?;
    if !tensor::is_scalar_tensor(&tensor) {
        return Err(sampling_error(
            "dividerand",
            format!("dividerand: {label} must be a scalar"),
        ));
    }
    let raw = tensor::tensor_value_f64(&tensor, 0);
    if !raw.is_finite() || raw < 0.0 {
        return Err(sampling_error(
            "dividerand",
            format!("dividerand: {label} must be a finite nonnegative scalar"),
        ));
    }
    Ok(raw)
}

fn dividerand_counts(q: usize, ratios: [f64; 3]) -> [usize; 3] {
    if q == 0 {
        return [0, 0, 0];
    }
    let max_ratio = ratios.iter().copied().fold(0.0, f64::max);
    let normalized = ratios.map(|ratio| ratio / max_ratio);
    let total = normalized.iter().sum::<f64>();
    let mut counts = [0usize; 3];
    let mut remainders = [(0usize, 0.0f64); 3];
    let mut assigned = 0usize;
    for (idx, ratio) in normalized.iter().enumerate() {
        let exact = (*ratio / total) * q as f64;
        let count = exact.floor().min(q as f64) as usize;
        counts[idx] = count;
        assigned += count;
        remainders[idx] = (idx, exact - count as f64);
    }
    remainders.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    for (idx, _) in remainders.iter().take(q.saturating_sub(assigned)) {
        counts[*idx] += 1;
    }
    counts
}

fn dividerand_permutation(q: usize) -> BuiltinResult<Vec<usize>> {
    if q > MAX_DIVIDERAND_Q {
        return Err(sampling_error(
            "dividerand",
            format!("dividerand: Q exceeds the maximum supported value of {MAX_DIVIDERAND_Q}"),
        ));
    }
    let uniforms = random::generate_uniform(q, "dividerand")?;
    let mut pool = Vec::new();
    pool.try_reserve_exact(q)
        .map_err(|_| sampling_error("dividerand", "dividerand: requested output is too large"))?;
    pool.extend(0..q);
    for (draw, u) in uniforms.into_iter().enumerate() {
        let span = q - draw;
        let offset = ((u * span as f64).floor() as usize).min(span - 1);
        pool.swap(draw, draw + offset);
    }
    Ok(pool)
}

fn row_index_value(indices: &[usize]) -> BuiltinResult<Value> {
    let mut data = Vec::new();
    data.try_reserve_exact(indices.len())
        .map_err(|_| sampling_error("dividerand", "dividerand: requested output is too large"))?;
    data.extend(indices.iter().map(|idx| (idx + 1) as f64));
    Tensor::new(data, vec![1, indices.len()])
        .map(Value::Tensor)
        .map_err(|err| sampling_error("dividerand", format!("dividerand: {err}")))
}

fn dividerand_compute(args: DividerandArgs) -> BuiltinResult<[Value; 3]> {
    let counts = dividerand_counts(args.q, args.ratios);
    let permutation = if args.q == 0 {
        Vec::new()
    } else {
        dividerand_permutation(args.q)?
    };
    let train_end = counts[0];
    let val_end = train_end + counts[1];
    Ok([
        row_index_value(&permutation[..train_end])?,
        row_index_value(&permutation[train_end..val_end])?,
        row_index_value(&permutation[val_end..])?,
    ])
}

pub mod dividerand {
    use super::*;
    sampling_descriptor!(
        "dividerand",
        DIVIDERAND_SIGNATURES,
        BuiltinOutputMode::ByRequestedOutputCount
    );

    #[runtime_builtin(
        name = "dividerand",
        category = "stats/random",
        summary = "Divide target indices randomly into training, validation, and test sets.",
        keywords = "dividerand,random,partition,train,validation,test,statistics,machine-learning",
        type_resolver(super::sampling_type),
        extensions(super::DIVIDERAND_EXTENSIONS),
        integer_capabilities(super::DIVIDERAND_INTEGER_CAPABILITIES),
        descriptor(self::DESCRIPTOR),
        builtin_path = "crate::builtins::stats::random::sampling::dividerand"
    )]
    pub(crate) async fn dividerand_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        if args
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &super::DIVIDERAND_RESIDENT_ARGUMENT_EXTENSION,
                "dividerand",
            )?;
        }
        let parsed = super::parse_dividerand_args(args).await?;
        let [train, val, test] = super::dividerand_compute(parsed)?;
        match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::OutputList(Vec::new())),
            Some(1) => Ok(Value::OutputList(vec![train])),
            Some(2) => Ok(Value::OutputList(vec![train, val])),
            Some(3) => Ok(Value::OutputList(vec![train, val, test])),
            None => Ok(train),
            Some(_) => Err(super::sampling_error(
                "dividerand",
                "dividerand: too many output arguments",
            )),
        }
    }
}

struct BootstrpArgs {
    nboot: usize,
    bootfun: Value,
    data: Vec<Value>,
    sample_axis: usize,
    sample_len: usize,
    weights: Option<Vec<f64>>,
}

struct BootstrpEval {
    bootstat: Value,
    bootsam: Option<Value>,
}

fn is_integer_bootstrp_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn is_logical_bootstrp_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

fn is_text_bootstrp_callable(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn enable_bootstrp_extension(extension: &BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, "bootstrp")
}

fn ensure_bootstrp_extensions_enabled(args: &[Value]) -> BuiltinResult<()> {
    if args.len() < 3 {
        return Ok(());
    }
    if is_integer_bootstrp_value(&args[0]) {
        enable_bootstrp_extension(&BOOTSTRP_INTEGER_NBOOT_EXTENSION)?;
    }
    if is_logical_bootstrp_value(&args[0]) {
        enable_bootstrp_extension(&BOOTSTRP_LOGICAL_NBOOT_EXTENSION)?;
    }
    if is_text_bootstrp_callable(&args[1]) {
        enable_bootstrp_extension(&BOOTSTRP_TEXT_CALLABLE_EXTENSION)?;
    }

    let mut idx = 2usize;
    while idx < args.len() {
        if let Some(keyword) = keyword_of(&args[idx]) {
            match keyword.as_str() {
                "weights" => {
                    if let Some(value) = args.get(idx + 1) {
                        if is_integer_bootstrp_value(value) {
                            enable_bootstrp_extension(&BOOTSTRP_INTEGER_WEIGHTS_EXTENSION)?;
                        }
                        if is_logical_bootstrp_value(value) {
                            enable_bootstrp_extension(&BOOTSTRP_LOGICAL_WEIGHTS_EXTENSION)?;
                        }
                    }
                    idx += 2;
                    continue;
                }
                "options" => {
                    idx += 2;
                    continue;
                }
                _ => {}
            }
        }
        if is_integer_bootstrp_value(&args[idx]) {
            enable_bootstrp_extension(&BOOTSTRP_INTEGER_DATA_EXTENSION)?;
        }
        idx += 1;
    }

    if args
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        enable_bootstrp_extension(&BOOTSTRP_GPU_INPUT_EXTENSION)?;
    }
    Ok(())
}

fn is_empty_function(value: &Value) -> bool {
    match value {
        Value::Tensor(t) => tensor::tensor_element_len(t) == 0,
        Value::LogicalArray(a) => a.data.is_empty(),
        Value::Cell(c) => c.data.is_empty(),
        Value::String(s) => s.is_empty(),
        Value::StringArray(a) => a.data.is_empty(),
        Value::CharArray(a) => a.data.is_empty(),
        _ => false,
    }
}

fn is_scalar_boot_arg(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(_) => true,
        Value::Tensor(t) => tensor::is_scalar_tensor(t),
        Value::LogicalArray(a) => a.data.len() == 1,
        Value::StringArray(a) => a.data.len() == 1,
        Value::CharArray(a) => a.data.len() == 1,
        Value::Cell(a) => a.data.len() == 1,
        _ => false,
    }
}

fn bootstrp_data_axis(
    value: &Value,
    single_data_arg: bool,
) -> BuiltinResult<Option<(usize, usize)>> {
    if is_scalar_boot_arg(value) {
        return Ok(None);
    }
    let shape = shape_of_sampled_value(value)
        .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {}", err.message())))?;
    if single_data_arg && shape.iter().filter(|dim| **dim > 1).count() == 1 {
        let axis = first_non_singleton(&shape);
        Ok(Some((axis, shape[axis])))
    } else {
        Ok(Some((0, shape[0])))
    }
}

async fn parse_bootstrp_args(args: Vec<Value>) -> BuiltinResult<BootstrpArgs> {
    if args.len() < 3 {
        return Err(sampling_error(
            "bootstrp",
            "bootstrp: expected nboot, bootfun, and at least one data argument",
        ));
    }
    let nboot_value = gathered(args[0].clone(), "bootstrp").await?;
    let nboot = parse_positive_usize("bootstrp", &nboot_value, "nboot")?;
    let bootfun = gathered(args[1].clone(), "bootstrp").await?;
    let mut data = Vec::new();
    let mut weight_value = None;
    let mut idx = 2usize;
    while idx < args.len() {
        if let Some(keyword) = keyword_of(&args[idx]) {
            match keyword.as_str() {
                "weights" => {
                    let Some(value) = args.get(idx + 1) else {
                        return Err(sampling_error(
                            "bootstrp",
                            "bootstrp: weights requires a value",
                        ));
                    };
                    weight_value = Some(gathered(value.clone(), "bootstrp").await?);
                    idx += 2;
                    continue;
                }
                "options" => {
                    let Some(value) = args.get(idx + 1) else {
                        return Err(sampling_error(
                            "bootstrp",
                            "bootstrp: options requires a value",
                        ));
                    };
                    let options = gathered(value.clone(), "bootstrp").await?;
                    if !matches!(options, Value::Struct(_)) {
                        return Err(sampling_error(
                            "bootstrp",
                            "bootstrp: Options must be a statset/options struct",
                        ));
                    }
                    idx += 2;
                    continue;
                }
                _ => {}
            }
        }
        data.push(gathered(args[idx].clone(), "bootstrp").await?);
        idx += 1;
    }
    if data.is_empty() {
        return Err(sampling_error(
            "bootstrp",
            "bootstrp: at least one data argument is required",
        ));
    }
    let single_nonscalar_arg = data
        .iter()
        .filter(|value| !is_scalar_boot_arg(value))
        .count()
        == 1;
    let mut sample_axis = 0usize;
    let mut sample_len = None;
    for value in &data {
        let Some((axis, len)) = bootstrp_data_axis(value, single_nonscalar_arg)? else {
            continue;
        };
        if let Some(expected) = sample_len {
            if len != expected {
                return Err(sampling_error(
                    "bootstrp",
                    "bootstrp: nonscalar data arguments must have the same number of rows",
                ));
            }
        } else {
            sample_axis = axis;
            sample_len = Some(len);
        }
    }
    let sample_len = sample_len.unwrap_or(1);
    if sample_len == 0 {
        return Err(sampling_error("bootstrp", "bootstrp: data cannot be empty"));
    }
    let weights = match weight_value {
        Some(value) => Some(parse_weights("bootstrp", value, sample_len)?),
        None => None,
    };
    Ok(BootstrpArgs {
        nboot,
        bootfun,
        data,
        sample_axis,
        sample_len,
        weights,
    })
}

enum BootstatRow {
    Numeric(NumericStorage),
    Logical(Vec<u8>),
}

impl BootstatRow {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(storage) => storage.len(),
            Self::Logical(values) => values.len(),
        }
    }

    fn class_name(&self) -> &'static str {
        match self {
            Self::Numeric(storage) => storage.class_name(),
            Self::Logical(_) => "logical",
        }
    }
}

async fn bootstat_row(value: Value) -> BuiltinResult<BootstatRow> {
    let mut value = gather_if_needed_async(&value)
        .await
        .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))?;
    if let Value::OutputList(values) = value {
        if values.len() != 1 {
            return Err(sampling_error(
                "bootstrp",
                "bootstrp: bootfun must return exactly one output",
            ));
        }
        value = values.into_iter().next().unwrap_or(Value::Num(0.0));
        value = gather_if_needed_async(&value)
            .await
            .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))?;
    }
    match value {
        Value::OutputList(_) => Err(sampling_error(
            "bootstrp",
            "bootstrp: bootfun must return exactly one output",
        )),
        Value::Num(v) => Ok(BootstatRow::Numeric(NumericStorage::F64(vec![v]))),
        Value::Int(v) => Ok(BootstatRow::Numeric(
            runmat_builtins::IntegerStorage::from_scalar(v).into(),
        )),
        Value::Bool(v) => Ok(BootstatRow::Logical(vec![u8::from(v)])),
        Value::Tensor(t) => t
            .into_numeric_storage()
            .map(BootstatRow::Numeric)
            .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}"))),
        Value::LogicalArray(a) => Ok(BootstatRow::Logical(a.data)),
        other => Err(sampling_error(
            "bootstrp",
            format!("bootstrp: bootfun must return numeric or logical values, got {other:?}"),
        )),
    }
}

fn bootsam_value(samples: &[Vec<usize>], n: usize, nboot: usize) -> BuiltinResult<Value> {
    let len = n.checked_mul(nboot).ok_or_else(|| {
        sampling_error(
            "bootstrp",
            "bootstrp: bootstrap sample index output is too large",
        )
    })?;
    let mut data = Vec::with_capacity(len);
    for sample in samples {
        data.extend(sample.iter().map(|idx| (*idx + 1) as f64));
    }
    Tensor::new(data, vec![n, nboot])
        .map(Value::Tensor)
        .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))
}

fn empty_bootstat(nboot: usize) -> BuiltinResult<Value> {
    Tensor::new(Vec::new(), vec![nboot, 0])
        .map(Value::Tensor)
        .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))
}

fn assemble_bootstat(rows: Vec<BootstatRow>, nboot: usize) -> BuiltinResult<Value> {
    let Some(first) = rows.first() else {
        return empty_bootstat(nboot);
    };
    let width = first.len();
    let class_name = first.class_name();
    if rows
        .iter()
        .any(|row| row.len() != width || row.class_name() != class_name)
    {
        return Err(sampling_error(
            "bootstrp",
            "bootstrp: bootfun output size and class must be consistent across bootstrap samples",
        ));
    }
    let len = nboot.checked_mul(width).ok_or_else(|| {
        sampling_error(
            "bootstrp",
            "bootstrp: bootstrap statistic output is too large",
        )
    })?;

    match first {
        BootstatRow::Numeric(first_storage) => {
            let mut storage = NumericStorage::zeros(first_storage.numeric_dtype(), len);
            for (boot_idx, row) in rows.iter().enumerate() {
                let BootstatRow::Numeric(row_storage) = row else {
                    unreachable!("class consistency checked above");
                };
                for col in 0..width {
                    let value = row_storage.value_at(col).ok_or_else(|| {
                        sampling_error(
                            "bootstrp",
                            "bootstrp: callback output storage does not match its shape",
                        )
                    })?;
                    storage
                        .set_value(boot_idx + col * nboot, value)
                        .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))?;
                }
            }
            Tensor::from_numeric_storage(storage, vec![nboot, width])
                .map(tensor::tensor_into_value)
                .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))
        }
        BootstatRow::Logical(_) => {
            let mut data = vec![0u8; len];
            for (boot_idx, row) in rows.iter().enumerate() {
                let BootstatRow::Logical(values) = row else {
                    unreachable!("class consistency checked above");
                };
                for (col, value) in values.iter().enumerate() {
                    data[boot_idx + col * nboot] = u8::from(*value != 0);
                }
            }
            LogicalArray::new(data, vec![nboot, width])
                .map(Value::LogicalArray)
                .map_err(|err| sampling_error("bootstrp", format!("bootstrp: {err}")))
        }
    }
}

async fn bootstrp_compute(
    args: BootstrpArgs,
    include_bootsam: bool,
) -> BuiltinResult<BootstrpEval> {
    let mut samples = if include_bootsam {
        Some(Vec::with_capacity(args.nboot))
    } else {
        None
    };
    let mut rows = Vec::with_capacity(args.nboot);
    for _ in 0..args.nboot {
        let indices = sample_indices(
            "bootstrp",
            args.sample_len,
            args.sample_len,
            true,
            args.weights.as_deref(),
        )?;
        if !is_empty_function(&args.bootfun) {
            let mut callback_args = Vec::with_capacity(args.data.len());
            for value in args.data.iter().cloned() {
                if is_scalar_boot_arg(&value) {
                    callback_args.push(value);
                } else {
                    callback_args.push(sample_value_axis(
                        value,
                        args.sample_axis,
                        &indices,
                        "bootstrp",
                    )?);
                }
            }
            let result =
                crate::call_feval_async_with_outputs(args.bootfun.clone(), &callback_args, 1)
                    .await
                    .map_err(|err| {
                        sampling_error(
                            "bootstrp",
                            format!("bootstrp: bootfun failed: {}", err.message()),
                        )
                    })?;
            rows.push(bootstat_row(result).await?);
        }
        if let Some(samples) = samples.as_mut() {
            samples.push(indices);
        }
    }
    let bootsam = match samples {
        Some(samples) => Some(bootsam_value(&samples, args.sample_len, args.nboot)?),
        None => None,
    };
    let bootstat = if is_empty_function(&args.bootfun) {
        empty_bootstat(args.nboot)?
    } else {
        assemble_bootstat(rows, args.nboot)?
    };
    Ok(BootstrpEval { bootstat, bootsam })
}

pub mod bootstrp {
    use super::*;
    sampling_descriptor!(
        "bootstrp",
        BOOTSTRP_SIGNATURES,
        BuiltinOutputMode::ByRequestedOutputCount
    );

    #[runtime_builtin(
        name = "bootstrp",
        category = "stats/random",
        summary = "Bootstrap samples and evaluate a statistic.",
        keywords = "bootstrp,bootstrap,resampling,statistics,weights",
        type_resolver(super::sampling_type),
        descriptor(self::DESCRIPTOR),
        extensions(super::BOOTSTRP_EXTENSIONS),
        integer_capabilities(super::BOOTSTRP_INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::random::sampling::bootstrp"
    )]
    pub(crate) async fn bootstrp_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        let requested_outputs = crate::output_count::current_output_count();
        match requested_outputs {
            Some(0) => return Ok(Value::OutputList(Vec::new())),
            Some(count) if count > 2 => {
                return Err(super::sampling_error(
                    "bootstrp",
                    "bootstrp: too many output arguments; maximum is 2",
                ));
            }
            _ => {}
        }
        super::ensure_bootstrp_extensions_enabled(&args)?;
        let include_bootsam = matches!(requested_outputs, Some(2));
        let args = super::parse_bootstrp_args(args).await?;
        let eval = super::bootstrp_compute(args, include_bootsam).await?;
        match requested_outputs {
            Some(1) => Ok(Value::OutputList(vec![eval.bootstat])),
            Some(2) => Ok(Value::OutputList(vec![
                eval.bootstat,
                eval.bootsam.unwrap_or(Value::Num(0.0)),
            ])),
            None => Ok(eval.bootstat),
            Some(0) | Some(_) => unreachable!("validated output count before evaluation"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{builtin_function_by_name, IntegerStorage, NumericStorage};

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Value {
        let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
        if tensor::element_count(&tensor.shape) == 0 {
        } else {
        }
        Value::Tensor(tensor)
    }

    #[cfg(feature = "wgpu")]
    fn all_population_integer_storages() -> Vec<IntegerStorage> {
        vec![
            IntegerStorage::I8(vec![1, 2]),
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I64(vec![1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ]
    }

    #[test]
    fn boolean_options_read_every_integer_storage_variant_not_the_float_mirror() {
        for storage in [
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ] {
            assert!(parse_bool(
                "datasample",
                &poisoned_int_tensor(storage, vec![1, 1], f64::NAN),
                "Replace"
            )
            .unwrap());
        }
    }

    #[test]
    fn sampling_tensor_axis_preserves_every_native_numeric_class() {
        let storages = [
            NumericStorage::F64(vec![1.0, 2.0, 3.0, 4.0]),
            NumericStorage::F32(vec![1.0, 2.0, 3.0, 4.0]),
            NumericStorage::I8(vec![1, 2, 3, 4]),
            NumericStorage::I16(vec![1, 2, 3, 4]),
            NumericStorage::I32(vec![1, 2, 3, 4]),
            NumericStorage::I64(vec![1, 2, 3, 4]),
            NumericStorage::U8(vec![1, 2, 3, 4]),
            NumericStorage::U16(vec![1, 2, 3, 4]),
            NumericStorage::U32(vec![1, 2, 3, 4]),
            NumericStorage::U64(vec![1, 2, 3, 4]),
        ];

        for storage in storages {
            let expected = storage.gather(&[1, 0, 3, 2]).unwrap();
            let tensor = Tensor::from_numeric_storage(storage, vec![2, 2]).unwrap();
            let sampled =
                sample_value_axis(Value::Tensor(tensor), 0, &[1, 0], "datasample").expect("sample");
            let Value::Tensor(sampled) = sampled else {
                panic!("expected sampled tensor");
            };
            assert_eq!(sampled.shape, vec![2, 2]);
            assert_eq!(sampled.into_numeric_storage(), Ok(expected));
        }
    }

    #[test]
    fn datasample_samples_rows_and_returns_indices() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let data =
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], vec![3, 2]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(datasample::datasample_builtin(
            data,
            vec![Value::Num(2.0), Value::from("Replace"), Value::Bool(false)],
        ))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![2, 2]);
                        assert_eq!(t.materialize_f64().len(), 4);
                    }
                    other => panic!("expected tensor sample, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![2, 1]);
                        assert!(t
                            .materialize_f64()
                            .iter()
                            .all(|idx| (1.0..=3.0).contains(idx)));
                    }
                    other => panic!("expected tensor indices, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn datasample_supports_char_weights() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let data = Value::CharArray(CharArray::new_row("ACGT"));
        let out = block_on(datasample::datasample_builtin(
            data,
            vec![
                Value::Num(5.0),
                Value::from("Weights"),
                Value::Tensor(Tensor::new(vec![0.0, 0.0, 1.0, 0.0], vec![1, 4]).unwrap()),
            ],
        ))
        .unwrap();
        match out {
            Value::CharArray(chars) => {
                assert_eq!(chars.rows, 1);
                assert_eq!(chars.cols, 5);
                assert_eq!(chars.data, vec!['G'; 5]);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn datasample_supports_cell_arrays() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let data = Value::Cell(
            CellArray::new(
                vec![
                    Value::from("a"),
                    Value::from("b"),
                    Value::from("c"),
                    Value::from("d"),
                ],
                2,
                2,
            )
            .unwrap(),
        );
        let out = block_on(datasample::datasample_builtin(
            data,
            vec![
                Value::Num(3.0),
                Value::from("Weights"),
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            ],
        ))
        .unwrap();
        match out {
            Value::Cell(cell) => {
                assert_eq!(cell.shape, vec![3, 2]);
                assert_eq!(
                    cell.data,
                    vec![
                        Value::from("c"),
                        Value::from("d"),
                        Value::from("c"),
                        Value::from("d"),
                        Value::from("c"),
                        Value::from("d"),
                    ]
                );
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[test]
    fn datasample_extensions_are_independently_mode_gated() {
        let integer_data = || {
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).unwrap())
        };
        let data = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let integer_weights = || {
            Value::Tensor(Tensor::new_integer(IntegerStorage::U8(vec![1, 1]), vec![2, 1]).unwrap())
        };
        let logical_weights =
            || Value::LogicalArray(LogicalArray::new(vec![1, 1], vec![2, 1]).unwrap());
        let cases = [
            (
                integer_data(),
                vec![Value::Num(1.0)],
                DATASAMPLE_INTEGER_DATA_EXTENSION.error_identifier,
            ),
            (
                data(),
                vec![Value::Int(IntValue::U8(1))],
                DATASAMPLE_INTEGER_K_EXTENSION.error_identifier,
            ),
            (
                data(),
                vec![Value::Num(1.0), Value::Int(IntValue::U8(1))],
                DATASAMPLE_INTEGER_DIM_EXTENSION.error_identifier,
            ),
            (
                data(),
                vec![Value::Num(1.0), Value::from("Weights"), integer_weights()],
                DATASAMPLE_INTEGER_WEIGHTS_EXTENSION.error_identifier,
            ),
            (
                data(),
                vec![Value::Num(1.0), Value::from("Weights"), logical_weights()],
                DATASAMPLE_LOGICAL_WEIGHTS_EXTENSION.error_identifier,
            ),
            (
                data(),
                vec![Value::Num(1.0), Value::from("Replace"), Value::Num(0.0)],
                DATASAMPLE_NUMERIC_REPLACE_EXTENSION.error_identifier,
            ),
        ];
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        for (population, rest, identifier) in cases {
            let error = block_on(datasample::datasample_builtin(population, rest))
                .expect_err("strict rejection");
            assert_eq!(error.identifier(), identifier);
        }
    }

    #[test]
    fn datasample_preserves_every_integer_class_exactly() {
        let _lock = random::test_lock().lock().unwrap();
        random::set_seed(396).unwrap();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = [
            IntegerStorage::I8(vec![-8, 7]),
            IntegerStorage::I16(vec![-16, 15]),
            IntegerStorage::I32(vec![-32, 31]),
            IntegerStorage::I64(vec![-9_007_199_254_740_993, 7]),
            IntegerStorage::U8(vec![8, 7]),
            IntegerStorage::U16(vec![16, 15]),
            IntegerStorage::U32(vec![32, 31]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, 7]),
        ];
        for storage in storages {
            let expected = storage.value_at(0).unwrap();
            let population = Value::Tensor(Tensor::new_integer(storage, vec![2, 1]).unwrap());
            let weights = Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap());
            let Value::Tensor(output) = block_on(datasample::datasample_builtin(
                population,
                vec![Value::Num(3.0), Value::from("Weights"), weights],
            ))
            .unwrap() else {
                panic!("expected integer tensor");
            };
            assert_eq!(
                output.integer_storage().unwrap().exact_values(),
                vec![expected; 3]
            );
        }
    }

    #[test]
    fn datasample_capabilities_and_resident_gate_are_auditable() {
        assert_eq!(DATASAMPLE_INTEGER_CAPABILITIES.len(), 4);
        assert!(DATASAMPLE_INTEGER_CAPABILITIES
            .iter()
            .all(|capability| capability.inputs[0].classes.len() == 8));
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[2, 1],
                })
                .unwrap();
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(datasample::datasample_builtin(
                Value::GpuTensor(handle),
                vec![Value::Num(1.0)],
            ))
            .expect_err("resident input gate must run before gather");
            assert_eq!(
                error.identifier(),
                DATASAMPLE_RESIDENT_INPUT_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn randsample_range_and_population_vector() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let range = block_on(randsample::randsample_builtin(vec![
            Value::Num(5.0),
            Value::Num(3.0),
            Value::Bool(false),
        ]))
        .unwrap();
        match range {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert!(t
                    .materialize_f64()
                    .iter()
                    .all(|value| (1.0..=5.0).contains(value)));
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        random::reset_rng();
        let population = Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap());
        let out = block_on(randsample::randsample_builtin(vec![
            population,
            Value::Num(4.0),
            Value::Bool(true),
        ]))
        .unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert!(t
                    .materialize_f64()
                    .iter()
                    .all(|value| [10.0, 20.0, 30.0].contains(value)));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn randsample_restores_exact_integer_population_to_its_owner() {
        let _lock = random::test_lock().lock().unwrap();
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        random::reset_rng();
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let base = 9_007_199_254_740_992_u64;
            let values = [base, base + 1];
            let shape = [1usize, 2usize];
            let source = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&values),
                    shape: &shape,
                })
                .expect("upload integer population");
            let source =
                source.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let Value::GpuTensor(output) = block_on(randsample::randsample_builtin(vec![
                Value::GpuTensor(source),
                Value::Num(2.0),
            ]))
            .expect("resident integer population") else {
                panic!("expected resident output");
            };
            assert_eq!(output.shape, vec![1, 2]);
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&output),
                Some(runmat_accelerate_api::IntegerElementType::U64)
            );
            let downloaded =
                block_on(provider.download_integer(&output)).expect("download integer population");
            let runmat_accelerate_api::HostIntegerDataOwned::U64(mut sampled) = downloaded.data
            else {
                panic!("expected uint64 download");
            };
            sampled.sort_unstable();
            assert_eq!(sampled, values);
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn randsample_wgpu_preserves_residency_and_class_for_all_integer_populations() {
        use crate::builtins::common::{gpu_helpers, test_support};

        let _accel_guard = test_support::accel_test_lock();
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        for storage in all_population_integer_storages() {
            let expected_dtype = storage.numeric_dtype();
            let mut expected = storage.exact_values();
            expected.sort_by_key(|value| format!("{value:?}"));
            let population = Tensor::new_integer(storage, vec![1, 2]).expect("population");
            let source = gpu_helpers::upload_tensor(provider, &population).expect("upload");
            let source =
                source.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let output = block_on(randsample::randsample_builtin(vec![
                Value::GpuTensor(source),
                Value::Num(2.0),
            ]))
            .expect("resident integer population");
            let Value::GpuTensor(output_handle) = &output else {
                panic!("expected resident output");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(output_handle));
            let gathered = test_support::gather(output).expect("gather output");
            assert_eq!(gathered.numeric_dtype(), expected_dtype);
            let mut actual = gathered
                .integer_storage()
                .expect("integer output")
                .exact_values();
            actual.sort_by_key(|value| format!("{value:?}"));
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn unidrnd_generates_with_scalar_or_array_upper_bound() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let out = block_on(unidrnd::unidrnd_builtin(vec![
            Value::Num(3.0),
            Value::Num(2.0),
            Value::Num(2.0),
        ]))
        .unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert!(t
                    .materialize_f64()
                    .iter()
                    .all(|value| (1.0..=3.0).contains(value)));
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        random::reset_rng();
        let n = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let out = block_on(unidrnd::unidrnd_builtin(vec![n])).unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.materialize_f64()[0], 1.0);
                assert!((1.0..=2.0).contains(&t.materialize_f64()[1]));
                assert!((1.0..=3.0).contains(&t.materialize_f64()[2]));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn unidrnd_reads_typed_integer_upper_bound_storage_exactly() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let n = poisoned_int_tensor(IntegerStorage::U16(vec![3, 3, 3]), vec![3, 1], f64::NAN);
        let out = block_on(unidrnd::unidrnd_builtin(vec![n])).unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert!(t.materialize_f64().iter().all(|value| value.is_finite()));
                assert!(t
                    .materialize_f64()
                    .iter()
                    .all(|value| (1.0..=3.0).contains(value)));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn unidrnd_typed_integer_roles_are_gated_and_wide_limits_must_be_exact() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let compatibility = crate::compatibility::push_runmat_extensions_enabled(false);
        let limit_error = block_on(unidrnd::unidrnd_builtin(vec![Value::Int(IntValue::U16(3))]))
            .expect_err("typed limit must be gated");
        assert_eq!(
            limit_error.identifier(),
            UNIDRND_INTEGER_LIMIT_EXTENSION.error_identifier
        );
        let size_error = block_on(unidrnd::unidrnd_builtin(vec![
            Value::Num(3.0),
            Value::Int(IntValue::U8(2)),
        ]))
        .expect_err("typed size must be gated");
        assert_eq!(
            size_error.identifier(),
            UNIDRND_INTEGER_SIZE_EXTENSION.error_identifier
        );
        drop(compatibility);

        let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let lossy = block_on(unidrnd::unidrnd_builtin(vec![Value::Int(IntValue::U64(
            9_007_199_254_740_993,
        ))]))
        .expect_err("lossy upper limit must reject");
        assert!(lossy.message().contains("exactly representable"));
        drop(extensions);
    }

    #[test]
    fn dividerand_partitions_indices_into_row_vectors() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let _guard = crate::output_count::push_output_count(Some(3));
        let out = block_on(dividerand::dividerand_builtin(vec![
            Value::Num(10.0),
            Value::Num(0.6),
            Value::Num(0.2),
            Value::Num(0.2),
        ]))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 3);
                let mut seen = Vec::new();
                for (value, expected_len) in values.iter().zip([6usize, 2, 2]) {
                    match value {
                        Value::Tensor(t) => {
                            assert_eq!(t.shape, vec![1, expected_len]);
                            seen.extend_from_slice(&t.materialize_f64());
                        }
                        other => panic!("expected tensor indices, got {other:?}"),
                    }
                }
                seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert_eq!(seen, (1..=10).map(|idx| idx as f64).collect::<Vec<_>>());
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn dividerand_supports_defaults_and_empty_partitions() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        {
            let _guard = crate::output_count::push_output_count(Some(3));
            let out = block_on(dividerand::dividerand_builtin(vec![Value::Num(4.0)])).unwrap();
            match out {
                Value::OutputList(values) => {
                    let shapes = values
                        .iter()
                        .map(|value| match value {
                            Value::Tensor(t) => t.shape.clone(),
                            other => panic!("expected tensor indices, got {other:?}"),
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(shapes, vec![vec![1, 3], vec![1, 1], vec![1, 0]]);
                }
                other => panic!("expected output list, got {other:?}"),
            }
        }

        random::reset_rng();
        {
            let _guard = crate::output_count::push_output_count(Some(3));
            let out = block_on(dividerand::dividerand_builtin(vec![
                Value::Num(3.0),
                Value::Num(1.0),
                Value::Num(0.0),
                Value::Num(0.0),
            ]))
            .unwrap();
            match out {
                Value::OutputList(values) => {
                    assert!(matches!(&values[0], Value::Tensor(t) if t.shape == vec![1, 3]));
                    assert!(matches!(&values[1], Value::Tensor(t) if t.shape == vec![1, 0]));
                    assert!(matches!(&values[2], Value::Tensor(t) if t.shape == vec![1, 0]));
                }
                other => panic!("expected output list, got {other:?}"),
            }
        }
    }

    #[test]
    fn dividerand_q_accepts_every_typed_integer_scalar_exactly() {
        let values = [
            IntValue::I8(3),
            IntValue::I16(3),
            IntValue::I32(3),
            IntValue::I64(3),
            IntValue::U8(3),
            IntValue::U16(3),
            IntValue::U32(3),
            IntValue::U64(3),
        ];
        for value in values {
            assert_eq!(
                parse_nonnegative_usize("dividerand", Value::Int(value), "Q").unwrap(),
                3
            );
        }
        assert_eq!(DIVIDERAND_INTEGER_CAPABILITIES.len(), 1);
    }

    #[test]
    fn dividerand_resident_extension_rejects_before_gather() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[3.0],
                    shape: &[1, 1],
                })
                .expect("resident Q upload");
            provider.reset_telemetry();
            let strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(dividerand::dividerand_builtin(vec![Value::GpuTensor(
                handle.clone(),
            )]))
            .expect_err("resident argument must reject before gather");
            assert_eq!(
                error.identifier(),
                DIVIDERAND_RESIDENT_ARGUMENT_EXTENSION.error_identifier
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            drop(strict);
            provider.free(&handle).expect("free resident Q");
        });
    }

    #[test]
    fn dividerand_rejects_bad_arguments() {
        let err = block_on(dividerand::dividerand_builtin(vec![
            Value::Num(3.5),
            Value::Num(0.7),
            Value::Num(0.2),
            Value::Num(0.1),
        ]))
        .unwrap_err();
        assert!(err.message().contains("nonnegative integer"));

        let err = block_on(dividerand::dividerand_builtin(vec![
            Value::Num(3.0),
            Value::Num(0.0),
            Value::Num(0.0),
            Value::Num(0.0),
        ]))
        .unwrap_err();
        assert!(err.message().contains("at least one ratio"));

        let err = block_on(dividerand::dividerand_builtin(vec![
            Value::Num(3.0),
            Value::Num(1.0),
        ]))
        .unwrap_err();
        assert!(err.message().contains("expected Q"));
    }

    #[test]
    fn dividerand_handles_extreme_ratios_and_rejects_excessive_q() {
        assert_eq!(
            dividerand_counts(10, [f64::MAX, f64::MAX, f64::MAX]),
            [4, 3, 3]
        );

        let err = block_on(dividerand::dividerand_builtin(vec![Value::Num(
            (MAX_DIVIDERAND_Q + 1) as f64,
        )]))
        .unwrap_err();
        assert!(err.message().contains("maximum supported value"));
    }

    #[test]
    fn bootstrp_descriptor_declares_integer_forms_and_extensions() {
        let builtin = builtin_function_by_name("bootstrp").expect("registered bootstrp");
        assert_eq!(builtin.extensions, &BOOTSTRP_EXTENSIONS);
        assert_eq!(builtin.integer_capabilities.len(), 3);
        assert_eq!(
            builtin
                .integer_capabilities
                .iter()
                .map(|capability| capability.form)
                .collect::<Vec<_>>(),
            BOOTSTRP_INTEGER_CAPABILITIES
                .iter()
                .map(|capability| capability.form)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn bootstrp_extensions_are_independently_mode_gated() {
        let empty_callback =
            || Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty callback"));
        let data = || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("double data"));
        let integer_data = || {
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1])
                    .expect("integer data"),
            )
        };
        let integer_weights = || {
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U8(vec![1, 1]), vec![2, 1])
                    .expect("integer weights"),
            )
        };
        let logical_weights = || {
            Value::LogicalArray(LogicalArray::new(vec![1, 1], vec![2, 1]).expect("logical weights"))
        };

        let cases = [
            (
                vec![Value::Int(IntValue::U16(2)), empty_callback(), data()],
                BOOTSTRP_INTEGER_NBOOT_EXTENSION.error_identifier,
            ),
            (
                vec![Value::Bool(true), empty_callback(), data()],
                BOOTSTRP_LOGICAL_NBOOT_EXTENSION.error_identifier,
            ),
            (
                vec![Value::Num(2.0), empty_callback(), integer_data()],
                BOOTSTRP_INTEGER_DATA_EXTENSION.error_identifier,
            ),
            (
                vec![
                    Value::Num(2.0),
                    empty_callback(),
                    data(),
                    Value::from("Weights"),
                    integer_weights(),
                ],
                BOOTSTRP_INTEGER_WEIGHTS_EXTENSION.error_identifier,
            ),
            (
                vec![
                    Value::Num(2.0),
                    empty_callback(),
                    data(),
                    Value::from("Weights"),
                    logical_weights(),
                ],
                BOOTSTRP_LOGICAL_WEIGHTS_EXTENSION.error_identifier,
            ),
            (
                vec![Value::Num(2.0), Value::from("mean"), data()],
                BOOTSTRP_TEXT_CALLABLE_EXTENSION.error_identifier,
            ),
        ];

        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (args, identifier) in cases {
            let error =
                block_on(bootstrp::bootstrp_builtin(args)).expect_err("strict mode must reject");
            assert_eq!(error.identifier(), identifier);
        }
    }

    #[test]
    fn bootstrp_integer_extensions_preserve_exact_callback_output() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _outputs = crate::output_count::push_output_count(Some(2));
        let wide = 9_007_199_254_740_993_u64;
        let data = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![wide, 7]), vec![2, 1])
                .expect("integer data"),
        );
        let weights = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(vec![1, 0]), vec![2, 1])
                .expect("integer weights"),
        );
        let Value::OutputList(outputs) = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Int(IntValue::U16(2)),
            Value::FunctionHandle("min".to_string()),
            data,
            Value::from("Weights"),
            weights,
        ]))
        .expect("RunMat integer bootstrap") else {
            panic!("expected output list");
        };
        assert!(matches!(
            &outputs[0],
            Value::Tensor(tensor)
                if tensor.shape == vec![2, 1]
                    && tensor.integer_storage() == Some(&IntegerStorage::U64(vec![wide, wide]))
        ));
        assert!(matches!(
            &outputs[1],
            Value::Tensor(tensor)
                if tensor.shape == vec![2, 2]
                    && tensor.materialize_f64() == vec![1.0; 4]
        ));
    }

    #[test]
    fn bootstrp_gpu_extension_rejects_before_gather_and_admits_resident_nboot() {
        use runmat_accelerate_api::HostTensorView;

        crate::builtins::common::test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&HostTensorView {
                    data: &[2.0],
                    shape: &[1, 1],
                })
                .expect("resident nboot");
            let empty_callback =
                || Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty callback"));
            let data =
                || Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("double data"));
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = block_on(bootstrp::bootstrp_builtin(vec![
                    Value::GpuTensor(handle.clone()),
                    empty_callback(),
                    data(),
                ]))
                .expect_err("strict mode rejects resident input");
                assert_eq!(
                    error.identifier(),
                    BOOTSTRP_GPU_INPUT_EXTENSION.error_identifier
                );
            }
            {
                let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
                let Value::Tensor(output) = block_on(bootstrp::bootstrp_builtin(vec![
                    Value::GpuTensor(handle),
                    empty_callback(),
                    data(),
                ]))
                .expect("RunMat resident nboot") else {
                    panic!("expected empty bootstat tensor");
                };
                assert_eq!(output.shape, vec![2, 0]);
            }
        });
    }

    #[test]
    fn bootstrp_assembles_homogeneous_callback_rows_without_class_loss() {
        let numeric_cases = [
            NumericStorage::F64(vec![1.0, 2.0]),
            NumericStorage::F32(vec![1.0, 2.0]),
            NumericStorage::I8(vec![-1, 2]),
            NumericStorage::I16(vec![-1, 2]),
            NumericStorage::I32(vec![-1, 2]),
            NumericStorage::I64(vec![-1, 2]),
            NumericStorage::U8(vec![1, 2]),
            NumericStorage::U16(vec![1, 2]),
            NumericStorage::U32(vec![1, 2]),
            NumericStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];
        for storage in numeric_cases {
            let expected = storage.clone();
            let output = assemble_bootstat(vec![BootstatRow::Numeric(storage)], 1)
                .expect("numeric bootstat");
            let Value::Tensor(tensor) = output else {
                panic!("expected tensor bootstat");
            };
            assert_eq!(tensor.shape, vec![1, 2]);
            assert_eq!(tensor.into_numeric_storage(), Ok(expected));
        }

        let Value::LogicalArray(logical) =
            assemble_bootstat(vec![BootstatRow::Logical(vec![1, 0])], 1).expect("logical bootstat")
        else {
            panic!("expected logical bootstat");
        };
        assert_eq!(logical.shape, vec![1, 2]);
        assert_eq!(logical.data, vec![1, 0]);
    }

    #[test]
    fn bootstrp_weighted_mean_returns_stats_and_samples() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let data = Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(4.0),
            Value::FunctionHandle("mean".to_string()),
            data,
            Value::from("Weights"),
            Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![3, 1]).unwrap()),
        ]))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![4, 1]);
                        assert_eq!(t.materialize_f64(), vec![10.0; 4]);
                    }
                    Value::Num(value) => assert_eq!(*value, 10.0),
                    other => panic!("expected tensor bootstat, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![3, 4]);
                        assert_eq!(t.materialize_f64(), vec![1.0; 12]);
                    }
                    other => panic!("expected tensor bootsam, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn bootstrp_empty_function_returns_indices_without_evaluating() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let data = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(2.0),
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            data,
        ]))
        .unwrap();
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(t) => assert_eq!(t.shape, vec![2, 0]),
                    other => panic!("expected empty tensor bootstat, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![3, 2]);
                        assert!(t
                            .materialize_f64()
                            .iter()
                            .all(|idx| (1.0..=3.0).contains(idx)));
                    }
                    other => panic!("expected tensor bootsam, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn bootstrp_multiple_data_arguments_sample_rows_together() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());
        let y = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![4, 1]).unwrap());
        let out = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(3.0),
            Value::FunctionHandle("corr".to_string()),
            x,
            y,
        ]))
        .unwrap();
        match out {
            Value::Tensor(t) => assert_eq!(t.shape, vec![3, 1]),
            other => panic!("expected tensor bootstat, got {other:?}"),
        }
    }

    #[test]
    fn bootstrp_scalar_data_arguments_are_passed_through() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let out = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(3.0),
            Value::FunctionHandle("mean".to_string()),
            Value::Num(42.0),
        ]))
        .unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![3, 1]);
                assert_eq!(t.materialize_f64(), vec![42.0; 3]);
            }
            other => panic!("expected tensor bootstat, got {other:?}"),
        }
    }

    #[test]
    fn bootstrp_row_vector_sampling_ignores_scalar_passthrough_for_axis_choice() {
        let _lock = random::test_lock().lock().unwrap();
        random::reset_rng();
        let row = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap());
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(2.0),
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            row,
            Value::from("tag"),
        ]))
        .unwrap();
        match out {
            Value::OutputList(values) => match &values[1] {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![4, 2]);
                    assert!(t
                        .materialize_f64()
                        .iter()
                        .all(|idx| (1.0..=4.0).contains(idx)));
                }
                other => panic!("expected tensor bootsam, got {other:?}"),
            },
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn bootstrp_rejects_mismatched_data_and_extra_outputs() {
        let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let y = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let err = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(2.0),
            Value::FunctionHandle("corr".to_string()),
            x,
            y,
        ]))
        .unwrap_err();
        assert!(err.message().contains("same number of rows"));

        let _guard = crate::output_count::push_output_count(Some(3));
        let data = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
        let err = block_on(bootstrp::bootstrp_builtin(vec![
            Value::Num(2.0),
            Value::FunctionHandle("definitely_missing_bootstrap_callback".to_string()),
            data,
        ]))
        .unwrap_err();
        assert!(err.message().contains("too many output"));
    }

    #[test]
    fn sampling_count_parsers_preserve_typed_integers_and_reject_lossy_f64() {
        assert_eq!(
            parse_positive_usize("test", &Value::Int(runmat_builtins::IntValue::U16(3)), "k",)
                .unwrap(),
            3
        );
        assert_eq!(
            parse_nonnegative_usize("test", Value::Int(runmat_builtins::IntValue::U16(0)), "Q",)
                .unwrap(),
            0
        );
        for value in [
            Value::Int(runmat_builtins::IntValue::I8(-1)),
            Value::Num(1.5),
            Value::Num(usize::MAX as f64 + 1.0),
        ] {
            assert!(parse_positive_usize("test", &value, "k").is_err());
        }
        for value in [
            Value::Int(runmat_builtins::IntValue::I8(-1)),
            Value::Num(-0.5),
            Value::Num(usize::MAX as f64 + 1.0),
        ] {
            assert!(parse_nonnegative_usize("test", value, "Q").is_err());
        }
    }

    #[test]
    fn sampling_scalar_parsers_read_typed_integer_tensor_storage_exactly() {
        assert_eq!(
            parse_positive_usize(
                "datasample",
                &poisoned_int_tensor(IntegerStorage::U16(vec![3]), vec![1, 1], f64::NAN),
                "k",
            )
            .unwrap(),
            3
        );
        assert!(parse_positive_usize(
            "datasample",
            &poisoned_int_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1], 3.0),
            "k",
        )
        .is_err());
        assert_eq!(
            parse_nonnegative_usize(
                "dividerand",
                poisoned_int_tensor(IntegerStorage::U16(vec![4]), vec![1, 1], -1.0),
                "Q",
            )
            .unwrap(),
            4
        );
        assert!(parse_nonnegative_usize(
            "dividerand",
            poisoned_int_tensor(IntegerStorage::I16(vec![-1]), vec![1, 1], 4.0),
            "Q",
        )
        .is_err());
        assert_eq!(
            parse_nonnegative_scalar_ratio(
                poisoned_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1], f64::NAN),
                "trainRatio",
            )
            .unwrap(),
            1.0
        );
    }

    #[test]
    fn bootstrp_empty_function_detection_uses_typed_integer_storage_length() {
        let scalar =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("typed scalar");
        assert!(!is_empty_function(&Value::Tensor(scalar)));

        let empty =
            Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 0]).expect("typed empty");
        assert!(is_empty_function(&Value::Tensor(empty)));
    }

    #[test]
    fn sampling_weight_and_bootstat_converters_read_typed_integer_storage_exactly() {
        assert_eq!(
            parse_weights(
                "datasample",
                poisoned_int_tensor(IntegerStorage::U16(vec![0, 2, 3]), vec![3, 1], f64::NAN),
                3,
            )
            .unwrap(),
            vec![0.0, 2.0, 3.0]
        );

        let BootstatRow::Numeric(storage) = block_on(bootstat_row(poisoned_int_tensor(
            IntegerStorage::I16(vec![4, 5]),
            vec![1, 2],
            f64::NAN,
        )))
        .unwrap() else {
            panic!("expected numeric callback row");
        };
        assert_eq!(storage, NumericStorage::I16(vec![4, 5]));
    }
}
