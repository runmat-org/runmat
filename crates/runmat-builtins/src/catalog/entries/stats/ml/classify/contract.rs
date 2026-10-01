use crate::*;

const OUTPUT_CLASS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "class",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Predicted class labels for sample observations.",
};

const OUTPUT_ERR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "err",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Apparent training-set error rate.",
};

const OUTPUT_POSTERIOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "posterior",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Posterior class probabilities for sample observations.",
};

const OUTPUT_LOGP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "logp",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Log unconditional density for sample observations.",
};

const OUTPUT_COEFF: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "coeff",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Pairwise discriminant boundary coefficients.",
};

const PARAM_SAMPLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sample",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample observations in rows.",
};

const PARAM_TRAINING: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "training",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Training observations in rows.",
};

const PARAM_GROUP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "group",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Training group labels.",
};

const PARAM_TYPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "type",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("linear"),
    description: "Discriminant type: linear, quadratic, diagLinear, diagQuadratic, or mahalanobis.",
};

const PARAM_PRIOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "prior",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Prior probabilities as a numeric vector, 'empirical', or a struct with group and prob fields.",
};

const INPUTS_BASIC: [BuiltinParamDescriptor; 3] = [PARAM_SAMPLE, PARAM_TRAINING, PARAM_GROUP];
const INPUTS_FULL: [BuiltinParamDescriptor; 5] = [
    PARAM_SAMPLE,
    PARAM_TRAINING,
    PARAM_GROUP,
    PARAM_TYPE,
    PARAM_PRIOR,
];
const OUTPUT_CLASS_ONLY: [BuiltinParamDescriptor; 1] = [OUTPUT_CLASS];
const OUTPUT_ALL: [BuiltinParamDescriptor; 5] = [
    OUTPUT_CLASS,
    OUTPUT_ERR,
    OUTPUT_POSTERIOR,
    OUTPUT_LOGP,
    OUTPUT_COEFF,
];

const CLASSIFY_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "class = classify(sample, training, group)",
        inputs: &INPUTS_BASIC,
        outputs: &OUTPUT_CLASS_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "class = classify(sample, training, group, type, prior)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_CLASS_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "[class,err,posterior,logp,coeff] = classify(___)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_ALL,
    },
];

pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLASSIFY.INVALID_ARGUMENT",
    identifier: Some("RunMat:classify:InvalidArgument"),
    when: "Inputs, labels, dimensions, discriminant type, or priors are malformed.",
    message: "classify: invalid argument",
};

pub const ERROR_NUMERICAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLASSIFY.NUMERICAL",
    identifier: Some("RunMat:classify:Numerical"),
    when: "Covariance matrices cannot be solved numerically.",
    message: "classify: numerical failure",
};

pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CLASSIFY.INTERNAL",
    identifier: Some("RunMat:classify:Internal"),
    when: "RunMat cannot construct classification outputs.",
    message: "classify: internal error",
};

const CLASSIFY_ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_NUMERICAL, ERROR_INTERNAL];

pub const CLASSIFY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLASSIFY_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLASSIFY_ERRORS,
};

pub const CLASSIFY_INTEGER_SAMPLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "classify-integer-sample",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "classify with typed-integer sample observations is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ClassifyIntegerSampleExtension"),
    };
pub const CLASSIFY_INTEGER_TRAINING_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "classify-integer-training",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "classify with typed-integer training observations is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ClassifyIntegerTrainingExtension"),
    };
pub const CLASSIFY_INTEGER_GROUP_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "classify-integer-group",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "classify with typed-integer group labels is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ClassifyIntegerGroupExtension"),
    };
pub const CLASSIFY_INTEGER_PRIOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "classify-integer-prior",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "classify with typed-integer prior probabilities is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ClassifyIntegerPriorExtension"),
    };
pub const CLASSIFY_LOGICAL_PREDICTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "classify-logical-predictor",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "classify with logical sample or training observations is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ClassifyLogicalPredictorExtension"),
    };
pub const CLASSIFY_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "classify-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "classify with GPU-resident inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ClassifyResidentInputExtension"),
    };

pub const CLASSIFY_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    CLASSIFY_INTEGER_SAMPLE_EXTENSION,
    CLASSIFY_INTEGER_TRAINING_EXTENSION,
    CLASSIFY_INTEGER_GROUP_EXTENSION,
    CLASSIFY_INTEGER_PRIOR_EXTENSION,
    CLASSIFY_LOGICAL_PREDICTOR_EXTENSION,
    CLASSIFY_RESIDENT_INPUT_EXTENSION,
];

const CLASSIFY_INTEGER_SAMPLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sample",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented predictor storage is single or double. RunMat mode accepts all eight integer classes at an exact binary64 statistical boundary.",
    }];
const CLASSIFY_INTEGER_TRAINING_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "training",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Documented predictor storage is single or double. RunMat mode accepts all eight integer classes at an exact binary64 statistical boundary.",
    }];
const CLASSIFY_INTEGER_GROUP_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "group",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat preserves exact integer label identity, ordering, and predicted output class; the documented numeric group-label classes are single and double.",
    }];
const CLASSIFY_INTEGER_PRIOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "prior",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat mode accepts typed-integer prior probability vectors, including the prob field of structured priors, after exact binary64-boundary validation.",
    }];

pub const CLASSIFY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "class = classify(integer_sample,training,group,...)",
        inputs: &CLASSIFY_INTEGER_SAMPLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer sample observations enter the binary64 discriminant computation after compatibility and exactness checks.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "class = classify(sample,integer_training,group,...)",
        inputs: &CLASSIFY_INTEGER_TRAINING_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer training observations enter the binary64 discriminant computation after compatibility and exactness checks.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "class = classify(sample,training,integer_group,...)",
        inputs: &CLASSIFY_INTEGER_GROUP_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer group labels remain authoritative exact scalars and predicted labels reconstruct in the same integer class, including values above flintmax.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "class = classify(sample,training,group,type,integer_prior)",
        inputs: &CLASSIFY_INTEGER_PRIOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer prior probabilities enter normalization only after compatibility and exact binary64-boundary checks.",
    },
];
