use crate::*;

pub const CVPARTITION_INTEGER_N_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-integer-observation-count",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cvpartition with a typed-integer observation count is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionIntegerObservationCountExtension"),
    };
pub const CVPARTITION_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-integer-partition-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "cvpartition with typed-integer KFold or Holdout control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionIntegerPartitionControlExtension"),
    };
pub const CVPARTITION_INTEGER_STRATIFICATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-integer-stratification",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cvpartition with typed-integer stratification labels is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionIntegerStratificationExtension"),
    };
pub const CVPARTITION_INTEGER_CUSTOM_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-integer-custom-testsets",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cvpartition with typed-integer custom test sets is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionIntegerCustomTestsetsExtension"),
    };
pub const CVPARTITION_BOOLEAN_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-nonlogical-stratify-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cvpartition with a nonlogical Stratify option is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionNonlogicalStratifyOptionExtension"),
    };
pub const CVPARTITION_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    CVPARTITION_INTEGER_N_EXTENSION,
    CVPARTITION_INTEGER_CONTROL_EXTENSION,
    CVPARTITION_INTEGER_STRATIFICATION_EXTENSION,
    CVPARTITION_INTEGER_CUSTOM_EXTENSION,
    CVPARTITION_BOOLEAN_ALIAS_EXTENSION,
];

pub const TEST_INTEGER_INDEX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cvpartition-test-integer-index",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "test with typed-integer repetition indices is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CvpartitionTestIntegerIndexExtension"),
};
pub const TEST_LOGICAL_INDEX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cvpartition-test-logical-index",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "test with logical repetition indices is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CvpartitionTestLogicalIndexExtension"),
};
pub const TRAINING_INTEGER_INDEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-training-integer-index",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "training with typed-integer repetition indices is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionTrainingIntegerIndexExtension"),
    };
pub const TRAINING_LOGICAL_INDEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cvpartition-training-logical-index",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "training with logical repetition indices is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CvpartitionTrainingLogicalIndexExtension"),
    };
pub const TEST_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [TEST_INTEGER_INDEX_EXTENSION, TEST_LOGICAL_INDEX_EXTENSION];
pub const TRAINING_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TRAINING_INTEGER_INDEX_EXTENSION,
    TRAINING_LOGICAL_INDEX_EXTENSION,
];

const MASK_INTEGER_INDEX_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "i",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Public repetition indices are single or double. RunMat mode additionally decodes all eight integer classes exactly as one-based structural indices.",
    }];
pub const TEST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "idx = test(c, integer_i)",
        inputs: &MASK_INTEGER_INDEX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed repetition indices are gated before provider access, decoded without binary64 conversion, and select logical output columns.",
    }];
pub const TRAINING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "idx = training(c, integer_i)",
        inputs: &MASK_INTEGER_INDEX_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The adjacent training method shares test's exact typed-index policy and returns logical mask columns.",
    }];

const INTEGER_N_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "n", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The public observation count lists single and double; all eight typed integer scalar classes are independently gated and read exactly." }];
const INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "kOrP", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Typed KFold and integer-count Holdout controls are independently gated; documented single/double integer-valued controls remain accepted." }];
const INTEGER_STRAT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "stratvar", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "The public numeric stratification class list is single/double; RunMat mode additionally preserves all eight integer label classes through structural grouping." }];
const INTEGER_CUSTOM_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "testSets", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "The public custom-test-set class list is single/double/logical; typed integer assignment vectors or 0/1 masks are independently gated." }];
pub const CVPARTITION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "c = cvpartition(integer_n, ___)", inputs: &INTEGER_N_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Typed n determines object and mask sizes only; public count properties remain double and mask properties remain logical." },
    BuiltinIntegerCapabilityDescriptor { form: "c = cvpartition(___, integer_kOrP)", inputs: &INTEGER_CONTROL_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Typed partition controls are validated exactly before random assignment and do not determine property classes." },
    BuiltinIntegerCapabilityDescriptor { form: "c = cvpartition(integer_stratvar, ___)", inputs: &INTEGER_STRAT_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer labels are compared structurally from authoritative storage and never treated as floating observations." },
    BuiltinIntegerCapabilityDescriptor { form: "c = cvpartition(CustomPartition=integer_testSets)", inputs: &INTEGER_CUSTOM_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer test-set identifiers and 0/1 masks are interpreted exactly before logical mask construction." },
];

const OUTPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cross-validation partition object.",
}];

const OUTPUT_MASK: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "mask",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical observation mask for one or more test sets.",
}];

const PARAM_N_OR_STRATVAR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nOrStratvar",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Observation count or stratification variable.",
};

const PARAM_KIND: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "kind",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Partition kind such as KFold, Holdout, Leaveout, Resubstitution, or CustomPartition.",
};

const PARAM_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Partition parameter such as fold count, holdout fraction, holdout count, or custom test sets.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options including Stratify.",
};

const PARAM_C: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "c",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "cvpartition object.",
};

const PARAM_INDEX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "i",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "One-based test-set index, vector of indices, or 'all'.",
};

const CVPARTITION_INPUTS: [BuiltinParamDescriptor; 3] =
    [PARAM_N_OR_STRATVAR, PARAM_KIND, PARAM_VALUE];
const CVPARTITION_INPUTS_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_N_OR_STRATVAR, PARAM_KIND, PARAM_VALUE, PARAM_OPTIONS];
const MASK_INPUTS_C: [BuiltinParamDescriptor; 1] = [PARAM_C];
const MASK_INPUTS_C_I: [BuiltinParamDescriptor; 2] = [PARAM_C, PARAM_INDEX];

const CVPARTITION_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "c = cvpartition(n, 'KFold', k)",
        inputs: &CVPARTITION_INPUTS,
        outputs: &OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "c = cvpartition(stratvar, 'KFold', k, 'Stratify', tf)",
        inputs: &CVPARTITION_INPUTS_OPTIONS,
        outputs: &OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "c = cvpartition(n, 'Holdout', p)",
        inputs: &CVPARTITION_INPUTS,
        outputs: &OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "c = cvpartition(n, 'Leaveout')",
        inputs: &[PARAM_N_OR_STRATVAR, PARAM_KIND],
        outputs: &OUTPUT_C,
    },
    BuiltinSignatureDescriptor {
        label: "c = cvpartition('CustomPartition', testSets)",
        inputs: &[PARAM_KIND, PARAM_VALUE],
        outputs: &OUTPUT_C,
    },
];

const MASK_ALL_INPUTS: [BuiltinParamDescriptor; 2] = [PARAM_C, PARAM_INDEX];

const MASK_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "mask = training(c)",
        inputs: &MASK_INPUTS_C,
        outputs: &OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "mask = training(c, i)",
        inputs: &MASK_INPUTS_C_I,
        outputs: &OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "mask = training(c, \"all\")",
        inputs: &MASK_ALL_INPUTS,
        outputs: &OUTPUT_MASK,
    },
];

const TEST_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "mask = test(c)",
        inputs: &MASK_INPUTS_C,
        outputs: &OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "mask = test(c, i)",
        inputs: &MASK_INPUTS_C_I,
        outputs: &OUTPUT_MASK,
    },
    BuiltinSignatureDescriptor {
        label: "mask = test(c, \"all\")",
        inputs: &MASK_ALL_INPUTS,
        outputs: &OUTPUT_MASK,
    },
];

pub const CVPARTITION_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CVPARTITION.INVALID_ARGUMENT",
    identifier: Some("RunMat:cvpartition:InvalidArgument"),
    when: "Inputs, partition kinds, dimensions, indices, or name-value options are malformed.",
    message: "cvpartition: invalid argument",
};

pub const CVPARTITION_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CVPARTITION.INTERNAL",
    identifier: Some("RunMat:cvpartition:Internal"),
    when: "RunMat cannot allocate or construct a partition output.",
    message: "cvpartition: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [
    CVPARTITION_ERROR_INVALID_ARGUMENT,
    CVPARTITION_ERROR_INTERNAL,
];

pub const CVPARTITION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CVPARTITION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TRAINING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MASK_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TEST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TEST_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
