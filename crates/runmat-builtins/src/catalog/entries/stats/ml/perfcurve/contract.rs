use crate::*;

const OUTPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "X-coordinate criterion values for the performance curve.",
};

const OUTPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Y-coordinate criterion values for the performance curve.",
};

const OUTPUT_T: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Score thresholds used to compute the curve.",
};

const OUTPUT_AUC: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "AUC",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Area under the X/Y performance curve.",
};

const OUTPUT_OPTROCPT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "OPTROCPT",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Optimal operating point on the ROC curve.",
};

const OUTPUT_SUBY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "SUBY",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Y-criterion values computed for each negative subclass.",
};

const OUTPUT_SUBYNAMES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "SUBYNAMES",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Negative subclass labels corresponding to columns of SUBY.",
};

const PARAM_LABELS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "labels",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True class labels.",
};

const PARAM_SCORES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "scores",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Classifier scores for the positive class.",
};

const PARAM_POSCLASS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "posclass",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive class label.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as XCrit, YCrit, NegClass, Weights, Prior, Cost, TVals, XVals, UseNearest, and ProcessNaN.",
};

const INPUTS_BASIC: [BuiltinParamDescriptor; 3] = [PARAM_LABELS, PARAM_SCORES, PARAM_POSCLASS];
const INPUTS_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_LABELS, PARAM_SCORES, PARAM_POSCLASS, PARAM_OPTIONS];
const OUTPUTS_X: [BuiltinParamDescriptor; 1] = [OUTPUT_X];
const OUTPUTS_ALL: [BuiltinParamDescriptor; 7] = [
    OUTPUT_X,
    OUTPUT_Y,
    OUTPUT_T,
    OUTPUT_AUC,
    OUTPUT_OPTROCPT,
    OUTPUT_SUBY,
    OUTPUT_SUBYNAMES,
];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "X = perfcurve(labels, scores, posclass)",
        inputs: &INPUTS_BASIC,
        outputs: &OUTPUTS_X,
    },
    BuiltinSignatureDescriptor {
        label: "X = perfcurve(labels, scores, posclass, Name, Value)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUTS_X,
    },
    BuiltinSignatureDescriptor {
        label: "[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(___)",
        inputs: &INPUTS_OPTIONS,
        outputs: &OUTPUTS_ALL,
    },
];

pub const PERFCURVE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERFCURVE.INVALID_ARGUMENT",
    identifier: Some("RunMat:perfcurve:InvalidArgument"),
    when: "Labels, scores, class selections, criteria, thresholds, weights, priors, costs, or name-value options are malformed.",
    message: "perfcurve: invalid argument",
};

pub const PERFCURVE_ERROR_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERFCURVE.UNSUPPORTED",
    identifier: Some("RunMat:perfcurve:Unsupported"),
    when: "A documented advanced mode requires confidence bounds, bootstrap/cross-validation outputs, or score matrix class-column inference that is not implemented yet.",
    message: "perfcurve: unsupported mode",
};

pub const PERFCURVE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERFCURVE.INTERNAL",
    identifier: Some("RunMat:perfcurve:Internal"),
    when: "RunMat cannot allocate or construct perfcurve outputs.",
    message: "perfcurve: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    PERFCURVE_ERROR_INVALID_ARGUMENT,
    PERFCURVE_ERROR_UNSUPPORTED,
    PERFCURVE_ERROR_INTERNAL,
];

pub const PERFCURVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const PERFCURVE_INTEGER_LABELS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "perfcurve-integer-labels",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "perfcurve accepts typed-integer labels and class selections as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PerfcurveIntegerLabelsExtension"),
    };
pub const PERFCURVE_INTEGER_SCORES_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "perfcurve-integer-scores",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "perfcurve accepts typed-integer classifier scores as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PerfcurveIntegerScoresExtension"),
    };
pub const PERFCURVE_INTEGER_NUMERIC_OPTIONS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "perfcurve-integer-numeric-options",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "perfcurve accepts typed-integer numeric option values as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PerfcurveIntegerNumericOptionsExtension"),
    };
pub const PERFCURVE_INTEGER_LOGICAL_CONTROLS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "perfcurve-integer-logical-controls",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "perfcurve accepts typed-integer logical controls as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PerfcurveIntegerLogicalControlsExtension"),
    };
pub const PERFCURVE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    PERFCURVE_INTEGER_LABELS_EXTENSION,
    PERFCURVE_INTEGER_SCORES_EXTENSION,
    PERFCURVE_INTEGER_NUMERIC_OPTIONS_EXTENSION,
    PERFCURVE_INTEGER_LOGICAL_CONTROLS_EXTENSION,
];
const PERFCURVE_INTEGER_LABEL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "labels, posclass, or NegClass",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents floating numeric, logical, text, cellstr, and categorical labels, but not typed-integer labels. RunMat compares native integer labels exactly without routing them through f64.",
    }];
const PERFCURVE_INTEGER_SCORE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "scores",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents floating classifier scores. RunMat accepts typed integers only when every score is exactly representable in the binary64 computation domain.",
    }];
const PERFCURVE_INTEGER_OPTION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Weights, Prior, Cost, TVals, or XVals",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat extends the documented floating numeric option domains with typed integers after proving exact binary64 representability.",
    }];
const PERFCURVE_INTEGER_CONTROL_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "UseNearest",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "RunMat accepts an exact typed-integer zero or one for the logical UseNearest control.",
    }];
pub const PERFCURVE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "perfcurve(integer_labels, scores, integer_posclass, 'NegClass', integer_classes)",
        inputs: &PERFCURVE_INTEGER_LABEL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Labels retain their authoritative integer payload through class selection, deduplication, ordering, and equality; curve coordinates remain floating outputs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "perfcurve(labels, integer_scores, posclass)",
        inputs: &PERFCURVE_INTEGER_SCORE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The extension and exactness gate run before provider lookup or gathering, then scores enter the floating ROC computation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "perfcurve(..., Name, integer_numeric_value)",
        inputs: &PERFCURVE_INTEGER_OPTION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Weights, priors, costs, and requested threshold or X-coordinate values cross an explicit checked binary64 boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "perfcurve(..., 'UseNearest', integer_zero_or_one)",
        inputs: &PERFCURVE_INTEGER_CONTROL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The native integer is interpreted structurally and must be exactly zero or one.",
    },
];
