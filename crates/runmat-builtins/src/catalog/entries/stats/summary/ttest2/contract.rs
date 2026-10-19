//! Canonical descriptor, extension, and integer contracts for ttest2.
use crate::*;

pub const TTEST2_INTEGER_SAMPLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ttest2-integer-sample",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ttest2 with native-class integer sample data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Ttest2IntegerSampleExtension"),
    };
pub const TTEST2_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ttest2-integer-option",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ttest2 with native-class integer numeric options is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Ttest2IntegerOptionExtension"),
    };
pub const TTEST2_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    TTEST2_INTEGER_SAMPLE_EXTENSION,
    TTEST2_INTEGER_OPTION_EXTENSION,
];

const TTEST2_INTEGER_SAMPLE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer samples are gated before gather and checked for exact binary64 representation.",
    },
    BuiltinIntegerInputCapability {
        name: "y",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer samples are gated before gather and checked for exact binary64 representation.",
    },
];
const TTEST2_INTEGER_OPTION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Alpha or Dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer numeric options are independently gated and checked before scalar conversion.",
    }];
pub const TTEST2_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = ttest2(integer_x, integer_y, ___)",
        inputs: &TTEST2_INTEGER_SAMPLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "h is logical and the remaining outputs are floating point. Lossy integer sample conversion rejects.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = ttest2(x, y, Alpha or Dim, integer_value)",
        inputs: &TTEST2_INTEGER_OPTION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Dim is decoded as a checked structural selector; Alpha crosses one checked binary64 boundary.",
    },
];

pub const OUTPUT_H: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hypothesis test decision.",
};

pub const OUTPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "P-value.",
};

const OUTPUT_CI: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ci",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Confidence interval for the difference in means.",
};

const OUTPUT_STATS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "stats",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Structure containing tstat, df, and sd fields.",
};

const OUTPUT_H_ONLY: [BuiltinParamDescriptor; 1] = [OUTPUT_H];
const OUTPUT_H_P: [BuiltinParamDescriptor; 2] = [OUTPUT_H, OUTPUT_P];
const OUTPUT_H_P_CI: [BuiltinParamDescriptor; 3] = [OUTPUT_H, OUTPUT_P, OUTPUT_CI];
const OUTPUT_FULL: [BuiltinParamDescriptor; 4] = [OUTPUT_H, OUTPUT_P, OUTPUT_CI, OUTPUT_STATS];

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "First sample data.",
};

const PARAM_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second sample data.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Alpha, Dim, Tail, and Vartype options.",
};

const INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [PARAM_X, PARAM_Y];
const INPUTS_X_Y_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_X, PARAM_Y, PARAM_OPTIONS];

const TTEST2_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = ttest2(x, y)",
        inputs: &INPUTS_X_Y,
        outputs: &OUTPUT_H_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "h = ttest2(x, y, Name, Value)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_H_ONLY,
    },
    BuiltinSignatureDescriptor {
        label: "[h, p] = ttest2(___)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_H_P,
    },
    BuiltinSignatureDescriptor {
        label: "[h, p, ci] = ttest2(___)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_H_P_CI,
    },
    BuiltinSignatureDescriptor {
        label: "[h, p, ci, stats] = ttest2(___)",
        inputs: &INPUTS_X_Y_OPTIONS,
        outputs: &OUTPUT_FULL,
    },
];

pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TTEST2.INVALID_ARGUMENT",
    identifier: Some("RunMat:ttest2:InvalidArgument"),
    when: "Inputs, dimensions, tails, variance type, or significance level are malformed.",
    message: "ttest2: invalid argument",
};

pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TTEST2.INTERNAL",
    identifier: Some("RunMat:ttest2:Internal"),
    when: "RunMat cannot construct ttest2 outputs.",
    message: "ttest2: internal error",
};

const TTEST2_ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const TTEST2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TTEST2_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TTEST2_ERRORS,
};
