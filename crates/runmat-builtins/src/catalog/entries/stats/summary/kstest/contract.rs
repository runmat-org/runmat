//! Canonical descriptor, extension, and integer contracts for kstest.
use crate::*;
const OUTPUT_H: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Hypothesis test decision.",
};
const OUTPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "P-value.",
};

pub const KSTEST_INTEGER_SAMPLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "kstest-integer-sample",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "kstest with a native-class integer sample is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KstestIntegerSampleExtension"),
    };
pub const KSTEST_INTEGER_CDF_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "kstest-integer-cdf",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "kstest with a native-class integer CDF table is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:KstestIntegerCdfExtension"),
};
pub const KSTEST_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "kstest-resident-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "kstest host fallback for explicit gpuArray inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:KstestResidentInputExtension"),
    };
pub const KSTEST_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    KSTEST_INTEGER_SAMPLE_EXTENSION,
    KSTEST_INTEGER_CDF_EXTENSION,
    KSTEST_RESIDENT_INPUT_EXTENSION,
];

const KSTEST_INTEGER_SAMPLE_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "x", classes: &crate::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Native integer samples are gated before gather and checked for exact binary64 representation." }];
const KSTEST_INTEGER_CDF_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "CDF table",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer CDF tables are gated and checked before interpolation.",
    }];
pub const KSTEST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "h = kstest(integer_x, ___)", inputs: &KSTEST_INTEGER_SAMPLE_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "h is logical; p, ksstat, and cv are double. Lossy integer conversion rejects." },
    BuiltinIntegerCapabilityDescriptor { form: "h = kstest(x, CDF=integer_table)", inputs: &KSTEST_INTEGER_CDF_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer CDF tables are a checked RunMat extension; automatic residency gathers transparently, while explicit gpuArray fallback remains separately gated." },
];

const KSTEST_PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample data.",
};

const KSTEST_PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Alpha, CDF, and Tail options.",
};

const OUTPUT_KSSTAT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ksstat",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Kolmogorov-Smirnov test statistic.",
};

const OUTPUT_CV: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "cv",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Approximate critical value.",
};

const KSTEST_INPUTS_X: [BuiltinParamDescriptor; 1] = [KSTEST_PARAM_X];
const KSTEST_INPUTS_OPTIONS: [BuiltinParamDescriptor; 2] = [KSTEST_PARAM_X, KSTEST_PARAM_OPTIONS];
const KSTEST_OUTPUT_H: [BuiltinParamDescriptor; 1] = [OUTPUT_H];
const KSTEST_OUTPUT_H_P: [BuiltinParamDescriptor; 2] = [OUTPUT_H, OUTPUT_P];
const KSTEST_OUTPUT_H_P_KSSTAT: [BuiltinParamDescriptor; 3] = [OUTPUT_H, OUTPUT_P, OUTPUT_KSSTAT];
const KSTEST_OUTPUT_FULL: [BuiltinParamDescriptor; 4] =
    [OUTPUT_H, OUTPUT_P, OUTPUT_KSSTAT, OUTPUT_CV];

const KSTEST_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "h = kstest(x)",
        inputs: &KSTEST_INPUTS_X,
        outputs: &KSTEST_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "h = kstest(x, Name, Value)",
        inputs: &KSTEST_INPUTS_OPTIONS,
        outputs: &KSTEST_OUTPUT_H,
    },
    BuiltinSignatureDescriptor {
        label: "[h, p] = kstest(___)",
        inputs: &KSTEST_INPUTS_OPTIONS,
        outputs: &KSTEST_OUTPUT_H_P,
    },
    BuiltinSignatureDescriptor {
        label: "[h, p, ksstat] = kstest(___)",
        inputs: &KSTEST_INPUTS_OPTIONS,
        outputs: &KSTEST_OUTPUT_H_P_KSSTAT,
    },
    BuiltinSignatureDescriptor {
        label: "[h, p, ksstat, cv] = kstest(___)",
        inputs: &KSTEST_INPUTS_OPTIONS,
        outputs: &KSTEST_OUTPUT_FULL,
    },
];

pub const KSTEST_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KSTEST.INVALID_ARGUMENT",
    identifier: Some("RunMat:kstest:InvalidArgument"),
    when: "Inputs, CDF specification, tail, or significance level are malformed.",
    message: "kstest: invalid argument",
};

pub const KSTEST_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.KSTEST.INTERNAL",
    identifier: Some("RunMat:kstest:Internal"),
    when: "RunMat cannot construct kstest outputs.",
    message: "kstest: internal error",
};

const KSTEST_ERRORS: [BuiltinErrorDescriptor; 2] =
    [KSTEST_ERROR_INVALID_ARGUMENT, KSTEST_ERROR_INTERNAL];

pub const KSTEST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &KSTEST_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &KSTEST_ERRORS,
};
