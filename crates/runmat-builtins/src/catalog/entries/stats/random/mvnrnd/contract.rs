//! Typed declarations conserved from the Runtime owner.
use crate::*;

pub const INTEGER_MU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mvnrnd-integer-mu",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mvnrnd with a typed-integer mean array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MvnrndIntegerMuExtension"),
};

pub const INTEGER_SIGMA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mvnrnd-integer-sigma",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mvnrnd with a typed-integer covariance array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MvnrndIntegerSigmaExtension"),
};

pub const INTEGER_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mvnrnd-integer-count",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mvnrnd with a typed-integer sample count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MvnrndIntegerCountExtension"),
};

pub const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mvnrnd-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mvnrnd with logical numeric inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MvnrndLogicalInputExtension"),
};

pub const RESIDENT_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mvnrnd-resident-count",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mvnrnd with an explicit gpuArray sample count is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MvnrndResidentCountExtension"),
};

pub const EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    INTEGER_MU_EXTENSION,
    INTEGER_SIGMA_EXTENSION,
    INTEGER_N_EXTENSION,
    LOGICAL_INPUT_EXTENSION,
    RESIDENT_N_EXTENSION,
];

pub const INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "mu",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer means are independently gated and require exact binary64 representation.",
    },
    BuiltinIntegerInputCapability {
        name: "Sigma",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer covariance values are independently gated and require exact binary64 representation.",
    },
];

pub const INTEGER_N_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "n",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Current MATLAB documents single and double n; native integer n is a gated RunMat structural extension.",
}];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "r = mvnrnd(integer_mu, integer_Sigma, ___)",
        inputs: &INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer distribution arrays are RunMat-only, cross one checked binary64 boundary, and return double unless another documented data input is single; fallback restores through the exact owner when it can preserve the required class, otherwise automatic residency may remain host and explicit residency errors.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = mvnrnd(mu, Sigma, integer_n)",
        inputs: &INTEGER_N_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer n is parsed exactly without selecting output class or residency; explicit resident n is separately gated.",
    },
];

pub const OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random samples with observations in rows.",
}];

pub const INPUT_MU: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mu",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mean vector or matrix with one mean vector per row.",
};

pub const INPUT_SIGMA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Sigma",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Covariance matrix or covariance matrix pages.",
};

pub const INPUT_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Number of random rows to draw when mu is a single mean vector.",
};

pub const INPUTS_MU_SIGMA: [BuiltinParamDescriptor; 2] = [INPUT_MU, INPUT_SIGMA];

pub const INPUTS_MU_SIGMA_N: [BuiltinParamDescriptor; 3] = [INPUT_MU, INPUT_SIGMA, INPUT_N];

pub const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "r = mvnrnd(mu, Sigma)",
        inputs: &INPUTS_MU_SIGMA,
        outputs: &OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = mvnrnd(mu, Sigma, n)",
        inputs: &INPUTS_MU_SIGMA_N,
        outputs: &OUTPUT_R,
    },
];

pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MVNRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:mvnrnd:InvalidArgument"),
    when: "Inputs are malformed, dimensions are incompatible, or covariance pages are not symmetric positive semidefinite.",
    message: "mvnrnd: invalid argument",
};

pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MVNRND.INTERNAL",
    identifier: Some("RunMat:mvnrnd:Internal"),
    when: "RunMat cannot allocate or construct the requested random samples.",
    message: "mvnrnd: internal operation failed",
};

pub const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_ARGUMENT, ERROR_INTERNAL];

pub const MVNRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
