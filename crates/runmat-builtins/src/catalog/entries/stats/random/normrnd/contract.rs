//! Typed declarations conserved from the Runtime owner.
use crate::*;

pub const INTEGER_MU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normrnd-integer-mu",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normrnd with a typed-integer mean parameter is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormrndIntegerMuExtension"),
};

pub const INTEGER_SIGMA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normrnd-integer-sigma",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normrnd with a typed-integer standard deviation is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormrndIntegerSigmaExtension"),
};

pub const INTEGER_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normrnd-integer-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normrnd with typed-integer size arguments is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormrndIntegerSizeExtension"),
};

pub const LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normrnd-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normrnd with logical numeric inputs is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormrndLogicalInputExtension"),
};

pub const RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "normrnd-resident-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "normrnd with explicit gpuArray size controls is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:NormrndResidentSizeExtension"),
};

pub const EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    INTEGER_MU_EXTENSION,
    INTEGER_SIGMA_EXTENSION,
    INTEGER_SIZE_EXTENSION,
    LOGICAL_INPUT_EXTENSION,
    RESIDENT_SIZE_EXTENSION,
];

pub const INTEGER_PARAMETER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "mu",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer means are independently gated and cross one exact binary64 boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "sigma",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer standard deviations are independently gated and cross one exact binary64 boundary.",
    },
];

pub const INTEGER_SIZE_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "sz, sz1, ...",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Current MATLAB documents single and double size arguments; native integer size controls are a gated RunMat extension parsed structurally.",
}];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "r = normrnd(integer_mu, integer_sigma, ___)",
        inputs: &INTEGER_PARAMETER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer distribution parameters are RunMat-only; output is double unless another documented data parameter is single. Fallback restores through the exact owner when it can preserve the required class, otherwise automatic residency may remain host and explicit residency errors.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "r = normrnd(mu, sigma, integer_sz)",
        inputs: &INTEGER_SIZE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer extents are parsed exactly without selecting output class or residency; explicit resident size controls are separately gated.",
    },
];

pub const NORMRND_OUTPUT_R: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "r",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Random sample array from normal distribution.",
}];

pub const NORMRND_INPUTS_MU_SIGMA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean parameter.",
    },
    BuiltinParamDescriptor {
        name: "sigma",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard deviation parameter (must be >= 0).",
    },
];

pub const NORMRND_INPUTS_MU_SIGMA_SZ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean parameter.",
    },
    BuiltinParamDescriptor {
        name: "sigma",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard deviation parameter (must be >= 0).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Size scalar or size vector argument.",
    },
];

pub const NORMRND_INPUTS_MU_SIGMA_DIMS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Mean parameter.",
    },
    BuiltinParamDescriptor {
        name: "sigma",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard deviation parameter (must be >= 0).",
    },
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension extents for output shape.",
    },
];

pub const NORMRND_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "r = normrnd(mu, sigma)",
        inputs: &NORMRND_INPUTS_MU_SIGMA,
        outputs: &NORMRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = normrnd(mu, sigma, sz)",
        inputs: &NORMRND_INPUTS_MU_SIGMA_SZ,
        outputs: &NORMRND_OUTPUT_R,
    },
    BuiltinSignatureDescriptor {
        label: "r = normrnd(mu, sigma, sz1, sz2, ...)",
        inputs: &NORMRND_INPUTS_MU_SIGMA_DIMS,
        outputs: &NORMRND_OUTPUT_R,
    },
];

pub const NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.NORMRND.SIGMA_MUST_BE_NONNEGATIVE",
        identifier: Some("RunMat:normrnd:SigmaMustBeNonnegative"),
        when: "sigma is negative.",
        message: "normrnd: sigma must be non-negative",
    };

pub const NORMRND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMRND.INVALID_ARGUMENT",
    identifier: Some("RunMat:normrnd:InvalidArgument"),
    when: "Input parameters or size arguments are missing or malformed.",
    message: "normrnd: invalid argument",
};

pub const NORMRND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMRND.INTERNAL",
    identifier: Some("RunMat:normrnd:Internal"),
    when: "Internal conversion/allocation/provider decode fails.",
    message: "normrnd: internal operation failed",
};

pub const NORMRND_ERRORS: [BuiltinErrorDescriptor; 3] = [
    NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE,
    NORMRND_ERROR_INVALID_ARGUMENT,
    NORMRND_ERROR_INTERNAL,
];

pub const NORMRND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NORMRND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NORMRND_ERRORS,
};
