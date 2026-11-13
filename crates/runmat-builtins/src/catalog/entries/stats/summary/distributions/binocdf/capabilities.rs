use crate::*;

pub const BINOCDF_INTEGER_X_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "binocdf-integer-x",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "binocdf with typed-integer evaluation points is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BinocdfIntegerXExtension"),
};

pub const BINOCDF_INTEGER_N_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "binocdf-integer-trials",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "binocdf with typed-integer trial counts is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BinocdfIntegerTrialsExtension"),
};

pub const BINOCDF_INTEGER_P_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "binocdf-integer-probability",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "binocdf with typed-integer probabilities is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:BinocdfIntegerProbabilityExtension"),
};

pub const BINOCDF_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "binocdf-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "binocdf with logical numeric inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BinocdfLogicalInputExtension"),
    };

pub const BINOCDF_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    BINOCDF_INTEGER_X_EXTENSION,
    BINOCDF_INTEGER_N_EXTENSION,
    BINOCDF_INTEGER_P_EXTENSION,
    BINOCDF_LOGICAL_INPUT_EXTENSION,
];

const INTEGER_X_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer evaluation points are gated by binocdf-integer-x and must be exactly representable at the floating binomial-CDF boundary.",
    }];

const INTEGER_N_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer trial counts are gated by binocdf-integer-trials and must be exactly representable at the documented floating computation boundary.",
    }];

const INTEGER_P_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed-integer probabilities are gated by binocdf-integer-probability; only values in the ordinary probability domain can produce non-NaN results.",
    }];

pub const BINOCDF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "y = binocdf(x, n, p) with integer x",
        inputs: &INTEGER_X_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only typed-integer x values produce double probabilities unless a documented single input makes the output single; resident inputs use host fallback and the result is re-uploaded to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = binocdf(x, n, p) with integer n",
        inputs: &INTEGER_N_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only typed-integer n values produce double probabilities unless a documented single input makes the output single, after an exact-representability check at the binary64 boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "y = binocdf(x, n, p) with integer p",
        inputs: &INTEGER_P_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only typed-integer p values produce double probabilities unless a documented single input makes the output single; resident inputs use host fallback and preserve output residency.",
    },
];
