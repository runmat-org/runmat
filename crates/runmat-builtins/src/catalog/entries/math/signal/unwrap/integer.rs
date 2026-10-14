use crate::*;

pub const UNWRAP_INTEGER_PHASE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "unwrap-integer-phase-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "unwrap with typed-integer phase data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UnwrapIntegerPhaseDataExtension"),
};
pub const UNWRAP_LOGICAL_PHASE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "unwrap-logical-phase-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "unwrap with logical phase data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UnwrapLogicalPhaseDataExtension"),
};
pub const UNWRAP_INTEGER_TOLERANCE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "unwrap-integer-tolerance",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "unwrap with a typed-integer tolerance is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:UnwrapIntegerToleranceExtension"),
    };
pub const UNWRAP_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    UNWRAP_INTEGER_PHASE_EXTENSION,
    UNWRAP_LOGICAL_PHASE_EXTENSION,
    UNWRAP_INTEGER_TOLERANCE_EXTENSION,
];

const PHASE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "P",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed phase data cross a checked exact binary64 boundary before arithmetic.",
}];
const TOLERANCE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "tol",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed tolerance crosses a checked exact binary64 comparison boundary.",
}];
const DIMENSION: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "dim",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Dimension is decoded exactly as a structural index.",
}];
pub const UNWRAP_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = unwrap(integer_P, tol?, dim?)",
        inputs: &PHASE,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Phase values must be exactly representable before unwrapping; output is double.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = unwrap(P, integer_tol, dim?)",
        inputs: &TOLERANCE,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes:
            "The tolerance is independently gated without changing the output class selected by P.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Q = unwrap(P, tol, integer_dim)",
        inputs: &DIMENSION,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The dimension stays exact through validation and does not enter phase arithmetic.",
    },
];
