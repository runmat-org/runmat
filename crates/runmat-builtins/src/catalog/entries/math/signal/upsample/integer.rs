use crate::*;
pub const UPSAMPLE_INTEGER_FACTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "upsample-integer-factor",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "upsample with a typed-integer factor is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:UpsampleIntegerFactorExtension"),
    };
pub const UPSAMPLE_INTEGER_PHASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "upsample-integer-phase",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "upsample with a typed-integer phase is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:UpsampleIntegerPhaseExtension"),
    };
pub const UPSAMPLE_ND_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "upsample-nd-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "upsample with an input having more than two dimensions is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UpsampleNdInputExtension"),
};
pub const UPSAMPLE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    UPSAMPLE_INTEGER_FACTOR_EXTENSION,
    UPSAMPLE_INTEGER_PHASE_EXTENSION,
    UPSAMPLE_ND_INPUT_EXTENSION,
];
const DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Zero insertion preserves every signed and unsigned integer class exactly.",
}];
const FACTOR: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "N",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "RunMat mode parses typed factors exactly as host sizes.",
}];
const PHASE: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "PHASE",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "RunMat mode parses typed phases exactly before range validation.",
}];
pub const UPSAMPLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = upsample(integer_X,N,PHASE?)",
        inputs: &DATA,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Host and owner-resolved resident scatter preserve authoritative storage and class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = upsample(X,integer_N,PHASE?)",
        inputs: &FACTOR,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The factor is gated before X/provider access and converted exactly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = upsample(X,N,integer_PHASE)",
        inputs: &PHASE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The phase is gated before X/provider access and checked against N.",
    },
];
