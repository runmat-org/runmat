use crate::*;
pub const DOWNSAMPLE_INTEGER_FACTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "downsample-integer-factor",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "downsample with a typed-integer factor is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DownsampleIntegerFactorExtension"),
    };
pub const DOWNSAMPLE_INTEGER_PHASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "downsample-integer-phase",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "downsample with a typed-integer phase is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DownsampleIntegerPhaseExtension"),
    };
pub const DOWNSAMPLE_ND_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "downsample-nd-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "downsample with an input having more than two dimensions is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DownsampleNdInputExtension"),
};
pub const DOWNSAMPLE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    DOWNSAMPLE_INTEGER_FACTOR_EXTENSION,
    DOWNSAMPLE_INTEGER_PHASE_EXTENSION,
    DOWNSAMPLE_ND_INPUT_EXTENSION,
];
const DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Structural selection preserves every signed and unsigned integer class exactly.",
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
pub const DOWNSAMPLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = downsample(integer_X,N,PHASE?)",
        inputs: &DATA,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes:
            "Host and owner-resolved resident selection preserve authoritative storage and class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = downsample(X,integer_N,PHASE?)",
        inputs: &FACTOR,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The factor is gated before X/provider access and converted exactly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = downsample(X,N,integer_PHASE)",
        inputs: &PHASE,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The phase is gated before X/provider access and checked against N.",
    },
];
