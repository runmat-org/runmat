use crate::*;
pub const RESAMPLE_INTEGER_OPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor { id: "resample-integer-options", mode: BuiltinExtensionMode::RunMatOnly, description: "resample accepts typed-integer rate, filter-design, filter, and dimension arguments as a RunMat extension", error_identifier: Some("RunMat:compatibility:ResampleIntegerOptionsExtension") };
pub const RESAMPLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [RESAMPLE_INTEGER_OPTION_EXTENSION];
const DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "X", classes: &crate::catalog::integer::ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Rejected, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "The documented signal input is single or double; integer signal storage rejects before execution." }];
const STRUCTURAL: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "P/Q/N/Dimension",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "RunMat reads typed rate and dimension controls exactly.",
}];
const FLOAT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "BETA/B",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer beta/filter coefficients require exact binary64 conversion.",
}];
pub const RESAMPLE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "resample(integer_X,P,Q,...)",
        inputs: &DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer signal storage rejects from authoritative host/resident metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = resample(X,integer_P,integer_Q,integer_N,Dimension=integer_dim)",
        inputs: &STRUCTURAL,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Checked structural controls own filter/output geometry.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = resample(X,P,Q,integer_BETA_or_B)",
        inputs: &FLOAT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Filter integers cross one checked binary64 boundary.",
    },
];
