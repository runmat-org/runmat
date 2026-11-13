use crate::*;

pub const ENVELOPE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "envelope-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "envelope with typed-integer signal data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EnvelopeIntegerDataExtension"),
    };
pub const ENVELOPE_INTEGER_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "envelope-integer-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "envelope with a typed-integer length control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EnvelopeIntegerControlExtension"),
    };
pub const ENVELOPE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ENVELOPE_INTEGER_DATA_EXTENSION,
    ENVELOPE_INTEGER_CONTROL_EXTENSION,
];
const DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "x",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integer signal data cross an exact floating computation boundary.",
}];
const CONTROL: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "n",
    classes: &crate::catalog::integer::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Typed filter/window/separation controls are parsed as exact structural integers.",
}];
pub const ENVELOPE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[yupper, ylower] = envelope(integer_x, ...)",
        inputs: &DATA,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Signal integers remain authoritative until floating envelope arithmetic; resident input gathers through its owner.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[yupper, ylower] = envelope(x, integer_n, method?)",
        inputs: &CONTROL,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Control is extension-gated before accessing x or a provider; signal precision is preserved.",
    },
];
