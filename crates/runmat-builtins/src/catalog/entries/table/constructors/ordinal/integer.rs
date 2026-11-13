use crate::*;

pub const ORDINAL_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ordinal-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ordinal host construction for explicit gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:OrdinalExplicitGpuInputExtension"),
};
pub const ORDINAL_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ORDINAL_GPU_INPUT_EXTENSION];

const ORDINAL_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric input accepts all eight integer classes; exact native values determine sorted ordinal levels and labels.",
    }];
const ORDINAL_INTEGER_LEVEL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer source values retain exact identity while levels are assigned.",
    },
    BuiltinIntegerInputCapability {
        name: "levels",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Explicit numeric levels accept all eight integer classes and are matched through the same exact categorical construction path.",
    },
];
const ORDINAL_INTEGER_EDGE_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric source accepts all eight integer classes and is compared directly with each bin edge.",
    },
    BuiltinIntegerInputCapability {
        name: "edges",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented numeric edge vector accepts all eight integer classes; strict ordering and half-open bin membership use exact mixed-class numeric comparison.",
    },
];
pub const ORDINAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = ordinal(integer_X)",
        inputs: &ORDINAL_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact integer identity determines levels and codes before default labels cross the documented display-only five-significant-digit boundary; colliding display labels reject. Automatic residency gathers transparently, while explicit gpuArray input is gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = ordinal(integer_X, labels, integer_levels)",
        inputs: &ORDINAL_INTEGER_LEVEL_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Source and level vectors preserve exact signedness, width, and value identity until the opaque ordinal metadata object is assembled.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = ordinal(integer_X, labels, [], integer_edges)",
        inputs: &ORDINAL_INTEGER_EDGE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Strictly increasing integer edges are compared exactly; bins are left-closed and right-open except that the last bin includes the final edge, and duplicate textual labels merge bins.",
    },
];
