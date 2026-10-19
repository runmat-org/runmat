use crate::*;

pub const WBLINV_INTEGER_P_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblinv-integer-probability",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblinv with typed-integer probabilities is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblinvIntegerProbabilityExtension"),
};
pub const WBLINV_INTEGER_A_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblinv-integer-scale",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblinv with typed-integer scale parameters is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblinvIntegerScaleExtension"),
};
pub const WBLINV_INTEGER_B_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblinv-integer-shape",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblinv with typed-integer shape parameters is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblinvIntegerShapeExtension"),
};
pub const WBLINV_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblinv-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblinv with logical numeric input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblinvLogicalInputExtension"),
};
pub const WBLINV_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "wblinv-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "wblinv with explicit gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WblinvExplicitGpuInputExtension"),
};
pub const WBLINV_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    WBLINV_INTEGER_P_EXTENSION,
    WBLINV_INTEGER_A_EXTENSION,
    WBLINV_INTEGER_B_EXTENSION,
    WBLINV_LOGICAL_INPUT_EXTENSION,
    WBLINV_EXPLICIT_GPU_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "p",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented probability datatype is single or double; typed integers are independently gated and must be exactly representable at the binary64 distribution boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "a",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented scale datatype is single or double; typed integers are independently gated and checked before conversion.",
    },
    BuiltinIntegerInputCapability {
        name: "b",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented shape datatype is single or double; typed integers are independently gated and checked before conversion.",
    },
];
pub const WBLINV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "x = wblinv(integer_p, integer_a, integer_b)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each typed role is gated independently before provider access. Admitted values cross one checked binary64 Weibull boundary; single dominates output only when another documented input is single, and automatic fallback restores through a physically compatible owner.",
    }];
