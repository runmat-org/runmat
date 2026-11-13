use crate::*;
pub const MAXK_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "maxk-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "maxk with a resident GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MaxkGpuInputExtension"),
};

pub const MINK_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "mink-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "mink with a resident GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:MinkGpuInputExtension"),
};

pub const MAXK_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [MAXK_GPU_INPUT_EXTENSION];
pub const MINK_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [MINK_GPU_INPUT_EXTENSION];

pub const TOPK_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented input-array domain includes all eight real integer classes.",
    },
    BuiltinIntegerInputCapability {
        name: "k",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented nonnegative integer-scalar selector is parsed exactly from every integer class; integer-valued scalar double is also accepted.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The optional documented positive integer-scalar dimension is parsed exactly from every integer class; integer-valued scalar double is also accepted.",
    },
];

pub const MAXK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[B, I] = maxk(integer_A, integer_k, integer_dim)",
        inputs: &TOPK_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "B preserves A's exact integer class and stable equal-value order; optional I is one-based double. Resident execution is a separately gated RunMat extension.",
    }];

pub const MINK_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[B, I] = mink(integer_A, integer_k, integer_dim)",
        inputs: &TOPK_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "B preserves A's exact integer class and stable equal-value order; optional I is one-based double. Resident execution is a separately gated RunMat extension.",
    }];

pub const TOPK_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOPK.INVALID_ARGUMENT",
    identifier: Some("RunMat:topk:InvalidArgument"),
    when: "Argument count, k, dimension, or option values are invalid.",
    message: "topk: invalid argument",
};

pub const TOPK_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOPK.INVALID_INPUT",
    identifier: Some("RunMat:topk:InvalidInput"),
    when: "Input values cannot be converted to supported top-k domains.",
    message: "topk: invalid input",
};

pub const TOPK_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TOPK.INTERNAL",
    identifier: Some("RunMat:topk:Internal"),
    when: "Top-k selection fails due to gather or allocation internals.",
    message: "topk: internal failure",
};

pub const TOPK_ERRORS: [BuiltinErrorDescriptor; 3] = [
    TOPK_ERROR_INVALID_ARGUMENT,
    TOPK_ERROR_INVALID_INPUT,
    TOPK_ERROR_INTERNAL,
];
