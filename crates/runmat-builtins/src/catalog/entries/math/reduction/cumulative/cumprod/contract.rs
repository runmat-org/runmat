use super::signatures::CUMPROD_SIGNATURES;
use crate::*;
pub const CUMPROD_GPU_NANFLAG_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cumprod-gpu-nanflag",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cumprod with an explicit missing-value flag on a GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumprodGpuNanflagExtension"),
};

pub const CUMPROD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [CUMPROD_GPU_NANFLAG_EXTENSION];

pub const CUMPROD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMPROD.INVALID_ARGUMENT",
    identifier: Some("RunMat:cumprod:InvalidArgument"),
    when:
        "Input or output count, dimension, direction, or missing-value argument grammar is invalid.",
    message: "cumprod: invalid argument",
};

pub const CUMPROD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMPROD.INVALID_INPUT",
    identifier: Some("RunMat:cumprod:InvalidInput"),
    when: "Input value type is unsupported for cumulative product reduction.",
    message: "cumprod: invalid input",
};

pub const CUMPROD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMPROD.INTERNAL",
    identifier: Some("RunMat:cumprod:Internal"),
    when: "Reduction execution fails due to conversion, provider, or allocation operations.",
    message: "cumprod: internal reduction failure",
};

const CUMPROD_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CUMPROD_ERROR_INVALID_ARGUMENT,
    CUMPROD_ERROR_INVALID_INPUT,
    CUMPROD_ERROR_INTERNAL,
];

pub const CUMPROD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CUMPROD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CUMPROD_ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Ordinary real arrays accept all eight integer classes; complex-integer storage is rejected.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The optional positive scalar dimension is decoded exactly from typed integer or integer-valued floating storage.",
    },
];

pub const CUMPROD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = cumprod(A, dim, direction, nanflag)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Every prefix preserves the integer input class and shape with per-step saturating multiplication; forward/reverse and missing-value spellings share the host rule, explicit GPU nanflag use is a mode-gated extension, and provider scan order may differ.",
    }];
