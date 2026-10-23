use super::signatures::CUMMIN_SIGNATURES;
use crate::*;
pub const CUMMIN_GPU_NANFLAG_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cummin-gpu-nanflag",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cummin with an explicit missing-value flag on a GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CumminGpuNanflagExtension"),
};

pub const CUMMIN_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [CUMMIN_GPU_NANFLAG_EXTENSION];

pub const CUMMIN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMMIN.INVALID_ARGUMENT",
    identifier: Some("RunMat:cummin:InvalidArgument"),
    when:
        "Input or output count, dimension, direction, or missing-value argument grammar is invalid.",
    message: "cummin: invalid argument",
};

pub const CUMMIN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMMIN.INVALID_INPUT",
    identifier: Some("RunMat:cummin:InvalidInput"),
    when: "Input value type is unsupported for cumulative minimum reduction.",
    message: "cummin: invalid input",
};

pub const CUMMIN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMMIN.INTERNAL",
    identifier: Some("RunMat:cummin:Internal"),
    when: "Reduction execution fails due to conversion, provider, or allocation operations.",
    message: "cummin: internal reduction failure",
};

const CUMMIN_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CUMMIN_ERROR_INVALID_ARGUMENT,
    CUMMIN_ERROR_INVALID_INPUT,
    CUMMIN_ERROR_INTERNAL,
];

pub const CUMMIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CUMMIN_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CUMMIN_ERRORS,
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

pub const CUMMIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[M, I] = cummin(A, dim, direction, nanflag)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Cumulative minimum preserves the integer value class and exact shape in either direction. The optional second output contains one-based double indices. A provider unable to publish double indices declines that request before execution; explicit GPU missing-value flags are a mode-gated RunMat extension.",
    }];
