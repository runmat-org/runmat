use super::signatures::CUMMAX_SIGNATURES;
use crate::*;
pub const CUMMAX_GPU_NANFLAG_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cummax-gpu-nanflag",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cummax with an explicit missing-value flag on a GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CummaxGpuNanflagExtension"),
};

pub const CUMMAX_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [CUMMAX_GPU_NANFLAG_EXTENSION];

pub const CUMMAX_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMMAX.INVALID_ARGUMENT",
    identifier: Some("RunMat:cummax:InvalidArgument"),
    when:
        "Input or output count, dimension, direction, or missing-value argument grammar is invalid.",
    message: "cummax: invalid argument",
};

pub const CUMMAX_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMMAX.INVALID_INPUT",
    identifier: Some("RunMat:cummax:InvalidInput"),
    when: "Input value type is unsupported for cumulative maximum reduction.",
    message: "cummax: invalid input",
};

pub const CUMMAX_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMMAX.INTERNAL",
    identifier: Some("RunMat:cummax:Internal"),
    when: "Reduction execution fails due to conversion, provider, or allocation operations.",
    message: "cummax: internal reduction failure",
};

const CUMMAX_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CUMMAX_ERROR_INVALID_ARGUMENT,
    CUMMAX_ERROR_INVALID_INPUT,
    CUMMAX_ERROR_INTERNAL,
];

pub const CUMMAX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CUMMAX_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CUMMAX_ERRORS,
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

pub const CUMMAX_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[M, I] = cummax(A, dim, direction, nanflag)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Cumulative maximum preserves the integer value class and exact shape in either direction. The optional second output contains one-based double indices. A provider unable to publish double indices declines that request before execution; explicit GPU missing-value flags are a mode-gated RunMat extension.",
    }];
