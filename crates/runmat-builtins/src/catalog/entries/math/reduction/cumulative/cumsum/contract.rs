use super::signatures::CUMSUM_SIGNATURES;
use crate::*;
pub const CUMSUM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMSUM.INVALID_ARGUMENT",
    identifier: Some("RunMat:cumsum:InvalidArgument"),
    when:
        "Input or output count, dimension, direction, or missing-value argument grammar is invalid.",
    message: "cumsum: invalid argument",
};

pub const CUMSUM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMSUM.INVALID_INPUT",
    identifier: Some("RunMat:cumsum:InvalidInput"),
    when: "Input value type is unsupported for cumulative sum reduction.",
    message: "cumsum: invalid input",
};

pub const CUMSUM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CUMSUM.INTERNAL",
    identifier: Some("RunMat:cumsum:Internal"),
    when: "Reduction execution fails due to conversion, provider, or allocation operations.",
    message: "cumsum: internal reduction failure",
};

const CUMSUM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CUMSUM_ERROR_INVALID_ARGUMENT,
    CUMSUM_ERROR_INVALID_INPUT,
    CUMSUM_ERROR_INTERNAL,
];

pub const CUMSUM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CUMSUM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CUMSUM_ERRORS,
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

pub const CUMSUM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = cumsum(A, dim, direction, nanflag)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Every prefix preserves the integer input class and shape with per-step saturating addition; forward/reverse and missing-value spellings share the rule, and provider scan order may differ for signed saturated arithmetic.",
    }];
