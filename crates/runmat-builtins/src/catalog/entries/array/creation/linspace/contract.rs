//! Canonical linspace signatures, errors, extensions, and integer forms.
use crate::*;

const LINSPACE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of linearly spaced values.",
}];

const LINSPACE_SIG_2_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Starting value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ending value.",
    },
];

const LINSPACE_SIG_3_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Starting value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ending value.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("100"),
        description: "Number of points.",
    },
];

const LINSPACE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "x = linspace(start, stop)",
        inputs: &LINSPACE_SIG_2_INPUTS,
        outputs: &LINSPACE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "x = linspace(start, stop, n)",
        inputs: &LINSPACE_SIG_3_INPUTS,
        outputs: &LINSPACE_OUTPUT,
    },
];

const LINSPACE_INTEGER_ENDPOINT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "start",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer endpoints are rejected; linspace endpoints must be single or double.",
    },
    BuiltinIntegerInputCapability {
        name: "stop",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer endpoints are rejected; linspace endpoints must be single or double.",
    },
];

const LINSPACE_INTEGER_COUNT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are accepted as exact structural counts.",
    }];

pub const LINSPACE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "x = linspace(integer_start, integer_stop[, n])",
        inputs: &LINSPACE_INTEGER_ENDPOINT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes:
            "Host and resident integer endpoints are rejected before numeric sequence generation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "x = linspace(start, stop, integer_n)",
        inputs: &LINSPACE_INTEGER_COUNT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The count is read exactly; output class follows the floating-point endpoints.",
    },
];

pub const LINSPACE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.ARG_COUNT",
    identifier: None,
    when: "More than three input arguments are provided.",
    message: "linspace: expected at most three input arguments",
};

pub const LINSPACE_ERROR_COUNT_NOT_SCALAR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.COUNT_NOT_SCALAR",
    identifier: None,
    when: "The count argument is not a numeric scalar value.",
    message: "linspace: number of points must be a scalar",
};

pub const LINSPACE_ERROR_COUNT_NOT_FINITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.COUNT_NOT_FINITE",
    identifier: None,
    when: "The count argument is infinite.",
    message: "linspace: number of points must not be infinite",
};

pub const LINSPACE_ERROR_COUNT_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.COUNT_TOO_LARGE",
    identifier: None,
    when: "The count argument exceeds platform limits.",
    message: "linspace: number of points is too large for this platform",
};

const LINSPACE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    LINSPACE_ERROR_ARG_COUNT,
    LINSPACE_ERROR_COUNT_NOT_SCALAR,
    LINSPACE_ERROR_COUNT_NOT_FINITE,
    LINSPACE_ERROR_COUNT_TOO_LARGE,
];

pub const LINSPACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LINSPACE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LINSPACE_ERRORS,
};
