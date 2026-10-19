//! Canonical logspace signatures, errors, extensions, and integer forms.
use crate::*;

const LOGSPACE_INTEGER_COUNT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The point count accepts an exact real integer scalar; nonpositive signed values produce an empty row and oversized values reject before allocation.",
    }];

const LOGSPACE_INTEGER_ENDPOINT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "start",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Endpoints are restricted to documented single or double scalars.",
    },
    BuiltinIntegerInputCapability {
        name: "stop",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Endpoints are restricted to documented single or double scalars.",
    },
];

pub const LOGSPACE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "x = logspace(start,stop,integer_n)",
        inputs: &LOGSPACE_INTEGER_COUNT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The exact integer count controls row length; floating endpoints determine output precision and explicit endpoint residency determines device placement.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "x = logspace(integer_start,integer_stop[,n])",
        inputs: &LOGSPACE_INTEGER_ENDPOINT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Typed-integer endpoints are rejected on host and from resident dtype metadata before provider access.",
    },
];

const LOGSPACE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of logarithmically spaced values.",
}];

const LOGSPACE_SIG_2_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Starting exponent value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ending exponent value.",
    },
];

const LOGSPACE_SIG_3_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Starting exponent value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ending exponent value.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("50"),
        description: "Number of points.",
    },
];

const LOGSPACE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "x = logspace(start, stop)",
        inputs: &LOGSPACE_SIG_2_INPUTS,
        outputs: &LOGSPACE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "x = logspace(start, stop, n)",
        inputs: &LOGSPACE_SIG_3_INPUTS,
        outputs: &LOGSPACE_OUTPUT,
    },
];

pub const LOGSPACE_ERRORS: [BuiltinErrorDescriptor; 6] = [
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.ARG_COUNT",
        identifier: None,
        when: "More than three input arguments are provided.",
        message: "logspace: expected two or three input arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.COUNT_NOT_SCALAR",
        identifier: None,
        when: "The count argument is not a numeric scalar value.",
        message: "logspace: number of points must be a scalar",
    },
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.COUNT_NOT_FINITE",
        identifier: None,
        when: "The count argument is NaN or infinite.",
        message: "logspace: number of points must be finite",
    },
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.COUNT_TOO_LARGE",
        identifier: None,
        when: "The count argument exceeds platform limits.",
        message: "logspace: number of points is too large for this platform",
    },
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.PROVIDER_OWNERSHIP_MISMATCH",
        identifier: Some("RunMat:gpu:ProviderOwnershipMismatch"),
        when: "Resident scalar inputs do not have one exact owning provider.",
        message: "logspace: GPU scalar inputs must have the same owning provider",
    },
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.GPU_UPLOAD_FAILED",
        identifier: Some("RunMat:logspace:GpuUploadFailed"),
        when: "An explicitly resident result cannot be restored to its source provider.",
        message: "logspace: failed to preserve explicit gpuArray residency",
    },
];

pub const LOGSPACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOGSPACE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOGSPACE_ERRORS,
};
