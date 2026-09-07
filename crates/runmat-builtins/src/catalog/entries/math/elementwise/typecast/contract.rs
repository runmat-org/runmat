use crate::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ALL_INTEGER_CLASSES,
};

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Reinterpreted scalar or vector.",
}];
const NEWTYPE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Full numeric or logical scalar or vector.",
    },
    BuiltinParamDescriptor {
        name: "newtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or logical output class.",
    },
];
const LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Full numeric or logical scalar or vector.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Literal string \"like\".",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Host output class and complexity prototype.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = typecast(X, newtype)",
        inputs: &NEWTYPE_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = typecast(X, \"like\", prototype)",
        inputs: &LIKE_INPUTS,
        outputs: &OUTPUT,
    },
];

pub const TYPECAST_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TYPECAST.INVALID_ARGUMENT",
    identifier: Some("RunMat:typecast:InvalidArgument"),
    when: "The requested class, prototype, or argument list is invalid.",
    message: "typecast: invalid argument",
};
pub const TYPECAST_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TYPECAST.INVALID_INPUT",
    identifier: Some("RunMat:typecast:InvalidInput"),
    when: "The input is sparse, is not a scalar or vector, or has an incompatible byte count.",
    message: "typecast: invalid input",
};
pub const TYPECAST_ERROR_GPU_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TYPECAST.GPU_UNSUPPORTED",
    identifier: Some("RunMat:typecast:GpuUnsupported"),
    when: "A GPU call uses complex or logical input or the like syntax.",
    message: "typecast: unsupported gpuArray form",
};
pub const TYPECAST_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TYPECAST.INTERNAL",
    identifier: Some("RunMat:typecast:Internal"),
    when: "Exact gather, reconstruction, or resident restoration fails.",
    message: "typecast: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    TYPECAST_ERROR_INVALID_ARGUMENT,
    TYPECAST_ERROR_INVALID_INPUT,
    TYPECAST_ERROR_GPU_UNSUPPORTED,
    TYPECAST_ERROR_INTERNAL,
];

pub const TYPECAST_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "X", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Every native integer class is reinterpreted from its authoritative byte representation without numeric conversion." }];

pub const TYPECAST_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "Y = typecast(integer_X, newtype)", inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Scalar and vector inputs preserve their native byte sequence and orientation; output class and element count follow the requested element width." },
    BuiltinIntegerCapabilityDescriptor { form: "Y = typecast(gpuArray(integer_X), newtype)", inputs: &INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Real numeric resident input is transferred exactly through its owning provider, reinterpreted without floating conversion, and restored to that provider; complex, logical, and like GPU forms reject before transfer." },
];
