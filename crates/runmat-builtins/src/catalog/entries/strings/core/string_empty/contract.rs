use crate::*;

const STRING_EMPTY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty string array with at least one zero dimension.",
}];

const STRING_EMPTY_INPUT_SZ: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector or scalar.",
}];

const STRING_EMPTY_INPUT_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First dimension.",
    },
    BuiltinParamDescriptor {
        name: "n...",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional dimensions.",
    },
];

const STRING_EMPTY_INPUT_LIKE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dims...",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional explicit dimensions.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Literal option keyword \"like\".",
    },
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype supplying trailing dimensions.",
    },
];

const STRING_EMPTY_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "S = string.empty()",
        inputs: &[],
        outputs: &STRING_EMPTY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = string.empty(sz)",
        inputs: &STRING_EMPTY_INPUT_SZ,
        outputs: &STRING_EMPTY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = string.empty(m, n...)",
        inputs: &STRING_EMPTY_INPUT_DIMS,
        outputs: &STRING_EMPTY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = string.empty(___, \"like\", p)",
        inputs: &STRING_EMPTY_INPUT_LIKE,
        outputs: &STRING_EMPTY_OUTPUT,
    },
];

pub const STRING_EMPTY_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRING_EMPTY.INVALID_SIZE",
    identifier: Some("RunMat:string.empty:InvalidSize"),
    when: "Size inputs are not valid numeric dimensions or vectors.",
    message: "string.empty: size inputs must be numeric scalars or size vectors",
};

pub const STRING_EMPTY_ERROR_LIKE_MISSING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRING_EMPTY.LIKE_MISSING_PROTOTYPE",
    identifier: Some("RunMat:string.empty:LikeMissingPrototype"),
    when: "\"like\" keyword is present without a prototype.",
    message: "string.empty: expected prototype after 'like'",
};

pub const STRING_EMPTY_ERROR_LIKE_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRING_EMPTY.LIKE_DUPLICATE",
    identifier: Some("RunMat:string.empty:LikeDuplicate"),
    when: "Multiple \"like\" specifications are supplied.",
    message: "string.empty: multiple 'like' prototypes are not supported",
};

pub const STRING_EMPTY_ERROR_NOT_EMPTY_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRING_EMPTY.NONEMPTY_SHAPE",
    identifier: Some("RunMat:string.empty:NonEmptyShape"),
    when: "Parsed dimensions do not produce an empty array shape.",
    message: "string.empty: at least one dimension must be zero",
};

pub const STRING_EMPTY_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRING_EMPTY.INTERNAL",
    identifier: Some("RunMat:string.empty:InternalError"),
    when: "Internal empty string-array construction failed.",
    message: "string.empty: internal error",
};

pub const STRING_EMPTY_ERRORS: [BuiltinErrorDescriptor; 5] = [
    STRING_EMPTY_ERROR_INVALID_SIZE,
    STRING_EMPTY_ERROR_LIKE_MISSING,
    STRING_EMPTY_ERROR_LIKE_DUPLICATE,
    STRING_EMPTY_ERROR_NOT_EMPTY_SHAPE,
    STRING_EMPTY_ERROR_INTERNAL,
];

pub const STRING_EMPTY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRING_EMPTY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &STRING_EMPTY_ERRORS,
};

const STRING_EMPTY_INTEGER_SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "size",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Native integer dimensions are decoded exactly; the leading result dimension is always zero.",
    }];

pub const STRING_EMPTY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "S = string.empty(n | sz | m, n, ...)",
        inputs: &STRING_EMPTY_INTEGER_SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The size parser accepts all eight integer classes without a floating-point round trip. Negative integer dimensions currently clamp to zero in the shared constructor parser; negative floating dimensions reject.",
    }];
