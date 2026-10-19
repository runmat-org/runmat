use crate::*;

const STRINGS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Preallocated string array.",
}];

const STRINGS_INPUT_SZ: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector or scalar side length.",
}];

const STRINGS_INPUT_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First array dimension.",
    },
    BuiltinParamDescriptor {
        name: "n...",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional array dimensions.",
    },
];

const STRINGS_INPUT_LIKE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dims...",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional explicit size dimensions.",
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
        description: "Prototype value supplying output shape when dims are omitted.",
    },
];

const STRINGS_INPUT_FILL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dims...",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional explicit size dimensions.",
    },
    BuiltinParamDescriptor {
        name: "fill",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"empty\""),
        description: "Fill mode keyword: \"empty\" or \"missing\".",
    },
];

const STRINGS_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "S = strings()",
        inputs: &[],
        outputs: &STRINGS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = strings(sz)",
        inputs: &STRINGS_INPUT_SZ,
        outputs: &STRINGS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = strings(m, n...)",
        inputs: &STRINGS_INPUT_DIMS,
        outputs: &STRINGS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = strings(___, \"like\", p)",
        inputs: &STRINGS_INPUT_LIKE,
        outputs: &STRINGS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = strings(___, fill)",
        inputs: &STRINGS_INPUT_FILL,
        outputs: &STRINGS_OUTPUT,
    },
];

pub const STRINGS_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRINGS.INVALID_SIZE",
    identifier: Some("RunMat:strings:InvalidSize"),
    when: "Size arguments are not valid numeric scalar/vector dimensions.",
    message: "strings: size arguments must be numeric scalars or vectors",
};

pub const STRINGS_ERROR_LIKE_MISSING_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRINGS.LIKE_MISSING_PROTOTYPE",
    identifier: Some("RunMat:strings:LikeMissingPrototype"),
    when: "\"like\" is provided without a following prototype.",
    message: "strings: expected prototype after 'like'",
};

pub const STRINGS_ERROR_LIKE_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRINGS.LIKE_DUPLICATE",
    identifier: Some("RunMat:strings:LikeDuplicate"),
    when: "Multiple \"like\" options are supplied in one call.",
    message: "strings: multiple 'like' specifications are not supported",
};

pub const STRINGS_ERROR_SIZE_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRINGS.SIZE_OVERFLOW",
    identifier: Some("RunMat:strings:SizeOverflow"),
    when: "Requested dimensions overflow platform limits.",
    message: "strings: requested size exceeds platform limits",
};

pub const STRINGS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRINGS.INTERNAL",
    identifier: Some("RunMat:strings:InternalError"),
    when: "Internal string array construction failed.",
    message: "strings: internal error",
};

pub const STRINGS_ERRORS: [BuiltinErrorDescriptor; 5] = [
    STRINGS_ERROR_INVALID_SIZE,
    STRINGS_ERROR_LIKE_MISSING_PROTOTYPE,
    STRINGS_ERROR_LIKE_DUPLICATE,
    STRINGS_ERROR_SIZE_OVERFLOW,
    STRINGS_ERROR_INTERNAL,
];

pub const STRINGS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRINGS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRINGS_ERRORS,
};

pub const STRINGS_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "strings-like-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "strings(..., 'like', prototype) is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:StringsLikePrototypeExtension"),
};

pub const STRINGS_FILL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "strings-fill-mode",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "strings(..., 'empty'|'missing') fill selection is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:StringsFillModeExtension"),
};

pub const STRINGS_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [STRINGS_LIKE_EXTENSION, STRINGS_FILL_EXTENSION];

const STRINGS_SIZE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "size",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Scalar, separate, and vector size forms accept every built-in integer class. Negative values become zero, and dimensions are checked exactly against the host allocation domain.",
    }];

const STRINGS_LIKE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "prototype",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The RunMat-only like form inspects only prototype shape; integer class and payload are not converted or read.",
    }];

pub const STRINGS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "str = strings(n | sz | sz1,...,szN)",
        inputs: &STRINGS_SIZE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility target documents GPU-array size input but host execution. RunMat gathers through the owning provider, parses native integer storage exactly, and always returns a host string array.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "str = strings(..., 'like', prototype)",
        inputs: &STRINGS_LIKE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The like form is a mode-gated RunMat extension and is not presented as part of the compatibility target's strings contract.",
    },
];
