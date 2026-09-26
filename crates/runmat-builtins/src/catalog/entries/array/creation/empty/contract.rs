use crate::*;
pub const EMPTY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty array preserving the receiver class.",
}];
pub const EMPTY_NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const fn dimension_parameter() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "dimensions",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Integer scalar dimensions; at least one dimension must be zero.",
    }
}
pub const EMPTY_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [dimension_parameter()];
pub const EMPTY_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sizeVector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of integer dimensions; at least one dimension must be zero.",
}];
pub const EMPTY_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "A = ClassName.empty",
        inputs: &EMPTY_NO_INPUTS,
        outputs: &EMPTY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = ClassName.empty(sz1, ..., szN)",
        inputs: &EMPTY_DIMS_INPUTS,
        outputs: &EMPTY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = ClassName.empty(sizeVector)",
        inputs: &EMPTY_SIZE_VECTOR_INPUTS,
        outputs: &EMPTY_OUTPUT,
    },
];
pub const EMPTY_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EMPTY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:empty:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "empty: too many output arguments",
};
pub const EMPTY_ERROR_INVALID_DIMS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EMPTY.INVALID_DIMS",
    identifier: Some("RunMat:empty:InvalidSize"),
    when: "Size arguments are not integer-valued numeric scalars or one row size vector.",
    message: "empty: invalid size arguments",
};
pub const EMPTY_ERROR_NON_EMPTY_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EMPTY.NON_EMPTY_SHAPE",
    identifier: Some("RunMat:empty:NonEmptyShape"),
    when: "Every normalized dimension is nonzero.",
    message: "empty: at least one dimension must be zero",
};
pub const EMPTY_ERROR_UNSUPPORTED_CLASS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EMPTY.UNSUPPORTED_CLASS",
    identifier: Some("RunMat:empty:UnsupportedClass"),
    when: "The selected shorthand class cannot be represented as a shaped empty value.",
    message: "empty: unsupported output class",
};
pub const EMPTY_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.EMPTY.INTERNAL",
    identifier: Some("RunMat:empty:Internal"),
    when: "The empty value cannot be assembled.",
    message: "empty: internal error",
};
pub const EMPTY_ERRORS: [BuiltinErrorDescriptor; 5] = [
    EMPTY_ERROR_INVALID_DIMS,
    EMPTY_ERROR_NON_EMPTY_SHAPE,
    EMPTY_ERROR_UNSUPPORTED_CLASS,
    EMPTY_ERROR_INTERNAL,
    EMPTY_ERROR_TOO_MANY_OUTPUTS,
];

pub const EMPTY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EMPTY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EMPTY_ERRORS,
};

const GLOBAL_INPUTS: [BuiltinParamDescriptor; 2] = [
    dimension_parameter(),
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("double"),
        description: "Output class: numeric, logical, char, string, cell, or gpuArray. Struct is currently rejected.",
    },
];
pub const EMPTY_GLOBAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "A = empty()",
            inputs: &EMPTY_NO_INPUTS,
            outputs: &EMPTY_OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "A = empty(dimensions..., typename)",
            inputs: &GLOBAL_INPUTS,
            outputs: &EMPTY_OUTPUT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EMPTY_ERRORS,
};

pub const EMPTY_GLOBAL_CALL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "empty-global-call",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "calling empty as a global function is a RunMat extension; MATLAB uses ClassName.empty",
    error_identifier: Some("RunMat:compatibility:EmptyGlobalCallExtension"),
};
pub const EMPTY_TYPENAME_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "empty-typename",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "selecting the empty output class with a trailing typename is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EmptyTypenameExtension"),
};
pub const EMPTY_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "empty-resident-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "empty with a resident size control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EmptyResidentSizeExtension"),
};
pub const EMPTY_GLOBAL_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    EMPTY_GLOBAL_CALL_EXTENSION,
    EMPTY_TYPENAME_EXTENSION,
    EMPTY_RESIDENT_SIZE_EXTENSION,
];
pub const EMPTY_STATIC_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [EMPTY_RESIDENT_SIZE_EXTENSION];

pub const EMPTY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "IntegerClass receiver",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Each of the eight integer primitive classes exposes ClassName.empty and determines the exact output storage class.",
    },
    BuiltinIntegerInputCapability {
        name: "size dimensions",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer scalar dimensions and row size vectors are decoded from authoritative storage; signed negatives clamp to zero.",
    },
];
pub const EMPTY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "A = IntegerClass.empty, IntegerClass.empty(sz1, ..., szN), or IntegerClass.empty(sizeVector)",
        inputs: &EMPTY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Output is empty authoritative storage of the receiver class. At least one normalized dimension is zero, trailing singleton dimensions after dimension two are removed, and resident controls are independently gated before exact gather.",
    }];
