use crate::*;

pub const CAT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Concatenated output array.",
}];

pub const CAT_INPUTS_CORE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Concatenation dimension (1-based).",
    },
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "A2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input array.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional input arrays.",
    },
];

pub const CAT_INPUTS_LIKE: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Concatenation dimension (1-based).",
    },
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First input array.",
    },
    BuiltinParamDescriptor {
        name: "A2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second input array.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional input arrays.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::PropertyName,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Name-value key ('like').",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype that sets output class/device.",
    },
];

pub const CAT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = cat(dim, A1, A2, An...)",
        inputs: &CAT_INPUTS_CORE,
        outputs: &CAT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = cat(dim, A1, A2, An..., \"like\", prototype)",
        inputs: &CAT_INPUTS_LIKE,
        outputs: &CAT_OUTPUT,
    },
];

pub const CAT_ERROR_TOO_FEW_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAT.TOO_FEW_INPUTS",
    identifier: Some("RunMat:cat:TooFewInputs"),
    when: "Fewer than two input arrays are supplied after dim.",
    message: "cat: at least two input arrays are required",
};

pub const CAT_ERROR_INVALID_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAT.INVALID_DIMENSION",
    identifier: Some("RunMat:cat:InvalidDimension"),
    when: "dim is non-numeric, non-integer, or less than one.",
    message: "cat: dimension must be numeric and >= 1",
};

pub const CAT_ERROR_GPU: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAT.GPU",
    identifier: Some("RunMat:cat:GpuError"),
    when: "A required gpuArray gather, upload, or provider operation fails.",
    message: "cat: gpuArray operation failed",
};

pub const CAT_ERROR_TYPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAT.TYPE_MISMATCH",
    identifier: Some("RunMat:cat:TypeMismatch"),
    when: "Inputs are from incompatible classes for concatenation.",
    message: "cat: incompatible input classes for concatenation",
};

pub const CAT_ERROR_INVALID_LIKE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CAT.INVALID_LIKE",
    identifier: Some("RunMat:cat:InvalidLikePrototype"),
    when: "The 'like' name-value form is malformed or uses an unsupported prototype.",
    message: "cat: invalid 'like' prototype",
};

pub const CAT_ERRORS: [BuiltinErrorDescriptor; 5] = [
    CAT_ERROR_TOO_FEW_INPUTS,
    CAT_ERROR_INVALID_DIMENSION,
    CAT_ERROR_GPU,
    CAT_ERROR_TYPE_MISMATCH,
    CAT_ERROR_INVALID_LIKE,
];

pub const CAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CAT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CAT_ERRORS,
};

pub const CAT_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cat-like-prototype",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the trailing cat \"like\" prototype form is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CatLikePrototypeExtension"),
};

pub const CAT_RESIDENT_DIMENSION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cat-resident-dimension",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cat with a resident gpuArray dimension control is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CatResidentDimensionExtension"),
    };

pub const CAT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [CAT_LIKE_EXTENSION, CAT_RESIDENT_DIMENSION_EXTENSION];

pub const CAT_INTEGER_DIMENSION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A host scalar from any integer class is read exactly as the positive one-based concatenation dimension; scalar double remains accepted by the same structural parser.",
    }];

pub const CAT_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A1,...,An",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Every ordered mixture of the eight real integer classes is accepted together with real single, double, or logical data; the leftmost participating integer class determines the numeric output class.",
    }];

pub const CAT_COMPLEX_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "complex A1,...,An",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight paired complex-integer classes concatenate structurally with exact real and imaginary components.",
    }];

pub const CAT_RESIDENT_INTEGER_DIMENSION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "gpuArray dim",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode may gather a resident scalar dimension exactly before structural concatenation; the documented dimension control is a host positive integer scalar.",
    }];

pub const CAT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = cat(integer_dim,A1,A2,...,An)",
        inputs: &CAT_INTEGER_DIMENSION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The integer dimension is a structural control only and does not affect output class. It is range-checked exactly before shape arithmetic; data inputs independently determine output class and residency.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = cat(dim,A1,A2,...,An) with real integer data",
        inputs: &CAT_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The leftmost nonempty integer class dominates real numeric and logical operands. Unlike integer values and floating or logical values convert to that class with ordinary rounding and saturation. Empty operands are omitted when a nonempty operand remains and therefore do not affect the result class. Resident data uses the provider hook where representable or exact gather/assemble/upload fallback.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = cat(dim,complex_integer_A1,...,An)",
        inputs: &CAT_COMPLEX_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Typed complex-integer concatenation applies the leftmost integer component class independently to real and imaginary parts, preserves exact paired storage, and restores resident output to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = cat(gpuArray(integer_dim),A1,A2,...,An)",
        inputs: &CAT_RESIDENT_INTEGER_DIMENSION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "RunMat-only resident dimension controls are gathered exactly after the extension gate; the scalar remains a structural control and data inputs independently determine output class and residency.",
    },
];
