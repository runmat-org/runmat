//! Canonical eye signatures, errors, extensions, and integer forms.
use crate::*;

const EYE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Identity matrix/tensor.",
}];

const EYE_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const EYE_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const EYE_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const EYE_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const EYE_SIG_PROTOTYPE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "prototype",
    ty: BuiltinParamType::LikePrototype,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Prototype value when no numeric dimension arguments are provided.",
}];

pub const EYE_IMPLICIT_PROTOTYPE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "eye-implicit-prototype",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eye(A) implicit size-and-class prototype syntax is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EyeImplicitPrototypeExtension"),
    };

pub const EYE_ND_DIMENSIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "eye-nd-dimensions",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "eye with more than two dimensions is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EyeNdDimensionsExtension"),
};

pub const EYE_COLUMN_SIZE_VECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "eye-column-size-vector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "eye with a column size vector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EyeColumnSizeVectorExtension"),
    };

pub const EYE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    EYE_IMPLICIT_PROTOTYPE_EXTENSION,
    EYE_ND_DIMENSIONS_EXTENSION,
    EYE_COLUMN_SIZE_VECTOR_EXTENSION,
];

const EYE_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/m/sz",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are exact structural size controls. Negative signed values clamp to zero; values outside the platform allocation domain reject before allocation.",
    }];

const EYE_INTEGER_LIKE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "p",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An integer prototype selects its exact class and applicable residency; it never supplies output shape.",
    }];

pub const EYE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "I = eye(integer_n[, integer_m])",
        inputs: &EYE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The default output is double; typename or like can select logical, single, or an exact integer class. MATLAB-compatible mode admits at most two dimensions.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "I = eye(integer_sz)",
        inputs: &EYE_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented size vector is a row vector containing no more than two exact integer values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "I = eye(..., integer_typename)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Every integer typename creates exact native zero and one storage in the selected class.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "I = eye(..., like=integer_p)",
        inputs: &EYE_INTEGER_LIKE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Dense integer prototypes preserve exact class. Resident prototypes use their owning provider; unsupported native construction falls back through exact typed upload.",
    },
];

const EYE_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Class name override (double|single|logical|int8|int16|int32|int64|uint8|uint16|uint32|uint64).",
    },
];

const EYE_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
    BuiltinParamDescriptor {
        name: "like_kw",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "Like keyword.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype array used for class/device.",
    },
];

const EYE_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "A = eye()",
        inputs: &EYE_SIG_EMPTY_INPUTS,
        outputs: &EYE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = eye(n)",
        inputs: &EYE_SIG_N_INPUTS,
        outputs: &EYE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = eye(size_vector)",
        inputs: &EYE_SIG_SIZE_VECTOR_INPUTS,
        outputs: &EYE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = eye(m, n, ...)",
        inputs: &EYE_SIG_DIMS_INPUTS,
        outputs: &EYE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = eye(prototype)",
        inputs: &EYE_SIG_PROTOTYPE_INPUTS,
        outputs: &EYE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = eye(..., typename)",
        inputs: &EYE_SIG_CLASS_INPUTS,
        outputs: &EYE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = eye(..., \"like\", prototype)",
        inputs: &EYE_SIG_LIKE_INPUTS,
        outputs: &EYE_OUTPUT,
    },
];

const EYE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.EYE.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "eye: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.EYE.CLASS_CONFLICT",
        identifier: None,
        when: "A class keyword and a 'like' prototype are both provided.",
        message: "eye: cannot combine 'like' with other class specifiers",
    },
    BuiltinErrorDescriptor {
        code: "RM.EYE.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not a supported class keyword.",
        message: "eye: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.EYE.INVALID_DIMS",
        identifier: None,
        when: "Dimension arguments fail numeric/shape parsing.",
        message: "eye: dimension arguments must be numeric and nonnegative",
    },
];

pub const EYE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EYE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EYE_ERRORS,
};
