//! Canonical randi signatures, errors, extensions, and integer forms.
use crate::*;

const RANDI_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Uniform random integers.",
}];

pub const RANDI_WIDE_INTEGER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "randi-wide-integer-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "int64 and uint64 output classes are RunMat extensions",
    error_identifier: Some("RunMat:compatibility:RandiWideIntegerExtension"),
};

pub const RANDI_IMPLICIT_PROTOTYPE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "randi-implicit-prototype",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "the bare randi(bounds, prototype) shorthand is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RandiImplicitPrototypeExtension"),
    };

pub const RANDI_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    RANDI_WIDE_INTEGER_EXTENSION,
    RANDI_IMPLICIT_PROTOTYPE_EXTENSION,
];

const RANDI_SIG_IMAX_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "imax",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper bound; lower bound defaults to 1.",
}];

const RANDI_SIG_BOUNDS_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bounds",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Two-element bounds vector [imin imax].",
}];

const RANDI_SIG_IMAX_N_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "imax",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound; lower bound defaults to 1.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Square size.",
    },
];

const RANDI_SIG_BOUNDS_N_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element bounds vector [imin imax].",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Square size.",
    },
];

const RANDI_SIG_IMAX_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "imax",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound; lower bound defaults to 1.",
    },
    BuiltinParamDescriptor {
        name: "size_vector",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Size vector defining output dimensions.",
    },
];

const RANDI_SIG_BOUNDS_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element bounds vector [imin imax].",
    },
    BuiltinParamDescriptor {
        name: "size_vector",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Size vector defining output dimensions.",
    },
];

const RANDI_SIG_IMAX_DIMS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "imax",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound; lower bound defaults to 1.",
    },
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
];

const RANDI_SIG_BOUNDS_DIMS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Two-element bounds vector [imin imax].",
    },
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Dimension sizes.",
    },
];

const RANDI_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound scalar or two-element bounds vector.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Output class override; int64 and uint64 are RunMat-mode extensions.",
    },
];

const RANDI_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound scalar or two-element bounds vector.",
    },
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

const RANDI_SIG_BOUNDS_PROTOTYPE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "bounds",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper bound scalar or two-element bounds vector.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "RunMat-only shorthand that copies prototype shape and class without a like keyword.",
    },
];

const RANDI_SIGNATURES: [BuiltinSignatureDescriptor; 11] = [
    BuiltinSignatureDescriptor {
        label: "R = randi(imax)",
        inputs: &RANDI_SIG_IMAX_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi([imin imax])",
        inputs: &RANDI_SIG_BOUNDS_VECTOR_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(imax, n)",
        inputs: &RANDI_SIG_IMAX_N_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi([imin imax], n)",
        inputs: &RANDI_SIG_BOUNDS_N_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(imax, size_vector)",
        inputs: &RANDI_SIG_IMAX_SIZE_VECTOR_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi([imin imax], size_vector)",
        inputs: &RANDI_SIG_BOUNDS_SIZE_VECTOR_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(imax, m, n, ...)",
        inputs: &RANDI_SIG_IMAX_DIMS_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi([imin imax], m, n, ...)",
        inputs: &RANDI_SIG_BOUNDS_DIMS_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(..., typename)",
        inputs: &RANDI_SIG_CLASS_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(..., \"like\", prototype)",
        inputs: &RANDI_SIG_LIKE_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "R = randi(bounds, prototype)",
        inputs: &RANDI_SIG_BOUNDS_PROTOTYPE_INPUTS,
        outputs: &RANDI_OUTPUT,
    },
];

const RANDI_ERRORS: [BuiltinErrorDescriptor; 9] = [
    BuiltinErrorDescriptor {
        code: "RM.RANDI.MISSING_BOUNDS",
        identifier: None,
        when: "No bounds argument is provided.",
        message: "randi: requires at least one input argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.INVALID_BOUNDS",
        identifier: None,
        when: "Bounds are invalid or unsupported (empty, non-finite, non-integer, or malformed).",
        message: "randi: bounds must be numeric scalars or vectors",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.BOUNDS_ORDER",
        identifier: None,
        when: "Lower bound exceeds upper bound.",
        message: "randi: lower bound must be <= upper bound",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "randi: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "randi: multiple 'like' specifications are not supported",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.CLASS_CONFLICT",
        identifier: None,
        when: "A class keyword and a 'like' prototype are both provided.",
        message: "randi: cannot combine 'like' with class specifiers",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.UNSUPPORTED_CLASS",
        identifier: None,
        when: "An unsupported output class is requested.",
        message: "randi: output class is not implemented",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not recognized.",
        message: "randi: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDI.INVALID_DIMS",
        identifier: None,
        when: "Dimension arguments fail numeric/shape parsing.",
        message: "randi: dimension arguments must be numeric and nonnegative",
    },
];

pub const RANDI_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RANDI_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RANDI_ERRORS,
};

const RANDI_INTEGER_BOUNDS_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "imax or [imin imax]",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer bounds are decoded exactly and define an inclusive discrete interval without passing through binary64.",
    }];

const RANDI_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz1...szN/sz",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight documented integer size classes are structural; signed negatives clamp to zero.",
    }];

const RANDI_INTEGER_PROTOTYPE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "like prototype or integer output class",
        classes: &INTEGER_CLASSES_THROUGH_32_BITS,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The six documented integer output classes through 32 bits preserve native storage; supported resident prototypes preserve their owner and class.",
    }];

const RANDI_WIDE_INTEGER_PROTOTYPE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "int64/uint64 like prototype or output class",
        classes: &[
            BuiltinIntegerClass::Int64,
            BuiltinIntegerClass::Uint64,
        ],
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target omits int64/uint64 from the documented output class and prototype lists; RunMat preserves both as independently gated exact host extensions.",
    }];

pub const RANDI_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "R = randi(integer_imax,...) or randi(integer_[imin imax],...)",
        inputs: &RANDI_INTEGER_BOUNDS_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Bounds retain signedness and full width through range validation and unbiased sampling.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "R = randi(bounds,integer_n[,integer_sz2,...]) or randi(bounds,integer_sz)",
        inputs: &RANDI_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Dimensions are exact allocation controls and never determine sampled payload values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "R = randi(bounds,...,integer_typename) or randi(bounds,...,'like',integer_prototype)",
        inputs: &RANDI_INTEGER_PROTOTYPE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Result class and supported residency follow the documented class/prototype; sampled values outside that class saturate during conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "R = randi(bounds,...,'int64'|'uint64') or randi(bounds,...,'like',wide_integer_prototype)",
        inputs: &RANDI_WIDE_INTEGER_PROTOTYPE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The stable randi-wide-integer-output gate preserves exact native int64/uint64 generation without presenting those classes as MATLAB-compatible output forms.",
    },
];
