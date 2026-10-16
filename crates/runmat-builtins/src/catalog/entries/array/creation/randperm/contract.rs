//! Canonical randperm signatures, errors, extensions, and integer forms.
use crate::*;

pub const RANDPERM_EXPLICIT_DOUBLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "randperm-explicit-double",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "the explicit randperm \"double\" selector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RandpermExplicitDoubleExtension"),
    };

pub const RANDPERM_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "randperm-like",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "the randperm \"like\" prototype selector is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandpermLikeExtension"),
};

pub const RANDPERM_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [RANDPERM_EXPLICIT_DOUBLE_EXTENSION, RANDPERM_LIKE_EXTENSION];

const RANDPERM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector containing permutation values.",
}];

const RANDPERM_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Population size.",
}];

const RANDPERM_SIG_NK_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Population size.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Number of selected elements.",
    },
];

const RANDPERM_SIG_N_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Population size.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Class override ('double').",
    },
];

const RANDPERM_SIG_NK_CLASS_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Population size.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Number of selected elements.",
    },
    BuiltinParamDescriptor {
        name: "typename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"double\""),
        description: "Class override ('double').",
    },
];

const RANDPERM_SIG_N_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Population size.",
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

const RANDPERM_SIG_NK_LIKE_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Population size.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Number of selected elements.",
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

const RANDPERM_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "p = randperm(n)",
        inputs: &RANDPERM_SIG_N_INPUTS,
        outputs: &RANDPERM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "p = randperm(n, k)",
        inputs: &RANDPERM_SIG_NK_INPUTS,
        outputs: &RANDPERM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "p = randperm(n, \"double\")",
        inputs: &RANDPERM_SIG_N_CLASS_INPUTS,
        outputs: &RANDPERM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "p = randperm(n, k, \"double\")",
        inputs: &RANDPERM_SIG_NK_CLASS_INPUTS,
        outputs: &RANDPERM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "p = randperm(n, \"like\", prototype)",
        inputs: &RANDPERM_SIG_N_LIKE_INPUTS,
        outputs: &RANDPERM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "p = randperm(n, k, \"like\", prototype)",
        inputs: &RANDPERM_SIG_NK_LIKE_INPUTS,
        outputs: &RANDPERM_OUTPUT,
    },
];

const RANDPERM_ERRORS: [BuiltinErrorDescriptor; 8] = [
    BuiltinErrorDescriptor {
        code: "RM.RANDPERM.MISSING_N",
        identifier: None,
        when: "No N argument is provided.",
        message: "randperm: requires at least one input argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDPERM.INVALID_SIZE_ARG",
        identifier: None,
        when: "N or K is not a supported non-negative integer scalar.",
        message: "randperm: N/K must be non-negative integers",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDPERM.INVALID_K",
        identifier: None,
        when: "K is larger than N.",
        message: "randperm: K must satisfy 0 <= K <= N",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDPERM.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "randperm: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDPERM.MULTIPLE_LIKE",
        identifier: None,
        when: "The 'like' keyword is provided multiple times.",
        message: "randperm: duplicate 'like' prototype specified",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDPERM.CLASS_CONFLICT",
        identifier: None,
        when: "A class keyword and a 'like' prototype are both provided.",
        message: "randperm: cannot combine 'double' with a 'like' prototype",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDPERM.UNSUPPORTED_CLASS",
        identifier: None,
        when: "An unsupported output class is requested.",
        message: "randperm: single precision output is not implemented yet",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDPERM.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not recognized.",
        message: "randperm: unrecognised option",
    },
];

pub const RANDPERM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RANDPERM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RANDPERM_ERRORS,
};

const RANDPERM_INTEGER_COUNT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n or k",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Both counts accept all eight documented integer classes and are decoded exactly before bounded allocation.",
    }];

pub const RANDPERM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "p = randperm(integer_n[,integer_k])",
        inputs: &RANDPERM_INTEGER_COUNT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Ordinary calls return a double row vector; the one-input GPU form is separately resident, while explicit double/like selectors remain declared RunMat extensions.",
    }];
