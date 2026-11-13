//! Canonical rand signatures, errors, extensions, and integer forms.
use crate::*;

const RAND_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Uniform random array in [0,1).",
}];

const RAND_SEED_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "seed",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Legacy restorable RNG state token.",
}];

const RAND_NO_OUTPUTS: [BuiltinParamDescriptor; 0] = [];

const RAND_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const RAND_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const RAND_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const RAND_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const RAND_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
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
        description: "Class override ('double'|'single'|'gpuArray').",
    },
];

const RAND_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
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

const RAND_SIG_SEED_QUERY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "seed_option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"seed\""),
    description: "Legacy seed query option.",
}];

const RAND_SIG_SEED_SET_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "seed_option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"seed\""),
        description: "Legacy seed control option.",
    },
    BuiltinParamDescriptor {
        name: "seed",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Non-negative integer seed.",
    },
];

const RAND_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "A = rand()",
        inputs: &RAND_SIG_EMPTY_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = rand(n)",
        inputs: &RAND_SIG_N_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = rand(size_vector)",
        inputs: &RAND_SIG_SIZE_VECTOR_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = rand(m, n, ...)",
        inputs: &RAND_SIG_DIMS_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = rand(..., typename)",
        inputs: &RAND_SIG_CLASS_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = rand(..., \"like\", prototype)",
        inputs: &RAND_SIG_LIKE_INPUTS,
        outputs: &RAND_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "seed = rand(\"seed\")",
        inputs: &RAND_SIG_SEED_QUERY_INPUTS,
        outputs: &RAND_SEED_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "rand(\"seed\", seed)",
        inputs: &RAND_SIG_SEED_SET_INPUTS,
        outputs: &RAND_NO_OUTPUTS,
    },
];

pub const RAND_COLUMN_SIZE_VECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "rand-column-size-vector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "rand with a column size vector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RandColumnSizeVectorExtension"),
    };

pub const RAND_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rand-resident-size-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rand with a resident size control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandResidentSizeControlExtension"),
};

pub const RAND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    RAND_COLUMN_SIZE_VECTOR_EXTENSION,
    RAND_RESIDENT_SIZE_EXTENSION,
];

const RAND_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz1...szN/sz",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are exact structural size controls; negative signed values clamp to zero and trailing singleton dimensions normalize away.",
    }];

const RAND_INTEGER_SEED_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "seed",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The discouraged legacy seed syntax accepts a nonnegative exact integer scalar within the runtime's restorable seed-token domain.",
    }];

pub const RAND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "X = rand(integer_n[, integer_sz2, ...])",
        inputs: &RAND_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The default output is host double; typename can select single and explicit gpuArray syntax selects residency.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = rand(integer_sz)",
        inputs: &RAND_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented size vector is a row vector of exact integer values.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "rand('seed', integer_seed)",
        inputs: &RAND_INTEGER_SEED_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The legacy control reads all eight native integer scalar classes exactly, rejects negatives and values outside its restorable token domain, updates the shared stream, and synchronizes the active provider when supported. Generator sequence/state parity is a general RNG-engine conformance gap rather than an implicit integer conversion boundary.",
    },
];

const RAND_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.RAND.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "rand: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.RAND.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not supported.",
        message: "rand: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.RAND.UNSUPPORTED_PROTOTYPE",
        identifier: None,
        when: "A prototype type cannot be used for rand(..., 'like', prototype).",
        message: "rand: unsupported prototype",
    },
    BuiltinErrorDescriptor {
        code: "RM.RAND.INVALID_DIMS",
        identifier: None,
        when: "Dimension arguments fail numeric/shape parsing.",
        message: "rand: dimension arguments must be numeric and nonnegative",
    },
];

pub const RAND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RAND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RAND_ERRORS,
};
