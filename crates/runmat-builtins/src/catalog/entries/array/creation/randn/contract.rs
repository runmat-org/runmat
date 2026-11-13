//! Canonical randn signatures, errors, extensions, and integer forms.
use crate::*;

const RANDN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Normal random array (mean 0, variance 1).",
}];

const RANDN_SIG_EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];

const RANDN_SIG_N_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
}];

const RANDN_SIG_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "size_vector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector defining output dimensions.",
}];

const RANDN_SIG_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dims",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Dimension sizes.",
}];

const RANDN_SIG_CLASS_INPUTS: [BuiltinParamDescriptor; 2] = [
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

const RANDN_SIG_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
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

const RANDN_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "A = randn()",
        inputs: &RANDN_SIG_EMPTY_INPUTS,
        outputs: &RANDN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = randn(n)",
        inputs: &RANDN_SIG_N_INPUTS,
        outputs: &RANDN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = randn(size_vector)",
        inputs: &RANDN_SIG_SIZE_VECTOR_INPUTS,
        outputs: &RANDN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = randn(m, n, ...)",
        inputs: &RANDN_SIG_DIMS_INPUTS,
        outputs: &RANDN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = randn(..., typename)",
        inputs: &RANDN_SIG_CLASS_INPUTS,
        outputs: &RANDN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = randn(..., \"like\", prototype)",
        inputs: &RANDN_SIG_LIKE_INPUTS,
        outputs: &RANDN_OUTPUT,
    },
];

const RANDN_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.RANDN.LIKE_EXPECTED_PROTOTYPE",
        identifier: None,
        when: "The 'like' keyword is provided without a prototype argument.",
        message: "randn: expected prototype after 'like'",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDN.UNRECOGNIZED_OPTION",
        identifier: None,
        when: "A trailing option string is not supported.",
        message: "randn: unrecognised option",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDN.UNSUPPORTED_PROTOTYPE",
        identifier: None,
        when: "A prototype type cannot be used for randn(..., 'like', prototype).",
        message: "randn: unsupported prototype",
    },
    BuiltinErrorDescriptor {
        code: "RM.RANDN.INVALID_DIMS",
        identifier: None,
        when: "Dimension arguments fail numeric/shape parsing.",
        message: "randn: dimension arguments must be numeric and nonnegative",
    },
];

pub const RANDN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RANDN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RANDN_ERRORS,
};

pub const RANDN_COLUMN_SIZE_VECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "randn-column-size-vector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "randn with a column size vector is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:RandnColumnSizeVectorExtension"),
    };

pub const RANDN_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "randn-resident-size-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "randn with a resident size control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:RandnResidentSizeControlExtension"),
};

pub const RANDN_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    RANDN_COLUMN_SIZE_VECTOR_EXTENSION,
    RANDN_RESIDENT_SIZE_EXTENSION,
];

const RANDN_INTEGER_DIM_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n/sz1...szN/sz",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented structural sizes; signed negatives clamp to zero and trailing singleton dimensions normalize away.",
    }];

pub const RANDN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "X = randn(integer_n[,integer_sz2,...])",
        inputs: &RANDN_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Counts are decoded from authoritative storage and select double or single host/GPU generation without numeric conversion.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "X = randn(integer_sz)",
        inputs: &RANDN_INTEGER_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::FunctionSpecific,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented size vector is a row vector; integer prototypes remain unsupported because randn like supports only single/double numeric class and complexity.",
    },
];
