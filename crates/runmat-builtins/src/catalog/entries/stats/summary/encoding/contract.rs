//! Public signatures, errors, compatibility gates, and integer forms.
use crate::*;

const GROUP: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "group",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grouping variables.",
};
const LABELS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Labels to encode.",
};
const ENCODED: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Encoded numeric or logical array.",
};
const CLASSES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "classes",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Ordered class names.",
};
const DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "featureDim",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-based feature-vector dimension.",
};
const TYPENAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "typename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Requested output class.",
};
const OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value ClassNames and output-type controls.",
};
const DUMMY_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double dummy-variable design matrix.",
}];
const ENCODE_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-hot encoded array.",
}];
const DECODE_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Decoded class labels.",
}];
const DUMMY_IN: [BuiltinParamDescriptor; 1] = [GROUP];
const ENCODE_IN: [BuiltinParamDescriptor; 2] = [LABELS, DIM];
const ENCODE_TYPED_IN: [BuiltinParamDescriptor; 3] = [LABELS, DIM, TYPENAME];
const ENCODE_OPTIONS_IN: [BuiltinParamDescriptor; 3] = [LABELS, DIM, OPTIONS];
const DECODE_IN: [BuiltinParamDescriptor; 3] = [ENCODED, CLASSES, DIM];
const DECODE_TYPED_IN: [BuiltinParamDescriptor; 4] = [ENCODED, CLASSES, DIM, TYPENAME];
const DECODE_OPTIONS_IN: [BuiltinParamDescriptor; 4] = [ENCODED, CLASSES, DIM, OPTIONS];

const DUMMY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "D = dummyvar(group)",
    inputs: &DUMMY_IN,
    outputs: &DUMMY_OUT,
}];
const ENCODE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "B = onehotencode(A, featureDim)",
        inputs: &ENCODE_IN,
        outputs: &ENCODE_OUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = onehotencode(A, featureDim, typename)",
        inputs: &ENCODE_TYPED_IN,
        outputs: &ENCODE_OUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = onehotencode(A, featureDim, Name, Value)",
        inputs: &ENCODE_OPTIONS_IN,
        outputs: &ENCODE_OUT,
    },
];
const DECODE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "A = onehotdecode(B, classes, featureDim)",
        inputs: &DECODE_IN,
        outputs: &DECODE_OUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = onehotdecode(B, classes, featureDim, typename)",
        inputs: &DECODE_TYPED_IN,
        outputs: &DECODE_OUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = onehotdecode(B, classes, featureDim, Name, Value)",
        inputs: &DECODE_OPTIONS_IN,
        outputs: &DECODE_OUT,
    },
];

pub const DUMMYVAR_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.dummyvar.INVALID_ARGUMENT",
    identifier: Some("RunMat:dummyvar:InvalidArgument"),
    when: "Grouping values or geometry are invalid.",
    message: "dummyvar: invalid argument",
};
pub const DUMMYVAR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.dummyvar.INTERNAL",
    identifier: Some("RunMat:dummyvar:Internal"),
    when: "Internal allocation or conversion fails.",
    message: "dummyvar: internal error",
};
pub const ONEHOTENCODE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.onehotencode.INVALID_ARGUMENT",
    identifier: Some("RunMat:onehotencode:InvalidArgument"),
    when: "Labels, classes, dimension, output type, or options are invalid.",
    message: "onehotencode: invalid argument",
};
pub const ONEHOTENCODE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.onehotencode.INTERNAL",
    identifier: Some("RunMat:onehotencode:Internal"),
    when: "Internal allocation or conversion fails.",
    message: "onehotencode: internal error",
};
pub const ONEHOTDECODE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.onehotdecode.INVALID_ARGUMENT",
    identifier: Some("RunMat:onehotdecode:InvalidArgument"),
    when: "Encoded data, classes, dimension, output type, or options are invalid.",
    message: "onehotdecode: invalid argument",
};
pub const ONEHOTDECODE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.onehotdecode.INTERNAL",
    identifier: Some("RunMat:onehotdecode:Internal"),
    when: "Internal allocation or conversion fails.",
    message: "onehotdecode: internal error",
};

/// The owning public identity for shared encoding diagnostics. Public names
/// are presentation only; descriptor selection is exhaustive and typed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncodingBuiltin {
    Dummyvar,
    OnehotEncode,
    OnehotDecode,
}

impl EncodingBuiltin {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dummyvar => "dummyvar",
            Self::OnehotEncode => "onehotencode",
            Self::OnehotDecode => "onehotdecode",
        }
    }

    pub const fn invalid(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Dummyvar => &DUMMYVAR_ERROR_INVALID,
            Self::OnehotEncode => &ONEHOTENCODE_ERROR_INVALID,
            Self::OnehotDecode => &ONEHOTDECODE_ERROR_INVALID,
        }
    }

    pub const fn internal(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Dummyvar => &DUMMYVAR_ERROR_INTERNAL,
            Self::OnehotEncode => &ONEHOTENCODE_ERROR_INTERNAL,
            Self::OnehotDecode => &ONEHOTDECODE_ERROR_INTERNAL,
        }
    }
}

impl std::fmt::Display for EncodingBuiltin {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}
const DUMMY_ERRORS: [BuiltinErrorDescriptor; 2] = [DUMMYVAR_ERROR_INVALID, DUMMYVAR_ERROR_INTERNAL];
const ENCODE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ONEHOTENCODE_ERROR_INVALID, ONEHOTENCODE_ERROR_INTERNAL];
const DECODE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [ONEHOTDECODE_ERROR_INVALID, ONEHOTDECODE_ERROR_INTERNAL];
pub const DUMMYVAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DUMMY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DUMMY_ERRORS,
};
pub const ONEHOTENCODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ENCODE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ENCODE_ERRORS,
};
pub const ONEHOTDECODE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DECODE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DECODE_ERRORS,
};

pub const DUMMYVAR_INTEGER_GROUP_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "dummyvar-integer-group",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "dummyvar with typed-integer grouping variables is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DummyvarIntegerGroupExtension"),
    };
pub const DUMMYVAR_GPU_GROUP_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "dummyvar-gpu-group",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "dummyvar with a resident grouping variable is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DummyvarGpuGroupExtension"),
};
pub const DUMMYVAR_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DUMMYVAR_INTEGER_GROUP_EXTENSION,
    DUMMYVAR_GPU_GROUP_EXTENSION,
];
pub const ONEHOTENCODE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "onehotencode-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "onehotencode host fallback for explicit gpuArray inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OnehotencodeExplicitGpuInputExtension"),
    };
pub const ONEHOTENCODE_OUTPUT_TYPE_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "onehotencode-outputtype-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "onehotencode OutputType/Typename name-value aliases are RunMat extensions",
        error_identifier: Some("RunMat:compatibility:OnehotencodeOutputTypeAliasExtension"),
    };
pub const ONEHOTENCODE_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ONEHOTENCODE_EXPLICIT_GPU_EXTENSION,
    ONEHOTENCODE_OUTPUT_TYPE_ALIAS_EXTENSION,
];
pub const ONEHOTDECODE_BROAD_ENCODED_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "onehotdecode-integer-logical-encoded-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "onehotdecode with integer or logical encoded probability data is a RunMat extension",
        error_identifier: Some(
            "RunMat:compatibility:OnehotdecodeIntegerLogicalEncodedInputExtension",
        ),
    };
pub const ONEHOTDECODE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "onehotdecode-explicit-gpu-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "onehotdecode host fallback for explicit gpuArray inputs is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:OnehotdecodeExplicitGpuInputExtension"),
    };
pub const ONEHOTDECODE_LOGICAL_CELL_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "onehotdecode-logical-cell-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "onehotdecode logical or cellstr output types are RunMat extensions",
        error_identifier: Some("RunMat:compatibility:OnehotdecodeLogicalCellOutputExtension"),
    };
pub const ONEHOTDECODE_OUTPUT_TYPE_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "onehotdecode-outputtype-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "onehotdecode OutputType/Typename name-value aliases are RunMat extensions",
        error_identifier: Some("RunMat:compatibility:OnehotdecodeOutputTypeAliasExtension"),
    };
pub const ONEHOTDECODE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    ONEHOTDECODE_BROAD_ENCODED_EXTENSION,
    ONEHOTDECODE_EXPLICIT_GPU_EXTENSION,
    ONEHOTDECODE_LOGICAL_CELL_OUTPUT_EXTENSION,
    ONEHOTDECODE_OUTPUT_TYPE_ALIAS_EXTENSION,
];

const ALL_INTEGER: &[BuiltinIntegerClass] = &crate::ALL_INTEGER_CLASSES;
const DUMMY_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "group", classes: ALL_INTEGER, availability: BuiltinIntegerInputAvailability::RunMatOnly, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "All eight integer classes are admitted as exact positive grouping levels in RunMat mode; output is bounded before floating conversion." }];
const ENCODE_LABEL_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A/ClassNames",
    classes: ALL_INTEGER,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integer labels and classes retain native exact identity above flintmax.",
}];
const FEATURE_DIM_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "featureDim",
    classes: ALL_INTEGER,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Exact positive singleton dimension is validated before shape arithmetic.",
}];
const DECODE_CLASSES_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "classes",
    classes: ALL_INTEGER,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Exact class labels survive selection and requested native output conversion.",
}];
const DECODE_ENCODED_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "B",
    classes: ALL_INTEGER,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Integer zero-one encoded data is a checked RunMat-only extension.",
}];
pub const DUMMYVAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "D = dummyvar(integer_group)", inputs: &DUMMY_INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Exact positive levels select 1:max(group); oversized levels reject before floating conversion; output is host double." }];
pub const ONEHOTENCODE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = onehotencode(integer_A, featureDim, ClassNames=integer_classes)",
        inputs: &ENCODE_LABEL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Exact class matching yields zero-one values; missing labels need floating output.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = onehotencode(A, integer_featureDim, ...)",
        inputs: &FEATURE_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Exact one-based dimension controls output geometry.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = onehotencode(..., integer_typename)",
        inputs: &[],
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "All signed and unsigned integer typenames construct native zero-one storage.",
    },
];
pub const ONEHOTDECODE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "A = onehotdecode(B, integer_classes, featureDim, ...)", inputs: &DECODE_CLASSES_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::OptionDependent, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Argmax chooses exact integer class labels; requested numeric conversion uses assignment casting." },
    BuiltinIntegerCapabilityDescriptor { form: "A = onehotdecode(B, classes, integer_featureDim, ...)", inputs: &FEATURE_DIM_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::OptionDependent, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Exact dimension identifies the probability-vector axis." },
    BuiltinIntegerCapabilityDescriptor { form: "A = onehotdecode(B, classes, featureDim, integer_typename)", inputs: &[], computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::OptionDependent, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Integer typename yields authoritative native storage for numeric classes." },
    BuiltinIntegerCapabilityDescriptor { form: "A = onehotdecode(integer_B, classes, featureDim, ...)", inputs: &DECODE_ENCODED_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::OptionDependent, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Checked zero-one encoded values are RunMat-only and strict modes reject before provider access." },
];
