//! Canonical public contract for numeric normalization.
use crate::*;

const PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Floating input data; native integer and logical data are gated RunMat extensions.",
};

pub const NORMALIZE_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "normalize-integer-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "normalize with native-class integer data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:NormalizeIntegerDataExtension"),
    };

pub const NORMALIZE_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "normalize-logical-data",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "normalize with logical data is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:NormalizeLogicalDataExtension"),
    };

pub const NORMALIZE_RESIDENT_CONTROL_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "normalize-resident-control",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "normalize with explicit gpuArray control or parameter values is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:NormalizeResidentControlExtension"),
    };

pub const NORMALIZE_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    NORMALIZE_INTEGER_DATA_EXTENSION,
    NORMALIZE_LOGICAL_DATA_EXTENSION,
    NORMALIZE_RESIDENT_CONTROL_EXTENSION,
];

const INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Native integer data is gated by normalize-integer-data and enters the documented double normalization domain from authoritative storage.",
    }];

const INTEGER_DIM_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented positive integer scalar operating dimension is parsed exactly from every native integer class.",
    }];

const INTEGER_PARAMETER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "methodtype, centertype, or scaletype",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Documented numeric p-norm, range, center, and scale parameters retain exact native storage until the selected floating normalization boundary.",
    }];

pub const NORMALIZE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "N = normalize(integer_A, ___)",
        inputs: &INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer A is a RunMat-only extension and returns double normalized values; fallback restores through the exact owner when it can preserve double, otherwise automatic residency may remain host and explicit residency errors.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "N = normalize(A, integer_dim, ___)",
        inputs: &INTEGER_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Integer dimensions select an axis without conversion; explicit resident controls are separately gated while automatic controls may gather.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "N = normalize(A, method, integer_methodtype)",
        inputs: &INTEGER_PARAMETER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Numeric normalization parameters are authoritative until their method-specific floating calculation; output precision follows A.",
    },
];

const PARAM_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Dimension to operate along.",
};

const PARAM_METHOD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "method",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"zscore\""),
    description: "Normalization method: zscore, norm, scale, range, center, or medianiqr.",
};

const PARAM_METHODTYPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "methodtype",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Method subtype or explicit center/scale/range parameter.",
};

const PARAM_CENTER_KEYWORD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "center_keyword",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"center\""),
    description: "Literal center method selector.",
};

const PARAM_CENTER_TYPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "centertype",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Centering mode or explicit center values.",
};

const PARAM_SCALE_KEYWORD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "scale_keyword",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"scale\""),
    description: "Literal scale method selector.",
};

const PARAM_SCALE_TYPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "scaletype",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scaling mode or explicit scale values.",
};

const OUTPUT_N: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "N",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Normalized data.",
}];

const OUTPUT_NCS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalized data.",
    },
    BuiltinParamDescriptor {
        name: "C",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Centering values.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Scale values.",
    },
];

const INPUTS_A: [BuiltinParamDescriptor; 1] = [PARAM_A];
const INPUTS_A_DIM: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_DIM];
const INPUTS_A_METHOD: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_METHOD];
const INPUTS_A_DIM_METHOD: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_DIM, PARAM_METHOD];
const INPUTS_A_METHOD_TYPE: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_METHOD, PARAM_METHODTYPE];
const INPUTS_A_DIM_METHOD_TYPE: [BuiltinParamDescriptor; 4] =
    [PARAM_A, PARAM_DIM, PARAM_METHOD, PARAM_METHODTYPE];
const INPUTS_A_CENTER_SCALE: [BuiltinParamDescriptor; 5] = [
    PARAM_A,
    PARAM_CENTER_KEYWORD,
    PARAM_CENTER_TYPE,
    PARAM_SCALE_KEYWORD,
    PARAM_SCALE_TYPE,
];
const INPUTS_A_DIM_CENTER_SCALE: [BuiltinParamDescriptor; 6] = [
    PARAM_A,
    PARAM_DIM,
    PARAM_CENTER_KEYWORD,
    PARAM_CENTER_TYPE,
    PARAM_SCALE_KEYWORD,
    PARAM_SCALE_TYPE,
];

const SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "N = normalize(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, dim)",
        inputs: &INPUTS_A_DIM,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, method)",
        inputs: &INPUTS_A_METHOD,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, dim, method)",
        inputs: &INPUTS_A_DIM_METHOD,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, method, methodtype)",
        inputs: &INPUTS_A_METHOD_TYPE,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, dim, method, methodtype)",
        inputs: &INPUTS_A_DIM_METHOD_TYPE,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, \"center\", centertype, \"scale\", scaletype)",
        inputs: &INPUTS_A_CENTER_SCALE,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "N = normalize(A, dim, \"center\", centertype, \"scale\", scaletype)",
        inputs: &INPUTS_A_DIM_CENTER_SCALE,
        outputs: &OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "[N, C, S] = normalize(___)",
        inputs: &INPUTS_A,
        outputs: &OUTPUT_NCS,
    },
];

pub const NORMALIZE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.normalize.INVALID_ARGUMENT",
    identifier: Some("RunMat:normalize:InvalidArgument"),
    when: "Inputs, dimension, method, method type, or name-value options are malformed.",
    message: "normalize: invalid argument",
};

pub const NORMALIZE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.normalize.INTERNAL",
    identifier: Some("RunMat:normalize:Internal"),
    when: "Internal tensor conversion or allocation fails.",
    message: "normalize: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 2] =
    [NORMALIZE_ERROR_INVALID_ARGUMENT, NORMALIZE_ERROR_INTERNAL];

pub const NORMALIZE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
