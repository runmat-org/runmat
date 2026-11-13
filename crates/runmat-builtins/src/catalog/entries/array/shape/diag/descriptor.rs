use crate::*;

pub const DIAG_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "diag-explicit-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "diag with an explicit output size is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DiagExplicitSizeExtension"),
};
pub const DIAG_VECTOR_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "diag-vector-option",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "diag with the 'vector' option is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DiagVectorOptionExtension"),
};
pub const DIAG_CLASS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "diag-output-class",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "diag with an output-class override is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DiagOutputClassExtension"),
};
pub const DIAG_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "diag-like",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "diag with a 'like' prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DiagLikeExtension"),
};
pub const DIAG_TRAILING_SINGLETON_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "diag-trailing-singleton-dimensions",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "diag with more than two trailing singleton dimensions is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DiagTrailingSingletonDimensionsExtension"),
    };
pub const DIAG_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    DIAG_SIZE_EXTENSION,
    DIAG_VECTOR_EXTENSION,
    DIAG_CLASS_EXTENSION,
    DIAG_LIKE_EXTENSION,
    DIAG_TRAILING_SINGLETON_EXTENSION,
];

const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "v or A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "All eight real integer classes preserve exact native storage through diagonal construction or extraction." }];
const INTEGER_OFFSET: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "v or A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Integer data remains authoritative and is never converted through binary64 for structural placement." },
    BuiltinIntegerInputCapability { name: "k", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The real scalar diagonal offset is decoded exactly from native integer storage; invalid or out-of-range controls reject before shape arithmetic." },
];
pub const DIAG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "D = diag(integer_v) or x = diag(integer_A)", inputs: &INTEGER_DATA, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Same-class vector construction and matrix extraction preserve exact integer storage; resident fallback gathers and re-uploads to the owning provider." },
    BuiltinIntegerCapabilityDescriptor { form: "D = diag(integer_v,k) or x = diag(integer_A,k)", inputs: &INTEGER_OFFSET, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "The offset controls placement or extraction only; host and resident fallback retain exact data class and native integer values." },
];

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "D or x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Constructed diagonal matrix or extracted diagonal vector.",
}];
const fn input() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "v or A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar, vector, or matrix.",
    }
}
const fn offset() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Exact signed diagonal offset.",
    }
}
const fn size() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "sz",
        ty: BuiltinParamType::SizeArg,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "RunMat output size n or [m n].",
    }
}
const INPUT: [BuiltinParamDescriptor; 1] = [input()];
const INPUT_OFFSET: [BuiltinParamDescriptor; 2] = [input(), offset()];
const INPUT_SIZE: [BuiltinParamDescriptor; 2] = [input(), size()];
const INPUT_OFFSET_SIZE: [BuiltinParamDescriptor; 3] = [input(), offset(), size()];
const INPUT_VECTOR: [BuiltinParamDescriptor; 2] = [
    input(),
    BuiltinParamDescriptor {
        name: "option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("'vector'"),
        description: "RunMat vector extraction option.",
    },
];
const INPUT_CLASS: [BuiltinParamDescriptor; 2] = [
    input(),
    BuiltinParamDescriptor {
        name: "class",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "RunMat output class override.",
    },
];
const INPUT_LIKE: [BuiltinParamDescriptor; 3] = [
    input(),
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("'like'"),
        description: "RunMat prototype option.",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Prototype controlling output class and residency.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "D = diag(v)",
        inputs: &INPUT,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = diag(v, k)",
        inputs: &INPUT_OFFSET,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "x = diag(A)",
        inputs: &INPUT,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "x = diag(A, k)",
        inputs: &INPUT_OFFSET,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = diag(v, sz)",
        inputs: &INPUT_SIZE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = diag(v, k, sz)",
        inputs: &INPUT_OFFSET_SIZE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "x = diag(A, 'vector')",
        inputs: &INPUT_VECTOR,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = diag(v, class)",
        inputs: &INPUT_CLASS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = diag(v, 'like', prototype)",
        inputs: &INPUT_LIKE,
        outputs: &OUTPUT,
    },
];
pub const DIAG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIAG.INVALID_INPUT",
    identifier: Some("RunMat:diag:InvalidInput"),
    when: "Input type, option grammar, size override, or output conversion is invalid.",
    message: "diag: invalid input argument",
};
pub const DIAG_ERROR_INVALID_OFFSET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIAG.INVALID_OFFSET",
    identifier: Some("RunMat:diag:InvalidOffset"),
    when: "Diagonal offset is not a finite integer scalar.",
    message: "diag: invalid diagonal offset",
};
const ERRORS: [BuiltinErrorDescriptor; 2] = [DIAG_ERROR_INVALID_INPUT, DIAG_ERROR_INVALID_OFFSET];
pub const DIAG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
