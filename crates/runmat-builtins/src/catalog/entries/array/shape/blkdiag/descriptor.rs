use crate::*;

const BLKDIAG_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Block diagonal matrix containing each input along the diagonal.",
}];

const BLKDIAG_INPUTS_EMPTY: [BuiltinParamDescriptor; 0] = [];

const BLKDIAG_INPUTS_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First scalar, vector, matrix, sparse matrix, or gpuArray block.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional blocks placed down and right from the previous block.",
    },
];

const BLKDIAG_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = blkdiag()",
        inputs: &BLKDIAG_INPUTS_EMPTY,
        outputs: &BLKDIAG_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "B = blkdiag(A1, An...)",
        inputs: &BLKDIAG_INPUTS_VARIADIC,
        outputs: &BLKDIAG_OUTPUTS,
    },
];

pub const BLKDIAG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLKDIAG.INVALID_INPUT",
    identifier: Some("RunMat:blkdiag:InvalidInput"),
    when: "An input class, dimensionality, or block conversion is unsupported.",
    message: "blkdiag: invalid input",
};

pub const BLKDIAG_ERROR_SIZE_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLKDIAG.SIZE_OVERFLOW",
    identifier: Some("RunMat:blkdiag:SizeOverflow"),
    when: "The block diagonal output dimensions or allocation size exceed addressable memory.",
    message: "blkdiag: output size exceeds maximum supported size",
};

pub const BLKDIAG_ERROR_GPU: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BLKDIAG.GPU",
    identifier: Some("RunMat:blkdiag:GpuError"),
    when: "A required gpuArray gather or upload operation fails.",
    message: "blkdiag: gpuArray operation failed",
};

const BLKDIAG_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BLKDIAG_ERROR_INVALID_INPUT,
    BLKDIAG_ERROR_SIZE_OVERFLOW,
    BLKDIAG_ERROR_GPU,
];

pub const BLKDIAG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BLKDIAG_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BLKDIAG_ERRORS,
};

pub const BLKDIAG_SPARSE_INTEGER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "blkdiag-sparse-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "blkdiag with sparse integer blocks is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BlkdiagSparseIntegerInputExtension"),
    };

pub const BLKDIAG_COMPLEX_INTEGER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "blkdiag-complex-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "blkdiag with complex integer blocks is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:BlkdiagComplexIntegerInputExtension"),
    };

pub const BLKDIAG_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    BLKDIAG_SPARSE_INTEGER_EXTENSION,
    BLKDIAG_COMPLEX_INTEGER_EXTENSION,
];

pub const BLKDIAG_DENSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A1,...,AN",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Dense scalar, vector, or matrix blocks accept every integer class; when classes differ, the first integer block determines output class and floating or logical blocks convert into it.",
    }];

pub const BLKDIAG_SPARSE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sparse A1,...,AN",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat can structurally assemble same-class sparse integer blocks while preserving their exact values; the public sparse numeric storage domain is single, double, or logical.",
    }];

pub const BLKDIAG_COMPLEX_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "complex A1,...,AN",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat can structurally assemble same-class paired integer components; the public integer domain does not support complex-integer operations.",
    }];

pub const BLKDIAG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = blkdiag(A1,...,AN) with dense real integer blocks",
        inputs: &BLKDIAG_DENSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Blocks retain matrix geometry and occupy diagonal offsets. If any dense real input is integer, the first integer input class dominates all dense real and logical blocks; otherwise ordinary floating/logical rules apply. GPU inputs gather exactly and dense results return to the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = blkdiag(A1,...,AN) with sparse integer blocks",
        inputs: &BLKDIAG_SPARSE_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only same-class sparse integer assembly preserves exact CSC storage; sparse results remain host-resident.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = blkdiag(A1,...,AN) with complex integer blocks",
        inputs: &BLKDIAG_COMPLEX_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat-only same-class paired integer-component assembly preserves exact real and imaginary storage.",
    },
];
