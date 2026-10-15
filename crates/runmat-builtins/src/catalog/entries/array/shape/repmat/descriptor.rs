use crate::*;

pub const REPMAT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Replicated array tiled by requested replication factors.",
}];

pub const REPMAT_INPUTS_A_R: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array to replicate.",
    },
    BuiltinParamDescriptor {
        name: "r",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Replication vector or scalar factor.",
    },
];

pub const REPMAT_INPUTS_A_M_N: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array to replicate.",
    },
    BuiltinParamDescriptor {
        name: "m",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row replication factor.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Column replication factor.",
    },
];

pub const REPMAT_INPUTS_A_DIMS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array to replicate.",
    },
    BuiltinParamDescriptor {
        name: "dims",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Per-dimension replication factors.",
    },
];

pub const REPMAT_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "B = repmat(A, r)",
        inputs: &REPMAT_INPUTS_A_R,
        outputs: &REPMAT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = repmat(A, m, n)",
        inputs: &REPMAT_INPUTS_A_M_N,
        outputs: &REPMAT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = repmat(A, d1, d2, ...)",
        inputs: &REPMAT_INPUTS_A_DIMS,
        outputs: &REPMAT_OUTPUT,
    },
];

pub const REPMAT_ERROR_MISSING_FACTORS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPMAT.MISSING_FACTORS",
    identifier: Some("RunMat:repmat:MissingReplicationFactors"),
    when: "No replication factors are provided.",
    message: "repmat: replication factors must be specified",
};

pub const REPMAT_ERROR_INVALID_FACTORS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPMAT.INVALID_FACTORS",
    identifier: Some("RunMat:repmat:InvalidReplicationFactors"),
    when: "Replication factors are non-scalar, non-integer, negative, or otherwise invalid.",
    message: "repmat: invalid replication factors",
};

pub const REPMAT_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPMAT.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:repmat:UnsupportedInput"),
    when: "Input type is unsupported for repmat.",
    message: "repmat: unsupported input type",
};

pub const REPMAT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPMAT.INTERNAL",
    identifier: Some("RunMat:repmat:Internal"),
    when: "Internal allocation/indexing/provider path fails.",
    message: "repmat: internal operation failed",
};

pub const REPMAT_ERRORS: [BuiltinErrorDescriptor; 4] = [
    REPMAT_ERROR_MISSING_FACTORS,
    REPMAT_ERROR_INVALID_FACTORS,
    REPMAT_ERROR_UNSUPPORTED_INPUT,
    REPMAT_ERROR_INTERNAL,
];

pub const REPMAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REPMAT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REPMAT_ERRORS,
};

pub const REPMAT_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents all eight integer classes and class-preserving output.",
    }];

pub const REPMAT_INTEGER_FACTOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "r/m/n/dims",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed replication factors are exact nonnegative structural controls.",
    }];

pub const REPMAT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = repmat(integer_A, r)",
        inputs: &REPMAT_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Tiling copies authoritative native storage exactly and restores supported resident results through the owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = repmat(A, integer_r) or repmat(A, integer_m,integer_n,...)",
        inputs: &REPMAT_INTEGER_FACTOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Factors are never widened to binary64; checked output-size arithmetic rejects overflow.",
    },
];
