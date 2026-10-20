use crate::*;

pub const REPELEM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Array with element-wise replication applied.",
}];

pub const REPELEM_INPUTS_A_R: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Scalar or vector replication control argument.",
    },
];

pub const REPELEM_INPUTS_A_RN: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "R1_RN",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Per-dimension scalar/vector replication controls.",
    },
];

pub const REPELEM_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = repelem(A, R)",
        inputs: &REPELEM_INPUTS_A_R,
        outputs: &REPELEM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = repelem(A, R1, R2, ...)",
        inputs: &REPELEM_INPUTS_A_RN,
        outputs: &REPELEM_OUTPUT,
    },
];

pub const REPELEM_ERROR_MISSING_FACTORS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPELEM.MISSING_FACTORS",
    identifier: Some("RunMat:repelem:MissingReplicationFactors"),
    when: "No replication arguments are provided.",
    message: "repelem: replication factors must be specified",
};

pub const REPELEM_ERROR_INVALID_FACTORS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPELEM.INVALID_FACTORS",
    identifier: Some("RunMat:repelem:InvalidReplicationFactors"),
    when: "Replication arguments are non-numeric, non-integer, or shape-incompatible.",
    message: "repelem: invalid replication factors",
};

pub const REPELEM_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPELEM.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:repelem:UnsupportedInput"),
    when: "Input type is unsupported for repelem.",
    message: "repelem: unsupported input type",
};

pub const REPELEM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPELEM.INTERNAL",
    identifier: Some("RunMat:repelem:Internal"),
    when: "Internal allocation/indexing path fails.",
    message: "repelem: internal operation failed",
};

pub const REPELEM_ERRORS: [BuiltinErrorDescriptor; 4] = [
    REPELEM_ERROR_MISSING_FACTORS,
    REPELEM_ERROR_INVALID_FACTORS,
    REPELEM_ERROR_UNSUPPORTED_INPUT,
    REPELEM_ERROR_INTERNAL,
];

pub const REPELEM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REPELEM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REPELEM_ERRORS,
};
pub const REPELEM_INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents all eight integer classes; authoritative elements are copied without arithmetic or floating conversion.",
    }];
pub const REPELEM_INTEGER_FACTOR_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "R1,...,RN",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed repetition controls are read exactly as nonnegative structural sizes.",
    }];
pub const REPELEM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = repelem(integer_A, R1,...,RN)",
        inputs: &REPELEM_INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Replication preserves native class, shape order, and exact wide values; resident inputs use owner-preserving host fallback when no provider hook exists.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = repelem(A, integer_R1,...,integer_RN)",
        inputs: &REPELEM_INTEGER_FACTOR_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Replication counts never pass through binary64; checked size arithmetic rejects unrepresentable output geometry.",
    },
];
