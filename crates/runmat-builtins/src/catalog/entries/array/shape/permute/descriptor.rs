use crate::*;

pub const PERMUTE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array with dimensions reordered by order.",
}];

pub const PERMUTE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array/value.",
    },
    BuiltinParamDescriptor {
        name: "order",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Row/column vector containing a permutation of 1:ndims(A).",
    },
];

pub const PERMUTE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "B = permute(A, order)",
    inputs: &PERMUTE_INPUTS,
    outputs: &PERMUTE_OUTPUT,
}];

pub const PERMUTE_ERROR_INVALID_ORDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERMUTE.INVALID_ORDER",
    identifier: Some("RunMat:permute:InvalidOrder"),
    when: "order is non-numeric, non-vector, empty, non-integer, out of range, or duplicated.",
    message: "permute: invalid order vector",
};

pub const PERMUTE_ERROR_RANK_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERMUTE.RANK_MISMATCH",
    identifier: Some("RunMat:permute:RankMismatch"),
    when: "order length is less than ndims(A).",
    message: "permute: order length must be at least ndims(A)",
};

pub const PERMUTE_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PERMUTE.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:permute:UnsupportedInput"),
    when: "A has unsupported type for permute.",
    message: "permute: unsupported input type",
};

pub const PERMUTE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    PERMUTE_ERROR_INVALID_ORDER,
    PERMUTE_ERROR_RANK_MISMATCH,
    PERMUTE_ERROR_UNSUPPORTED_INPUT,
];

pub const PERMUTE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PERMUTE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PERMUTE_ERRORS,
};

pub const PERMUTE_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The generic array contract includes every integer class. Permutation only reorders authoritative elements and preserves class exactly.",
    }];

pub const PERMUTE_INTEGER_ORDER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dimorder",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public row-vector contract requires unique positive integer elements. RunMat decodes every supported integer class exactly.",
    }];

pub const PERMUTE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = permute(integer_A, dimorder)",
        inputs: &PERMUTE_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host, paired complex-integer, and provider-resident paths preserve exact class, shape, owner, and explicit residency while changing only layout.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = permute(A, integer_dimorder)",
        inputs: &PERMUTE_INTEGER_ORDER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Dimension order is parsed exactly as one-based usize values and validated as a complete permutation before data or provider dispatch.",
    },
];
