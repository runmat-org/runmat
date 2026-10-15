use crate::*;

pub const IPERMUTE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Array with inverse permutation applied.",
}];

pub const IPERMUTE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Permuted input array/value.",
    },
    BuiltinParamDescriptor {
        name: "order",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Permutation vector previously used with permute.",
    },
];

pub const IPERMUTE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "A = ipermute(B, order)",
    inputs: &IPERMUTE_INPUTS,
    outputs: &IPERMUTE_OUTPUT,
}];

pub const IPERMUTE_ERROR_INVALID_ORDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IPERMUTE.INVALID_ORDER",
    identifier: Some("RunMat:ipermute:InvalidOrder"),
    when: "order is non-numeric, non-vector, empty, non-integer, out of range, or duplicated.",
    message: "ipermute: invalid order vector",
};

pub const IPERMUTE_ERROR_RANK_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IPERMUTE.RANK_MISMATCH",
    identifier: Some("RunMat:ipermute:RankMismatch"),
    when: "order length is less than ndims(B).",
    message: "ipermute: order length must be at least ndims(B)",
};

pub const IPERMUTE_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IPERMUTE.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:ipermute:UnsupportedInput"),
    when: "B has unsupported type for ipermute.",
    message: "ipermute: unsupported input type",
};

pub const IPERMUTE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    IPERMUTE_ERROR_INVALID_ORDER,
    IPERMUTE_ERROR_RANK_MISMATCH,
    IPERMUTE_ERROR_UNSUPPORTED_INPUT,
];

pub const IPERMUTE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IPERMUTE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IPERMUTE_ERRORS,
};

pub const IPERMUTE_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "B",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every integer class is permuted structurally without changing class or values.",
    }];

pub const IPERMUTE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "A = ipermute(integer_B, order)", inputs: &IPERMUTE_INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "Host and resident integer storage is reordered exactly; provider fallback restores the result to the owning provider." }];
