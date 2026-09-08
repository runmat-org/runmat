use crate::*;

const OUTPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense array assembled from the cell contents.",
}];
const INPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array containing compatible array blocks.",
}];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "A = cell2mat(C)",
    inputs: INPUTS,
    outputs: OUTPUTS,
}];

pub const CELL2MAT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2MAT.INVALID_INPUT",
    identifier: Some("RunMat:cell2mat:InvalidInput"),
    when: "The input is not a cell array.",
    message: "cell2mat: expected a cell array input",
};
pub const CELL2MAT_ERROR_INVALID_CONTENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2MAT.INVALID_CONTENTS",
    identifier: Some("RunMat:cell2mat:InvalidContents"),
    when: "The cell contents cannot form one dense array.",
    message: "cell2mat: cell contents are not compatible for concatenation",
};
pub const CELL2MAT_ERROR_SIZE_EXCEEDED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2MAT.SIZE_EXCEEDED",
    identifier: Some("RunMat:cell2mat:SizeExceeded"),
    when: "The output shape or storage exceeds platform limits.",
    message: "cell2mat: resulting matrix exceeds platform limits",
};
pub const CELL2MAT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2MAT.INTERNAL",
    identifier: None,
    when: "An internal allocation or storage invariant fails.",
    message: "cell2mat: internal error",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    CELL2MAT_ERROR_INVALID_INPUT,
    CELL2MAT_ERROR_INVALID_CONTENTS,
    CELL2MAT_ERROR_SIZE_EXCEEDED,
    CELL2MAT_ERROR_INTERNAL,
];

pub const CELL2MAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};

const INTEGER_INPUTS: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "C contents",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
    notes: "Integer contents retain native storage. Mixed classes follow compatible concatenation assignment rules.",
}];
pub const CELL2MAT_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor {
        form: "A = cell2mat(C) for real integer cell contents",
        inputs: INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "The leftmost nonempty integer block selects the mixed-integer output class; later blocks saturate into that class. Scalar doubles cannot mix with int64 or uint64.",
    }];
