use crate::*;
pub(super) const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dA",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix decomposition object.",
}];

pub(super) const SOLVE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric solution.",
}];

const CONSTRUCTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Coefficient matrix, optional type, triangular flag, and name-value options.",
}];

pub(super) const METHOD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left operand.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right operand.",
    },
];

pub(super) const UNARY_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dA",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix decomposition object.",
}];

pub(super) const SUBSREF_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Matrix decomposition object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct indexing path.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "dA = decomposition(A)",
        inputs: &CONSTRUCTOR_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dA = decomposition(A, type)",
        inputs: &CONSTRUCTOR_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dA = decomposition(A, type, triangularFlag)",
        inputs: &CONSTRUCTOR_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dA = decomposition(___, Name, Value)",
        inputs: &CONSTRUCTOR_INPUTS,
        outputs: &OUTPUT,
    },
];

const BINARY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = decomposition.method(lhs, rhs)",
    inputs: &METHOD_INPUTS,
    outputs: &OUTPUT,
}];

const SOLVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = decomposition division(lhs, rhs)",
    inputs: &METHOD_INPUTS,
    outputs: &OUTPUT,
}];

const UNARY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = decomposition.method(dA)",
    inputs: &UNARY_INPUT,
    outputs: &OUTPUT,
}];

const SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = decomposition.subsref(obj, S)",
    inputs: &SUBSREF_INPUTS,
    outputs: &OUTPUT,
}];

pub const DECOMPOSITION_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECOMPOSITION.INVALID_INPUT",
    identifier: Some("RunMat:decomposition:InvalidInput"),
    when: "The matrix, type, option, or object operation is invalid.",
    message: "decomposition: invalid input",
};

pub const DECOMPOSITION_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECOMPOSITION.INTERNAL",
    identifier: Some("RunMat:decomposition:Internal"),
    when: "Runtime cannot materialize a decomposition object or transformed coefficient matrix.",
    message: "decomposition: internal runtime failure",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [
    DECOMPOSITION_ERROR_INVALID_INPUT,
    DECOMPOSITION_ERROR_INTERNAL,
];

pub const DECOMPOSITION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DECOMPOSITION_BINARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BINARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DECOMPOSITION_SOLVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SOLVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DECOMPOSITION_UNARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DECOMPOSITION_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
