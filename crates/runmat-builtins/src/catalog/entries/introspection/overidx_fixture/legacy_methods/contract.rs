use crate::*;
const OVERIDX_NUM_ARGUMENTS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "The nargs property of any class when present; otherwise scalar double three.",
}];
const OVERIDX_NUM_ARGUMENTS_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "OverIdx object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Substruct-compatible indexing descriptor.",
    },
    BuiltinParamDescriptor {
        name: "indexingContext",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing context.",
    },
];
const OVERIDX_NUM_ARGUMENTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "n = OverIdx.numArgumentsFromSubscript(obj, S, indexingContext)",
        inputs: &OVERIDX_NUM_ARGUMENTS_INPUTS,
        outputs: &OVERIDX_NUM_ARGUMENTS_OUTPUT,
    }];
pub const OVERIDX_NUM_ARGUMENTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_NUM_ARGUMENTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
const OVERIDX_SAVEOBJ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "payload",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Serialized OverIdx payload.",
}];
const OVERIDX_SAVEOBJ_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "OverIdx object receiver.",
}];
const OVERIDX_SAVEOBJ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "payload = OverIdx.saveobj(obj)",
    inputs: &OVERIDX_SAVEOBJ_INPUTS,
    outputs: &OVERIDX_SAVEOBJ_OUTPUT,
}];
pub const OVERIDX_SAVEOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_SAVEOBJ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
const OVERIDX_LOADOBJ_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Restored OverIdx object.",
}];
const OVERIDX_LOADOBJ_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "payload",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Serialized OverIdx payload.",
}];
const OVERIDX_LOADOBJ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = OverIdx.loadobj(payload)",
    inputs: &OVERIDX_LOADOBJ_INPUTS,
    outputs: &OVERIDX_LOADOBJ_OUTPUT,
}];
pub const OVERIDX_LOADOBJ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_LOADOBJ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[OVERIDX_FIXTURE_ERROR_ARITY, OVERIDX_FIXTURE_ERROR_OUTPUTS],
};
const OVERIDX_BINARY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operator result value.",
}];
const OVERIDX_BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "OverIdx object receiver.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right-hand operand.",
    },
];
const OVERIDX_UNARY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Operator result value.",
}];
const OVERIDX_UNARY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "OverIdx object receiver.",
}];
const OVERIDX_PLUS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.plus(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_TIMES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.times(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_MTIMES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.mtimes(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_LT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.lt(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_GT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.gt(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_EQ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.eq(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_UPLUS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.uplus(obj)",
    inputs: &OVERIDX_UNARY_INPUTS,
    outputs: &OVERIDX_UNARY_OUTPUT,
}];
const OVERIDX_RDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.rdivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_MRDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.mrdivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_LDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.ldivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_MLDIVIDE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.mldivide(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_AND_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.and(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_OR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.or(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
const OVERIDX_XOR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = OverIdx.xor(obj, rhs)",
    inputs: &OVERIDX_BINARY_INPUTS,
    outputs: &OVERIDX_BINARY_OUTPUT,
}];
pub const OVERIDX_PLUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_PLUS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_TIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_TIMES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_MTIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_MTIMES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_LT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_LT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_GT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_GT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_EQ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_EQ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_UPLUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_UPLUS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[OVERIDX_FIXTURE_ERROR_ARITY, OVERIDX_FIXTURE_ERROR_OUTPUTS],
};
pub const OVERIDX_RDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_RDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_MRDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_MRDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_LDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_LDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_MLDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_MLDIVIDE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_AND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_AND_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_OR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_OR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_XOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OVERIDX_XOR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &[
        OVERIDX_ERROR_RECEIVER_INVALID,
        OVERIDX_FIXTURE_ERROR_ARITY,
        OVERIDX_FIXTURE_ERROR_OUTPUTS,
    ],
};
pub const OVERIDX_FIXTURE_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIXTURE_METHOD.ARITY",
    identifier: Some("RunMat:FixtureMethod:Arity"),
    when: "Argument count differs from the fixed method signature.",
    message: "fixture method: wrong number of arguments",
};
pub const OVERIDX_FIXTURE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FIXTURE_METHOD.OUTPUTS",
    identifier: Some("RunMat:TooManyOutputs"),
    when: "More than one result is requested.",
    message: "fixture method: at most one output is supported",
};
