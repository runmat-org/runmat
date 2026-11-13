use crate::*;

const fn receiver() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Table or dictionary value whose indexing method is invoked.",
    }
}
const fn subscript() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Standard structure or structure-array indexing path with type and subs fields.",
    }
}
const fn rhs() -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value assigned at the selected location.",
    }
}
const REF_INPUTS: [BuiltinParamDescriptor; 2] = [receiver(), subscript()];
const ASSIGN_INPUTS: [BuiltinParamDescriptor; 3] = [receiver(), subscript(), rhs()];
const REF_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "The selected table, variable, dictionary value, or nested member.",
}];
const ASSIGN_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated value object; the original receiver is not mutated in place.",
}];

const TABLE_REF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = table.subsref(T, S)",
    inputs: &REF_INPUTS,
    outputs: &REF_OUTPUTS,
}];
const TABLE_ASSIGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "T = table.subsasgn(T, S, rhs)",
    inputs: &ASSIGN_INPUTS,
    outputs: &ASSIGN_OUTPUTS,
}];
const DICTIONARY_REF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = dictionary.subsref(d, S)",
    inputs: &REF_INPUTS,
    outputs: &REF_OUTPUTS,
}];
const DICTIONARY_ASSIGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "d = dictionary.subsasgn(d, S, rhs)",
        inputs: &ASSIGN_INPUTS,
        outputs: &ASSIGN_OUTPUTS,
    }];

const INDEX_ERRORS: [BuiltinErrorDescriptor; 3] = [
    super::super::shared::TABLE_ERROR_INVALID_ARGUMENT,
    super::super::shared::TABLE_ERROR_INVALID_INDEX,
    super::super::shared::TABLE_ERROR_INVALID_VARIABLE,
];

pub const TABLE_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TABLE_REF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &INDEX_ERRORS,
};
pub const TABLE_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TABLE_ASSIGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &INDEX_ERRORS,
};
pub const DICTIONARY_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DICTIONARY_REF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &INDEX_ERRORS,
};
pub const DICTIONARY_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DICTIONARY_ASSIGN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &INDEX_ERRORS,
};
