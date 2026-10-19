use crate::*;

const IND: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ind",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Positive one-based output indices as a vector, matrix, or cell array of vectors.",
};
const DATA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar or vector data to accumulate.",
};
const SIZE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("[]"),
    description: "Positive output-size vector, or [] to infer it from ind.",
};
const FUNCTION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fun",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("@sum"),
    description: "Scalar-returning group function, or [] for sum.",
};
const FILL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "fillval",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("0"),
    description: "Scalar fill value matching the group-function result, or [].",
};
const SPARSE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "issparse",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: Some("false"),
    description: "Logical or numeric scalar 0 or 1 selecting sparse output.",
};
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Accumulated full or sparse array.",
}];
const INPUTS_2: [BuiltinParamDescriptor; 2] = [IND, DATA];
const INPUTS_3: [BuiltinParamDescriptor; 3] = [IND, DATA, SIZE];
const INPUTS_4: [BuiltinParamDescriptor; 4] = [IND, DATA, SIZE, FUNCTION];
const INPUTS_5: [BuiltinParamDescriptor; 5] = [IND, DATA, SIZE, FUNCTION, FILL];
const INPUTS_6: [BuiltinParamDescriptor; 6] = [IND, DATA, SIZE, FUNCTION, FILL, SPARSE];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind,data)",
        inputs: &INPUTS_2,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind,data,sz)",
        inputs: &INPUTS_3,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind,data,sz,fun)",
        inputs: &INPUTS_4,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind,data,sz,fun,fillval)",
        inputs: &INPUTS_5,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "B = accumarray(ind,data,sz,fun,fillval,issparse)",
        inputs: &INPUTS_6,
        outputs: &OUTPUTS,
    },
];

pub(super) const ACCUMARRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: super::ERRORS,
};
