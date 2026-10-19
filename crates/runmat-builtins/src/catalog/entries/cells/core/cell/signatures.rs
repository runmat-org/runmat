use crate::{
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

const OUTPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Preallocated cell array.",
}];
const N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square size.",
};
const SIZE_VECTOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Size vector.",
};
const SIZES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sizes",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Size vector or individual dimension sizes.",
};
const LIKE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "like",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"like\""),
    description: "RunMat prototype selector.",
};
const PROTOTYPE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "prototype",
    ty: BuiltinParamType::LikePrototype,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Prototype for result shape or empty element representation.",
};
const NONE: &[BuiltinParamDescriptor] = &[];
const ONE: &[BuiltinParamDescriptor] = &[N];
const VECTOR: &[BuiltinParamDescriptor] = &[SIZE_VECTOR];
const VARIADIC: &[BuiltinParamDescriptor] = &[SIZES];
const LIKE_ONLY: &[BuiltinParamDescriptor] = &[LIKE, PROTOTYPE];
const VECTOR_LIKE: &[BuiltinParamDescriptor] = &[SIZE_VECTOR, LIKE, PROTOTYPE];
const VARIADIC_LIKE: &[BuiltinParamDescriptor] = &[SIZES, LIKE, PROTOTYPE];

pub(super) const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "C = cell()",
        inputs: NONE,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(n)",
        inputs: ONE,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(sz)",
        inputs: VECTOR,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(m, n, ...)",
        inputs: VARIADIC,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(\"like\", prototype)",
        inputs: LIKE_ONLY,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(sz, \"like\", prototype)",
        inputs: VECTOR_LIKE,
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "C = cell(m, n, ..., \"like\", prototype)",
        inputs: VARIADIC_LIKE,
        outputs: OUTPUTS,
    },
];
