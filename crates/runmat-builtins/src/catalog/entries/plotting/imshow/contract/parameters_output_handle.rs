use crate::*;

pub(super) const IMSHOW_OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle to the image object.",
}];

pub(super) const IMSHOW_INPUTS_IMAGE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "I",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Image matrix/array or filename.",
}];

pub(super) const IMSHOW_INPUTS_IMAGE_RANGE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "I",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric grayscale image matrix/array.",
    },
    BuiltinParamDescriptor {
        name: "DisplayRange",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: Some("[]"),
        description: "Display limits as [] or [low high].",
    },
];
