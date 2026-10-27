use crate::*;
pub const OPTIMMODEL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMMODEL.INVALID_INPUT",
    identifier: Some("RunMat:OptimModel:InvalidInput"),
    when: "Invalid expression dimensions, operand, index, property, point field or comparison.",
    message: "invalid optimization-model input",
};
const fn parameter(name: &'static str, arity: BuiltinParamArity) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::Any,
        arity,
        default: None,
        description:
            "Symbolic operand, ordinary metadata, subscript descriptor or evaluation point.",
    }
}
const OUTPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Expression or comparison object; evaluate returns its same-shaped double value.",
}];
const fn descriptor(signatures: &'static [BuiltinSignatureDescriptor]) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &[OPTIMMODEL_ERROR_INVALID_INPUT],
    }
}
pub const OPTIMEXPR_DESCRIPTOR: BuiltinDescriptor = descriptor(&[
    BuiltinSignatureDescriptor {
        label: "expr = optimexpr",
        inputs: &[],
        outputs: OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "expr = optimexpr(dimensions...)",
        inputs: &[parameter("dimensions", BuiltinParamArity::Variadic)],
        outputs: OUTPUTS,
    },
]);
pub(super) const UNARY: BuiltinDescriptor = descriptor(&[BuiltinSignatureDescriptor {
    label: "result = method(operand)",
    inputs: &[parameter("operand", BuiltinParamArity::Required)],
    outputs: OUTPUTS,
}]);
pub(super) const BINARY: BuiltinDescriptor = descriptor(&[BuiltinSignatureDescriptor {
    label: "result = method(left,right)",
    inputs: &[
        parameter("left", BuiltinParamArity::Required),
        parameter("right", BuiltinParamArity::Required),
    ],
    outputs: OUTPUTS,
}]);
pub(super) const WRITE: BuiltinDescriptor = descriptor(&[BuiltinSignatureDescriptor {
    label: "result = subsasgn(object,S,value)",
    inputs: &[
        parameter("object", BuiltinParamArity::Required),
        parameter("subscripts", BuiltinParamArity::Required),
        parameter("value", BuiltinParamArity::Required),
    ],
    outputs: OUTPUTS,
}]);
pub(super) const VARIADIC: BuiltinDescriptor = descriptor(&[BuiltinSignatureDescriptor {
    label: "result = function(operands,metadata...)",
    inputs: &[parameter(
        "operands_and_metadata",
        BuiltinParamArity::Variadic,
    )],
    outputs: OUTPUTS,
}]);

pub(super) const ZEROS_LIKE: BuiltinDescriptor = descriptor(&[BuiltinSignatureDescriptor {
    label: "expr = zerosLike(prototype,dimensions...)",
    inputs: &[
        parameter("prototype", BuiltinParamArity::Required),
        parameter("dimensions", BuiltinParamArity::Variadic),
    ],
    outputs: OUTPUTS,
}]);
