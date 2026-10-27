use crate::*;
pub const INTEGER_CONSTRAINT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTEGER_CONSTRAINT.INVALID_INPUT",
    identifier: Some("RunMat:integerConstraint:InvalidInput"),
    when:
        "Invalid arity, property name, double-vector index data, duplicate or overlapping indices.",
    message: "integerConstraint: invalid domain descriptor",
};
pub const INTEGER_CONSTRAINT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "intcon = integerConstraint(Name,Value,...)",
        inputs: &[BuiltinParamDescriptor {
            name: "Name,Value",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Variadic,
            default: None,
            description:
                "Optional Integer, SemiContinuous and SemiInteger double-vector properties.",
        }],
        outputs: &[BuiltinParamDescriptor {
            name: "intcon",
            ty: BuiltinParamType::Any,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Scalar IntegerConstraint value object.",
        }],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[INTEGER_CONSTRAINT_ERROR_INVALID_INPUT],
};
