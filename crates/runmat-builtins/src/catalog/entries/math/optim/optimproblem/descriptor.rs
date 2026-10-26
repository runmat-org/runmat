use crate::*;
pub const OPTIMPROBLEM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMPROBLEM.INVALID_INPUT",
    identifier: Some("RunMat:OptimProblem:InvalidInput"),
    when: "Invalid property, collection, objective sense, name collision, or native property path.",
    message: "invalid optimization problem",
};
const fn parameter(name: &'static str, arity: BuiltinParamArity) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::Any,
        arity,
        default: None,
        description:
            "Problem property name/value, receiver, standard subscript descriptor, or replacement.",
    }
}
const OUTPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "problem",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Native optimization problem or selected original property value.",
}];
pub(super) const fn descriptor(
    signatures: &'static [BuiltinSignatureDescriptor],
) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &[OPTIMPROBLEM_ERROR_INVALID_INPUT],
    }
}
pub const OPTIMPROBLEM_DESCRIPTOR: BuiltinDescriptor = descriptor(&[BuiltinSignatureDescriptor {
    label: "problem = optimproblem(Name,Value,...)",
    inputs: &[parameter("properties", BuiltinParamArity::Variadic)],
    outputs: OUTPUTS,
}]);
pub(super) const READ: BuiltinDescriptor = descriptor(&[BuiltinSignatureDescriptor {
    label: "value = subsref(problem,S)",
    inputs: &[
        parameter("problem", BuiltinParamArity::Required),
        parameter("subscripts", BuiltinParamArity::Required),
    ],
    outputs: OUTPUTS,
}]);
pub(super) const WRITE: BuiltinDescriptor = descriptor(&[BuiltinSignatureDescriptor {
    label: "problem = subsasgn(problem,S,value)",
    inputs: &[
        parameter("problem", BuiltinParamArity::Required),
        parameter("subscripts", BuiltinParamArity::Required),
        parameter("value", BuiltinParamArity::Required),
    ],
    outputs: OUTPUTS,
}]);
