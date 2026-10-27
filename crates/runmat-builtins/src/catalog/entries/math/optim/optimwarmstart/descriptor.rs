use crate::*;
pub const OPTIMWARMSTART_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMWARMSTART.INVALID_INPUT",
    identifier: Some("RunMat:optimwarmstart:InvalidInput"),
    when: "Invalid initial array, selected solver/options, native warm data or property path.",
    message: "optimwarmstart: invalid warm-start input",
};
pub const OPTIMWARMSTART_ERROR_READ_ONLY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMWARMSTART.READ_ONLY",
    identifier: Some("RunMat:PropertyReadOnly"),
    when: "Assignment to any warm-start property, including a nested edit.",
    message: "warm-start properties are read-only; recreate the object with optimwarmstart",
};
const OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Native immutable warm start or selected public property.",
}];
const fn parameter(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        description,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
    }
}
pub const OPTIMWARMSTART_DESCRIPTOR:BuiltinDescriptor=BuiltinDescriptor{
    signatures:&[BuiltinSignatureDescriptor{
        label:"ws = optimwarmstart(x0,options,Name,Value,...)",
        inputs:&[BuiltinParamDescriptor{
            name:"x0,options,limits",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Variadic,
            default:None,description:"Real double initial array, active-set lsqlin options and optionally both code-generation memory-limit properties.",
        }],outputs:OUTPUT,
    }],output_mode:BuiltinOutputMode::Fixed,completion_policy:BuiltinCompletionPolicy::Public,
    errors:&[OPTIMWARMSTART_ERROR_INVALID_INPUT,OPTIMWARMSTART_ERROR_READ_ONLY],
};
pub const ACTIVE_WARM_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "value = subsref(ws,S)",
        inputs: &[
            parameter(
                "ws",
                "Native immutable least-squares or quadratic warm start.",
            ),
            parameter("S", "Standard subscript descriptor."),
        ],
        outputs: OUTPUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMWARMSTART_ERROR_INVALID_INPUT],
};
pub const ACTIVE_WARM_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "ws = subsasgn(ws,S,value)",
        inputs: &[
            parameter(
                "ws",
                "Native immutable least-squares or quadratic warm start.",
            ),
            parameter("S", "Standard subscript descriptor."),
            parameter("value", "Rejected replacement: properties are read-only."),
        ],
        outputs: OUTPUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        OPTIMWARMSTART_ERROR_INVALID_INPUT,
        OPTIMWARMSTART_ERROR_READ_ONLY,
    ],
};
