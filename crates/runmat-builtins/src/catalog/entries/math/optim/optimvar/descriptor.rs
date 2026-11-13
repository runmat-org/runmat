use crate::*;
pub const OPTIMVAR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMVAR.INVALID_INPUT",
    identifier: Some("RunMat:optimvar:InvalidInput"),
    when:
        "Invalid variable name, dimensions, labels, type, bounds, reference or property operation.",
    message: "optimvar: invalid optimization-variable input",
};
pub const OPTIMVAR_ERROR_READ_ONLY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMVAR.READ_ONLY",
    identifier: Some("RunMat:PropertyReadOnly"),
    when: "Name assignment or an array-wide Type/IndexNames edit through an indexed reference.",
    message: "optimization-variable property is read-only on this reference",
};
const OUTPUTS:&[BuiltinParamDescriptor]=&[BuiltinParamDescriptor {
    name:"variable",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,
    description:"OptimizationVariable definition view or indexed reference retaining its original definition.",
}];
pub const OPTIMVAR_DESCRIPTOR:BuiltinDescriptor=BuiltinDescriptor {
    signatures:&[BuiltinSignatureDescriptor {
        label:"x = optimvar(name,dimensions...,Name,Value,...)",
        inputs:&[BuiltinParamDescriptor {
            name:"name,dimensions,properties",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Variadic,
            default:None,description:"Variable name; positive double dimensions, dimension vector or named axes; Type, LowerBound and UpperBound property pairs.",
        }],outputs:OUTPUTS,
    }],
    output_mode:BuiltinOutputMode::Fixed,completion_policy:BuiltinCompletionPolicy::Public,
    errors:&[OPTIMVAR_ERROR_INVALID_INPUT,OPTIMVAR_ERROR_READ_ONLY],
};
pub const OPTIMVAR_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "value = subsref(variable,S)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "variable",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Native optimization-variable reference.",
            },
            BuiltinParamDescriptor {
                name: "S",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Canonical standard subscript descriptor.",
            },
        ],
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMVAR_ERROR_INVALID_INPUT, OPTIMVAR_ERROR_READ_ONLY],
};
pub const OPTIMVAR_SUBSASGN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "variable = subsasgn(variable,S,value)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "variable",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Native optimization-variable reference.",
            },
            BuiltinParamDescriptor {
                name: "S",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Canonical property/subscript descriptor.",
            },
            BuiltinParamDescriptor {
                name: "value",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Selected property value.",
            },
        ],
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMVAR_ERROR_INVALID_INPUT, OPTIMVAR_ERROR_READ_ONLY],
};
