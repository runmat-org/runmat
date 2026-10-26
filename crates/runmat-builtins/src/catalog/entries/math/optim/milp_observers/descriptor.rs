use crate::*;
const INPUTS:&[BuiltinParamDescriptor]=&[
    BuiltinParamDescriptor{name:"x",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,description:"Newly found integer-feasible decision column, or []. savemilpsolutions reads nonempty iter points; optimplotmilp never reads x."},
    BuiltinParamDescriptor{name:"optimValues",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,description:"Scalar event struct. savemilpsolutions reads fval only for a new point; optimplotmilp reads time, fval and dualbound."},
    BuiltinParamDescriptor{name:"state",ty:BuiltinParamType::StringScalar,arity:BuiltinParamArity::Required,default:None,description:"init, iter or done."},
];
const OUTPUTS:&[BuiltinParamDescriptor]=&[BuiltinParamDescriptor{name:"stop",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Optional,default:None,description:"Logical false after the workspace/chart update; typed execution cancellation and errors propagate rather than masquerading as a stop flag."}];
pub const SAVEMILPSOLUTIONS_ERROR_INVALID_ARGUMENT:BuiltinErrorDescriptor=BuiltinErrorDescriptor{code:"RM.SAVEMILPSOLUTIONS.INVALID_ARGUMENT",identifier:Some("RunMat:savemilpsolutions:InvalidArgument"),when:"Invalid callback arity/state, needed point/objective data, saved history geometry, or unsupported device intent.",message:"savemilpsolutions: invalid callback argument"};
pub const OPTIMPLOTMILP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMPLOTMILP.INVALID_ARGUMENT",
    identifier: Some("RunMat:optimplotmilp:InvalidArgument"),
    when: "Invalid callback arity/state, needed bound/time data, or unsupported device intent.",
    message: "optimplotmilp: invalid callback argument",
};
pub const SAVEMILPSOLUTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = savemilpsolutions(x,optimValues,state)",
        inputs: INPUTS,
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[SAVEMILPSOLUTIONS_ERROR_INVALID_ARGUMENT],
};
pub const OPTIMPLOTMILP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "stop = optimplotmilp(x,optimValues,state)",
        inputs: INPUTS,
        outputs: OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[OPTIMPLOTMILP_ERROR_INVALID_ARGUMENT],
};
