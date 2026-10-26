use crate::*;
const fn param(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}
const INPUTS: &[BuiltinParamDescriptor] = &[
    param("fun", "Scalar objective; optional analytic gradient."),
    param("x0", "Finite real double starting array; shape preserved."),
    param(
        "ntheta",
        "Positive integer-valued double number of infinite constraint families.",
    ),
    param("seminfcon", "[c,ceq,K1,...,Kntheta,S] = seminfcon(x,S)."),
    param("A", "Linear inequality matrix or []."),
    param("b", "Inequality RHS or []."),
    param("Aeq", "Linear equality matrix or []."),
    param("beq", "Equality RHS or []."),
    param("lb", "Lower bounds or []."),
    param("ub", "Upper bounds or []."),
    param("options", "Canonical fseminf options or []."),
];
const OUTPUTS:&[BuiltinParamDescriptor]=&[
    param("x","Last accepted original-shaped point."),
    param("fval","Actual objective at x or [] before evaluation."),
    param("exitflag","1 KKT; 4 feasible small direction; 5 feasible directional derivative; 0 limit; -1 observer; -2 local infeasibility."),
    param("output","Honest iterations, actual calls, step, residuals and termination description."),
    param("lambda","Finite constraints and bounds only; excludes internal interpolated-peak rows."),
];
pub const FSEMINF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSEMINF.INVALID_ARGUMENT",
    identifier: Some("RunMat:fseminf:InvalidArgument"),
    when: "Invalid input, callback carrier or unsupported option.",
    message: "fseminf: invalid argument",
};
pub const FSEMINF_DESCRIPTOR:BuiltinDescriptor=BuiltinDescriptor{
    signatures:&[
        BuiltinSignatureDescriptor{label:"[x,fval,exitflag,output,lambda] = fseminf(fun,x0,ntheta,seminfcon)",inputs:INPUTS.split_at(4).0,outputs:OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = fseminf(fun,x0,ntheta,seminfcon,A,b)",inputs:INPUTS.split_at(6).0,outputs:OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = fseminf(fun,x0,ntheta,seminfcon,A,b,Aeq,beq)",inputs:INPUTS.split_at(8).0,outputs:OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = fseminf(fun,x0,ntheta,seminfcon,A,b,Aeq,beq,lb,ub)",inputs:INPUTS.split_at(10).0,outputs:OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = fseminf(fun,x0,ntheta,seminfcon,A,b,Aeq,beq,lb,ub,options)",inputs:INPUTS,outputs:OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = fseminf(problem)",inputs:&[param("problem","objective,x0,ntheta,seminfcon,solver='fseminf',options; optional finite constraints and bounds.")],outputs:OUTPUTS},
    ],output_mode:BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy:BuiltinCompletionPolicy::Public,errors:&[FSEMINF_ERROR_INVALID_ARGUMENT],
};
