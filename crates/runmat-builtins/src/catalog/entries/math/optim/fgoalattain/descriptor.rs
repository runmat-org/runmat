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
    param(
        "fun",
        "Vector/array objective and optional n-by-m Jacobian.",
    ),
    param("x0", "Finite starting array; original shape is preserved."),
    param("goal", "Finite real goals, one per objective."),
    param(
        "weight",
        "Finite signed weights: positive seeks F<=goal, negative seeks F>=goal; magnitude scales attainment and zero is a hard F<=goal.",
    ),
    param("A", "Linear inequality matrix or []."),
    param("b", "Linear inequality RHS or []."),
    param("Aeq", "Equality matrix or []."),
    param("beq", "Equality RHS or []."),
    param("lb", "Lower bounds or []."),
    param("ub", "Upper bounds or []."),
    param(
        "nonlcon",
        "Nonlinear callback [c,ceq] with optional Jacobians, or [].",
    ),
    param("options", "Supported scalar options structure, or []."),
];
const OUTPUTS:&[BuiltinParamDescriptor]=&[
param("x","Original-shaped last accepted point."),param("fval","Original objective array at x, or [] if unavailable."),param("attainfactor","Actual proposed attainment factor."),param("exitflag","1 original KKT; 4 feasible small step; 5 feasible small objective change; 0 limit; -1 observer; -2 local restoration failure; -4 numerical difficulty."),param("output","Actual iteration/callback accounting, residuals and goal-attainment algorithm."),param("lambda","Six original constraint/bound multiplier fields, excluding generated objective rows, or [] if unavailable.")];
pub const FGOALATTAIN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FGOALATTAIN.INVALID_ARGUMENT",
    identifier: Some("RunMat:fgoalattain:InvalidArgument"),
    when: "Invalid input or unsupported option.",
    message: "fgoalattain: invalid argument",
};
pub const FGOALATTAIN_DESCRIPTOR:BuiltinDescriptor=BuiltinDescriptor{signatures:&[
BuiltinSignatureDescriptor{label:"[x,fval,attainfactor,exitflag,output,lambda] = fgoalattain(fun,x0,goal,weight)",inputs:INPUTS.split_at(4).0,outputs:OUTPUTS},
BuiltinSignatureDescriptor{label:"[x,fval,attainfactor,exitflag,output,lambda] = fgoalattain(fun,x0,goal,weight,A,b)",inputs:INPUTS.split_at(6).0,outputs:OUTPUTS},
BuiltinSignatureDescriptor{label:"[x,fval,attainfactor,exitflag,output,lambda] = fgoalattain(fun,x0,goal,weight,A,b,Aeq,beq)",inputs:INPUTS.split_at(8).0,outputs:OUTPUTS},
BuiltinSignatureDescriptor{label:"[x,fval,attainfactor,exitflag,output,lambda] = fgoalattain(fun,x0,goal,weight,A,b,Aeq,beq,lb,ub)",inputs:INPUTS.split_at(10).0,outputs:OUTPUTS},
BuiltinSignatureDescriptor{label:"[x,fval,attainfactor,exitflag,output,lambda] = fgoalattain(fun,x0,goal,weight,A,b,Aeq,beq,lb,ub,nonlcon)",inputs:INPUTS.split_at(11).0,outputs:OUTPUTS},
BuiltinSignatureDescriptor{label:"[x,fval,attainfactor,exitflag,output,lambda] = fgoalattain(fun,x0,goal,weight,A,b,Aeq,beq,lb,ub,nonlcon,options)",inputs:INPUTS.split_at(12).0,outputs:OUTPUTS},
BuiltinSignatureDescriptor{label:"[...] = fgoalattain(problem)",inputs:&[param("problem","Scalar problem with objective,x0,solver,options,goal,weight and optional constraints/bounds.")],outputs:OUTPUTS}],output_mode:BuiltinOutputMode::ByRequestedOutputCount,completion_policy:BuiltinCompletionPolicy::Public,errors:&[FGOALATTAIN_ERROR_INVALID_ARGUMENT]};
