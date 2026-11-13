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
        "Objective returning a scalar, optionally its gradient, and the selected reflective Hessian or Hinfo.",
    ),
    param(
        "x0",
        "Finite nonempty real starting array; its shape is preserved.",
    ),
    param("A", "Sparse or dense linear inequality matrix, or []."),
    param("b", "Linear inequality RHS vector, or []."),
    param("Aeq", "Sparse or dense linear equality matrix, or []."),
    param("beq", "Linear equality RHS vector, or []."),
    param("lb", "Lower bound array, with -Inf for an absent bound."),
    param("ub", "Upper bound array, with Inf for an absent bound."),
    param(
        "nonlcon",
        "Nonlinear constraint function returning [c,ceq] and optionally [GC,GCeq], or [].",
    ),
    param(
        "options",
        "Scalar supported options struct; default interior-point or explicit sqp/active-set/trust-region-reflective.",
    ),
];
const CORE: &[BuiltinParamDescriptor] = INPUTS.split_at(4).0;
const LINEAR: &[BuiltinParamDescriptor] = INPUTS.split_at(6).0;
const BOUNDS: &[BuiltinParamDescriptor] = INPUTS.split_at(8).0;
const NONLINEAR: &[BuiltinParamDescriptor] = INPUTS.split_at(9).0;
const OUTPUTS:[BuiltinParamDescriptor;7]=[
    param("x","Last evaluated bound-feasible point, shaped like x0."),param("fval","Objective at x, or [] when no evaluation was permitted."),
    param("exitflag","1 original KKT; 2 feasible small step; 3 feasible small function change; 0 limit; -1 observer; -2 local restoration failure; -4 numerical difficulty."),
    param("output","Actual iterations, evaluated points/callback counts, feasibility, first-order measure, algorithm and elapsed admitted clock."),
    param("lambda","Original-row multiplier estimates in six fields, or [] before an estimate is available."),
    param("grad","Already evaluated objective gradient column, or [] if unavailable."),
    param("hessian","Last-used SQP/active-set/reflective model, final-point interior-point BFGS model, or original reflective Hinfo; [] if unavailable."),
];
pub const FMINCON_ERROR_INVALID_ARGUMENT:BuiltinErrorDescriptor=BuiltinErrorDescriptor{
    code:"RM.FMINCON.INVALID_ARGUMENT",identifier:Some("RunMat:fmincon:InvalidArgument"),
    when:"Invalid public grammar, unsupported algorithm/option, or invalid numeric/callback contract.",message:"fmincon: invalid argument",
};
pub const FMINCON_DESCRIPTOR:BuiltinDescriptor=BuiltinDescriptor{
    signatures:&[
        BuiltinSignatureDescriptor{label:"[x,fval,exitflag,output,lambda,grad,hessian] = fmincon(fun,x0,A,b)",inputs:CORE,outputs:&OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = fmincon(fun,x0,A,b,Aeq,beq)",inputs:LINEAR,outputs:&OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub)",inputs:BOUNDS,outputs:&OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon)",inputs:NONLINEAR,outputs:&OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)",inputs:INPUTS,outputs:&OUTPUTS},
        BuiltinSignatureDescriptor{label:"[...] = fmincon(problem)",inputs:&[param("problem","Scalar structure with objective, x0, solver and options, plus optional constraints/bounds.")],outputs:&OUTPUTS},
    ],output_mode:BuiltinOutputMode::ByRequestedOutputCount,completion_policy:BuiltinCompletionPolicy::Public,
    errors:&[FMINCON_ERROR_INVALID_ARGUMENT],
};
