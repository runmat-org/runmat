use crate::*;
pub(super) const fn parameter(
    name: &'static str,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        description,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
    }
}
const INPUTS: &[BuiltinParamDescriptor] = &[
    parameter("fun", "Objective function handle or scalar function name; receives the original x0 vector shape and returns one finite real scalar at the initial point."),
    parameter("x0", "Nonempty finite real double initial vector, row or column."),
    parameter("A", "Real double linear-inequality matrix, dense or sparse; [] means absent with b=[]."),
    parameter("b", "Real finite column of inequality right-hand sides, A*x<=b."),
    parameter("Aeq", "Real double linear-equality matrix, dense or sparse; [] means absent with beq=[]."),
    parameter("beq", "Real finite column of equality right-hand sides, Aeq*x=beq."),
    parameter("lb", "Real lower bounds or []; omitted trailing coordinates are unbounded and emit a warning."),
    parameter("ub", "Real upper bounds or []; omitted trailing coordinates are unbounded and emit a warning."),
    parameter("nonlcon", "Function handle/name returning c(x)<=0 and ceq(x)=0, or []. Output extents must remain fixed."),
    parameter("options", "Selected patternsearch controls or []; use optimoptions('patternsearch',...). Unsupported selected algorithms and execution modes are rejected."),
];
const OUTPUTS: &[BuiltinParamDescriptor] = &[
    parameter("x", "Last accepted real double vector with original x0 shape; an initial unevaluated point on a zero objective budget; [] for proven inconsistent linear constraints/bounds."),
    parameter("fval", "Actual objective value at returned x, or [] before any objective evaluation."),
    parameter("exitflag", "1 mesh/nonlinear complementarity; 2 measured step; 3 measured objective change; 4 represented step; 0 budget/time; -1 observer; -2 proven linear infeasibility; -4 numerical difficulty."),
    parameter("output", "function, problemtype, pollmethod, searchmethod, iterations, funccount, constraintcallcount, meshsize, maxconstraint, rngstate and message."),
];
pub const PATTERNSEARCH_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PATTERNSEARCH.INVALID_ARGUMENT",
    identifier: Some("RunMat:patternsearch:InvalidArgument"),
    when: "A selected input, callback result, option carrier or extent is invalid.",
    message: "patternsearch: invalid argument",
};
pub const PATTERNSEARCH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor { label: "[x,fval,exitflag,output] = patternsearch(fun,x0)", inputs: INPUTS.split_at(2).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = patternsearch(fun,x0,A,b)", inputs: INPUTS.split_at(4).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = patternsearch(fun,x0,A,b,Aeq,beq)", inputs: INPUTS.split_at(6).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = patternsearch(fun,x0,A,b,Aeq,beq,lb,ub)", inputs: INPUTS.split_at(8).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = patternsearch(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon)", inputs: INPUTS.split_at(9).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = patternsearch(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)", inputs: INPUTS, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = patternsearch(problem)", inputs: &[parameter("problem", "Required objective,x0,Aineq,bineq,Aeq,beq,lb,ub,nonlcon,options,solver='patternsearch'; optional rngstate.")], outputs: OUTPUTS },
    ], output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public, errors: &[PATTERNSEARCH_ERROR_INVALID_ARGUMENT],
};
