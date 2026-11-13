use crate::*;
const fn parameter(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        description,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
    }
}
const OBJECTIVE: BuiltinParamDescriptor = parameter(
    "objconstr",
    "Selected function returning Fval and/or nonlinear Ineq in one result.",
);
const LOWER: BuiltinParamDescriptor =
    parameter("lb", "Nonempty finite real double lower-bound vector.");
const UPPER: BuiltinParamDescriptor = parameter(
    "ub",
    "Finite real double upper bounds with the same original variable extent.",
);
const INTEGER: BuiltinParamDescriptor = parameter(
    "intcon",
    "Double one-based integer coordinate indices, or [].",
);
const OPTIONS: BuiltinParamDescriptor =
    parameter("options", "Selected canonical surrogate options, or [].");
const INPUTS: &[BuiltinParamDescriptor] = &[
    OBJECTIVE,
    LOWER,
    UPPER,
    INTEGER,
    parameter("A", "Linear inequality matrix, or []."),
    parameter("b", "Linear inequality right-hand sides, or []."),
    parameter("Aeq", "Linear equality matrix, or []."),
    parameter("beq", "Linear equality right-hand sides, or []."),
    OPTIONS,
];
const OUTPUTS: &[BuiltinParamDescriptor] = &[
    parameter("x", "Best measured original-space point; least maximum nonlinear violation when no feasible point is available."),
    parameter("fval", "Measured Fval at x, or [] when no objective value is present."),
    parameter("exitflag", "10 unique solution; 3 too few new sites; 1 objective limit; 0 function/time limit; -1 observer; -2 no feasible point."),
    parameter("output", "funccount, elapsedtime, message, constrviolation, ineq, and initial rngstate."),
    parameter("trials", "X rows, corresponding Fval column (or []), and Ineq rows from the retained measured trial history."),
];
pub const SURROGATEOPT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor { label: "[x,fval,exitflag,output,trials] = surrogateopt(objconstr,lb,ub)", inputs: INPUTS.split_at(3).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = surrogateopt(objconstr,lb,ub,intcon)", inputs: INPUTS.split_at(4).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = surrogateopt(objconstr,lb,ub,intcon,A,b,Aeq,beq)", inputs: INPUTS.split_at(8).0, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = surrogateopt(objconstr,lb,ub,options)", inputs: &[OBJECTIVE, LOWER, UPPER, OPTIONS], outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = surrogateopt(objconstr,lb,ub,intcon,options)", inputs: &[OBJECTIVE, LOWER, UPPER, INTEGER, OPTIONS], outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = surrogateopt(objconstr,lb,ub,intcon,A,b,Aeq,beq,options)", inputs: INPUTS, outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = surrogateopt(problem)", inputs: &[parameter("problem", "objective,lb,ub,solver='surrogateopt',options; optional intcon,Aineq,bineq,Aeq,beq,rngstate.")], outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = surrogateopt(checkpointFile)", inputs: &[parameter("checkpointFile", "Authenticated solver checkpoint path.")], outputs: OUTPUTS },
        BuiltinSignatureDescriptor { label: "[...] = surrogateopt(checkpointFile,opts)", inputs: &[parameter("checkpointFile", "Authenticated solver checkpoint path."), parameter("opts", "Permitted resume-option overrides.")], outputs: OUTPUTS },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[super::SURROGATEOPT_ERROR_INVALID_ARGUMENT, super::SURROGATEOPT_ERROR_NUMERICAL_FAILURE, super::SURROGATEOPT_ERROR_CHECKPOINT],
};
