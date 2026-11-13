use crate::*;
const fn input(
    name: &'static str,
    ty: BuiltinParamType,
    optional: bool,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: if optional {
            BuiltinParamArity::Optional
        } else {
            BuiltinParamArity::Required
        },
        default: None,
        description,
    }
}
const INPUTS: [BuiltinParamDescriptor; 10] = [
    input(
        "f",
        BuiltinParamType::NumericArray,
        false,
        "Finite real objective array, flattened column-major; [] selects feasibility.",
    ),
    input(
        "intcon",
        BuiltinParamType::Any,
        false,
        "Positive variable-index array or scalar IntegerConstraint; [] selects a linear relaxation.",
    ),
    input(
        "A",
        BuiltinParamType::NumericArray,
        false,
        "Finite real dense or sparse inequality coefficient matrix.",
    ),
    input(
        "b",
        BuiltinParamType::Any,
        false,
        "Full upper-bound vector or two-element range cell {bl,bu}.",
    ),
    input(
        "Aeq",
        BuiltinParamType::NumericArray,
        true,
        "Finite real dense or sparse equality coefficient matrix.",
    ),
    input(
        "beq",
        BuiltinParamType::NumericArray,
        true,
        "Finite full equality right-side vector.",
    ),
    input(
        "lb",
        BuiltinParamType::NumericArray,
        true,
        "Variable lower bounds, flattened; omitted trailing bounds are -Inf.",
    ),
    input(
        "ub",
        BuiltinParamType::NumericArray,
        true,
        "Variable upper bounds, flattened; omitted trailing bounds are Inf.",
    ),
    input(
        "x0",
        BuiltinParamType::NumericArray,
        true,
        "Finite initial-point array; used as an actual fixed-integer repair proposal.",
    ),
    input(
        "options",
        BuiltinParamType::Any,
        true,
        "Consumed intlinprog scalar options struct or [].",
    ),
];
const OUTPUTS: [BuiltinParamDescriptor; 4] = [
    input(
        "x",
        BuiltinParamType::NumericArray,
        false,
        "Verified publishable binary64 decision column or [].",
    ),
    input(
        "fval",
        BuiltinParamType::NumericArray,
        false,
        "Original objective evaluated at the actual returned x, or [].",
    ),
    input(
        "exitflag",
        BuiltinParamType::NumericScalar,
        false,
        "Certified gap/terminal result or truthful early/numerical stop.",
    ),
    input(
        "output",
        BuiltinParamType::Any,
        false,
        "Gap, counters, actual point violation, message and explicitly distinguished proof bounds.",
    ),
];
const PROBLEM:[BuiltinParamDescriptor;1]=[input("problem",BuiltinParamType::Any,false,"Scalar problem with f,intcon,solver='intlinprog',options and optional constraint/bound/x0 fields.")];
pub const INTLINPROG_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTLINPROG.INVALID_ARGUMENT",
    identifier: Some("RunMat:intlinprog:InvalidArgument"),
    when: "Unsupported input/output arity.",
    message: "intlinprog: invalid argument",
};
pub const INTLINPROG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTLINPROG.INVALID_INPUT",
    identifier: Some("RunMat:intlinprog:InvalidInput"),
    when: "Invalid numeric geometry, variable indices, option or unsupported public form.",
    message: "intlinprog: invalid input",
};
pub const INTLINPROG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "[x,fval,exitflag,output] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,x0,options)",
            inputs: &INPUTS,
            outputs: &OUTPUTS,
        },
        BuiltinSignatureDescriptor {
            label: "[x,fval,exitflag,output] = intlinprog(problem)",
            inputs: &PROBLEM,
            outputs: &OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        INTLINPROG_ERROR_INVALID_ARGUMENT,
        INTLINPROG_ERROR_INVALID_INPUT,
    ],
};
