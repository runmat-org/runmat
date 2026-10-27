use super::*;
pub const OPTIMPROBLEM_SOLVE_ERROR_UNVERIFIED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPTIMPROBLEM.SOLVE.UNVERIFIED",
    identifier: Some("RunMat:OptimProblem:NumericalUnverified"),
    when: "A working-model terminal cannot be verified against the original model.",
    message: "original model numerical evidence is unverified",
};
const fn parameter(
    name: &'static str,
    arity: BuiltinParamArity,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::Any,
        arity,
        default: None,
        description,
    }
}
const INPUTS: &[BuiltinParamDescriptor] = &[
    parameter(
        "problem",
        BuiltinParamArity::Required,
        "Original optimization or equation problem.",
    ),
    parameter(
        "x0",
        BuiltinParamArity::Optional,
        "Original variable-name starting-point struct. It is consumed as an integer repair proposal or required nonlinear start; residual/quadratic interior-point paths do not use it.",
    ),
    parameter(
        "controls",
        BuiltinParamArity::Variadic,
        "Solver and Options name/value pairs.",
    ),
];
const OUTPUTS:&[BuiltinParamDescriptor]=&[
    parameter("sol",BuiltinParamArity::Required,"Original variable-name structure; each successful variable retains its original shape. Without a point, variable fields are []."),
    parameter("fval",BuiltinParamArity::Required,"Original objective value at sol, or original equation-shaped residuals for an EquationProblem."),
    parameter("exitflag",BuiltinParamArity::Required,"OptimizationExitflag enumeration with double and string conversions."),
    parameter("output",BuiltinParamArity::Required,"Solver iteration, feasibility, duality, algorithm, message and selected solver information."),
    parameter("lambda",BuiltinParamArity::Required,"Original Variables Lower/Upper and Constraints multipliers; unavailable multiplier fields are []. A selected integer solver returns []. Equation problems do not return this fifth output."),
];
pub const OPTIMPROBLEM_SOLVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "sol = solve(problem[,x0],Name,Value,...)",
            inputs: INPUTS,
            outputs: OUTPUTS.split_at(1).0,
        },
        BuiltinSignatureDescriptor {
            label: "[sol,fval] = solve(problem[,x0],Name,Value,...)",
            inputs: INPUTS,
            outputs: OUTPUTS.split_at(2).0,
        },
        BuiltinSignatureDescriptor {
            label: "[sol,fval,exitflag] = solve(problem[,x0],Name,Value,...)",
            inputs: INPUTS,
            outputs: OUTPUTS.split_at(3).0,
        },
        BuiltinSignatureDescriptor {
            label: "[sol,fval,exitflag,output] = solve(problem[,x0],Name,Value,...)",
            inputs: INPUTS,
            outputs: OUTPUTS.split_at(4).0,
        },
        BuiltinSignatureDescriptor {
            label: "[sol,fval,exitflag,output,lambda] = solve(problem[,x0],Name,Value,...)",
            inputs: INPUTS,
            outputs: OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        OPTIMPROBLEM_ERROR_INVALID_INPUT,
        OPTIMPROBLEM_SOLVE_ERROR_UNVERIFIED,
    ],
};
