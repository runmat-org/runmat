use super::*;
use crate::*;
mod inference;
pub(in crate::catalog::entries::math::optim) use inference::infer as infer_solve;
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
        "Original native equation problem.",
    ),
    parameter(
        "x0",
        BuiltinParamArity::Optional,
        "Original variable-name and shape starting-point structure.",
    ),
    parameter(
        "controls",
        BuiltinParamArity::Variadic,
        "Solver and Options name/value pairs.",
    ),
];
const OUTPUTS: &[BuiltinParamDescriptor] = &[
    parameter(
        "sol",
        BuiltinParamArity::Required,
        "Original variable-name solution structure.",
    ),
    parameter(
        "fval",
        BuiltinParamArity::Required,
        "Original left-minus-right array, or structure for multiple named equation entries.",
    ),
    parameter(
        "exitflag",
        BuiltinParamArity::Required,
        "OptimizationExitflag for the selected equation solver.",
    ),
    parameter(
        "output",
        BuiltinParamArity::Required,
        "Selected solver, algorithm, stopping message, work counts and wall time.",
    ),
];
pub const EQNPROBLEM_SOLVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
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
            outputs: OUTPUTS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        EQNPROBLEM_ERROR_INVALID_INPUT,
        OPTIMPROBLEM_SOLVE_ERROR_UNVERIFIED,
    ],
};
pub const EQNPROBLEM_SOLVE_METHOD_CATALOG_ENTRY: BuiltinCatalogEntry = {
    let mut entry = entry::entry(
        "optim.problemdef.EquationProblem.solve",
        &EQNPROBLEM_SOLVE_DESCRIPTOR,
        BuiltinDocumentation {
            title: Some("EquationProblem.solve"),
            slug: Some("equation-problem-solve"),
            examples: &[],
            example_exemption: Some("Covered by the eqnproblem solver examples."),
            ..documentation::DOCUMENTATION
        },
        inference::infer,
    );
    entry.contract.effects = FMINCON_CATALOG_ENTRY.contract.effects;
    entry.extensions = &[
        LSQLIN_INTEGER_OPTION_EXTENSION,
        FSOLVE_OPTION_EXTENSION,
        FSOLVE_RESIDENT_EXTENSION,
        FZERO_OPTION_EXTENSION,
        FZERO_RESIDENT_EXTENSION,
    ];
    entry
};
