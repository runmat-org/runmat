use crate::*;

const LINPROG_OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Optimal decision vector.",
}];

const LINPROG_OUTPUT_X_FVAL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Optimal decision vector.",
    },
    BuiltinParamDescriptor {
        name: "fval",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Objective value f'*x at the solution.",
    },
];

const LINPROG_OUTPUT_X_FVAL_EXITFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Optimal decision vector.",
    },
    BuiltinParamDescriptor {
        name: "fval",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Objective value f'*x at the solution.",
    },
    BuiltinParamDescriptor {
        name: "exitflag",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Solver status code.",
    },
];

const LINPROG_OUTPUT_ALL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Optimal decision vector.",
    },
    BuiltinParamDescriptor {
        name: "fval",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Objective value f'*x at the solution.",
    },
    BuiltinParamDescriptor {
        name: "exitflag",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Solver status code.",
    },
    BuiltinParamDescriptor {
        name: "output",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Diagnostic metadata struct.",
    },
];

const LINPROG_INPUTS_CORE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Linear objective vector.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Inequality constraint matrix.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Inequality constraint right-hand side.",
    },
];

const LINPROG_INPUTS_EQ: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Linear objective vector.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Inequality constraint matrix.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Inequality constraint right-hand side.",
    },
    BuiltinParamDescriptor {
        name: "Aeq",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Equality constraint matrix.",
    },
    BuiltinParamDescriptor {
        name: "beq",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Equality constraint right-hand side.",
    },
];

const LINPROG_INPUTS_BOUNDS: [BuiltinParamDescriptor; 7] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Linear objective vector.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Inequality constraint matrix.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Inequality constraint right-hand side.",
    },
    BuiltinParamDescriptor {
        name: "Aeq",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Equality constraint matrix.",
    },
    BuiltinParamDescriptor {
        name: "beq",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Equality constraint right-hand side.",
    },
    BuiltinParamDescriptor {
        name: "lb",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Lower bounds.",
    },
    BuiltinParamDescriptor {
        name: "ub",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Upper bounds.",
    },
];

const LINPROG_INPUTS_OPTIONS: [BuiltinParamDescriptor; 8] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Linear objective vector.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Inequality constraint matrix.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Inequality constraint right-hand side.",
    },
    BuiltinParamDescriptor {
        name: "Aeq",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Equality constraint matrix.",
    },
    BuiltinParamDescriptor {
        name: "beq",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Equality constraint right-hand side.",
    },
    BuiltinParamDescriptor {
        name: "lb",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Lower bounds.",
    },
    BuiltinParamDescriptor {
        name: "ub",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Upper bounds.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Consumed MaxIterations/MaxIter and Display controls for the current host algorithm.",
    },
];

const LINPROG_OUTPUT_WITH_LAMBDA: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor { name: "x", ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required, default: None,
        description: "Verified optimum or feasible stopped incumbent; empty if unavailable." },
    BuiltinParamDescriptor { name: "fval", ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required, default: None, description: "Objective at returned x, or empty." },
    BuiltinParamDescriptor { name: "exitflag", ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required, default: None, description: "Solver status code." },
    BuiltinParamDescriptor { name: "output", ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required, default: None, description: "Original-model residuals and termination metadata." },
    BuiltinParamDescriptor { name: "lambda", ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required, default: None,
        description: "Verified optimal multipliers in original inequality, equality, lower-bound and upper-bound order; empty fields otherwise." },
];

const LINPROG_SIGNATURES: [BuiltinSignatureDescriptor; 20] = [
    BuiltinSignatureDescriptor {
        label: "x = linprog(f, A, b)",
        inputs: &LINPROG_INPUTS_CORE,
        outputs: &LINPROG_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = linprog(f, A, b, Aeq, beq)",
        inputs: &LINPROG_INPUTS_EQ,
        outputs: &LINPROG_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = linprog(f, A, b, Aeq, beq, lb, ub)",
        inputs: &LINPROG_INPUTS_BOUNDS,
        outputs: &LINPROG_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = linprog(f, A, b)",
        inputs: &LINPROG_INPUTS_CORE,
        outputs: &LINPROG_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = linprog(f, A, b, Aeq, beq)",
        inputs: &LINPROG_INPUTS_EQ,
        outputs: &LINPROG_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = linprog(f, A, b, Aeq, beq, lb, ub)",
        inputs: &LINPROG_INPUTS_BOUNDS,
        outputs: &LINPROG_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = linprog(f, A, b)",
        inputs: &LINPROG_INPUTS_CORE,
        outputs: &LINPROG_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = linprog(f, A, b, Aeq, beq)",
        inputs: &LINPROG_INPUTS_EQ,
        outputs: &LINPROG_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = linprog(f, A, b, Aeq, beq, lb, ub)",
        inputs: &LINPROG_INPUTS_BOUNDS,
        outputs: &LINPROG_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = linprog(f, A, b)",
        inputs: &LINPROG_INPUTS_CORE,
        outputs: &LINPROG_OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = linprog(f, A, b, Aeq, beq)",
        inputs: &LINPROG_INPUTS_EQ,
        outputs: &LINPROG_OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = linprog(f, A, b, Aeq, beq, lb, ub)",
        inputs: &LINPROG_INPUTS_BOUNDS,
        outputs: &LINPROG_OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "x = linprog(f, A, b, Aeq, beq, lb, ub, options)",
        inputs: &LINPROG_INPUTS_OPTIONS,
        outputs: &LINPROG_OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = linprog(f, A, b, Aeq, beq, lb, ub, options)",
        inputs: &LINPROG_INPUTS_OPTIONS,
        outputs: &LINPROG_OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = linprog(f, A, b, Aeq, beq, lb, ub, options)",
        inputs: &LINPROG_INPUTS_OPTIONS,
        outputs: &LINPROG_OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output] = linprog(f, A, b, Aeq, beq, lb, ub, options)",
        inputs: &LINPROG_INPUTS_OPTIONS,
        outputs: &LINPROG_OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, lambda] = linprog(f, A, b)",
        inputs: &LINPROG_INPUTS_CORE,
        outputs: &LINPROG_OUTPUT_WITH_LAMBDA,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, lambda] = linprog(f, A, b, Aeq, beq)",
        inputs: &LINPROG_INPUTS_EQ,
        outputs: &LINPROG_OUTPUT_WITH_LAMBDA,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, lambda] = linprog(f, A, b, Aeq, beq, lb, ub)",
        inputs: &LINPROG_INPUTS_BOUNDS,
        outputs: &LINPROG_OUTPUT_WITH_LAMBDA,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, lambda] = linprog(f, A, b, Aeq, beq, lb, ub, options)",
        inputs: &LINPROG_INPUTS_OPTIONS,
        outputs: &LINPROG_OUTPUT_WITH_LAMBDA,
    },
];

pub const LINPROG_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINPROG.INVALID_ARGUMENT",
    identifier: Some("RunMat:linprog:InvalidArgument"),
    when: "The argument count or optional argument grammar is invalid.",
    message: "linprog: invalid argument",
};

pub const LINPROG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINPROG.INVALID_INPUT",
    identifier: Some("RunMat:linprog:InvalidInput"),
    when: "Objective, constraint, or bound dimensions/types are invalid.",
    message: "linprog: invalid input",
};

const LINPROG_ERRORS: [BuiltinErrorDescriptor; 2] =
    [LINPROG_ERROR_INVALID_ARGUMENT, LINPROG_ERROR_INVALID_INPUT];

pub const LINPROG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LINPROG_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LINPROG_ERRORS,
};
