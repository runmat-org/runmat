use crate::*;

const INTEGER_SOLVER_INPUT: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "objective, cone, constraint, or bound numeric data", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Existing RunMat forms normalize authoritative native integer values to binary64, including rounding outside its exact integer range. This is not an exact-integer solver or a claim of MATLAB integer-overload parity.",
}];
const INTEGER_COUNT_INPUT: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "options.MaxIter", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Native integer structural counts are decoded exactly; zero is honored and the iteration limit must fit the backend's u32 range.",
}];
const INTEGER_TOLERANCE_INPUT: &[BuiltinIntegerInputCapability] = &[BuiltinIntegerInputCapability {
    name: "options.TolX or options.TolFun", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Unlike solver coefficients, integer tolerances must be exactly representable as finite binary64 at the shared options boundary.",
}];
pub const CONEPROG_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[
    BuiltinIntegerCapabilityDescriptor { form: "coneprog(integer-valued solver data)", inputs: INTEGER_SOLVER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific, backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple, notes: "Preserves the existing floating-boundary rounding policy; no new integer rejection or extension gate is introduced." },
    BuiltinIntegerCapabilityDescriptor { form: "coneprog(..., options)", inputs: INTEGER_COUNT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Negative and overflowing structural counts reject before iteration." },
    BuiltinIntegerCapabilityDescriptor { form: "coneprog(..., options)", inputs: INTEGER_TOLERANCE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Finite checked binary64 tolerance conversion uses the existing shared options owner." },
];

const OUTPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Optimal decision vector.",
}];

const OUTPUT_X_FVAL: [BuiltinParamDescriptor; 2] = [
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

const OUTPUT_X_FVAL_EXITFLAG: [BuiltinParamDescriptor; 3] = [
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

const OUTPUT_ALL: [BuiltinParamDescriptor; 5] = [
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
    BuiltinParamDescriptor {
        name: "lambda",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Actual original-row and scalar cone dual multipliers at a verified optimum; empty otherwise.",
    },
];
const OUTPUT_DIAGNOSTICS: [BuiltinParamDescriptor; 4] = [
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

const INPUTS_CORE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Linear objective vector.",
    },
    BuiltinParamDescriptor {
        name: "socConstraints",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second-order cone constraints from secondordercone.",
    },
];

const INPUTS_LINEAR: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Linear objective vector.",
    },
    BuiltinParamDescriptor {
        name: "socConstraints",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second-order cone constraints from secondordercone.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Inequality constraint matrix.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Inequality constraint right-hand side.",
    },
];

const INPUTS_FULL: [BuiltinParamDescriptor; 9] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Linear objective vector.",
    },
    BuiltinParamDescriptor {
        name: "socConstraints",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second-order cone constraints from secondordercone.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Inequality constraint matrix.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
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
        arity: BuiltinParamArity::Optional,
        default: Some("optimoptions('coneprog')"),
        description: "Options struct from optimoptions or optimset.",
    },
];

const INPUTS_PROBLEM: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "problem",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Problem struct with f, socConstraints, linear constraints, bounds, and options.",
}];

const CONEPROG_SIGNATURES: [BuiltinSignatureDescriptor; 20] = [
    BuiltinSignatureDescriptor {
        label: "x = coneprog(f, socConstraints)",
        inputs: &INPUTS_CORE,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = coneprog(f, socConstraints, A, b)",
        inputs: &INPUTS_LINEAR,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = coneprog(f, socConstraints, A, b, Aeq, beq, lb, ub, options)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "x = coneprog(problem)",
        inputs: &INPUTS_PROBLEM,
        outputs: &OUTPUT_X,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = coneprog(f, socConstraints)",
        inputs: &INPUTS_CORE,
        outputs: &OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = coneprog(f, socConstraints, A, b)",
        inputs: &INPUTS_LINEAR,
        outputs: &OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = coneprog(f, socConstraints, A, b, Aeq, beq, lb, ub, options)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval] = coneprog(problem)",
        inputs: &INPUTS_PROBLEM,
        outputs: &OUTPUT_X_FVAL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = coneprog(f, socConstraints)",
        inputs: &INPUTS_CORE,
        outputs: &OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = coneprog(f, socConstraints, A, b)",
        inputs: &INPUTS_LINEAR,
        outputs: &OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = coneprog(f, socConstraints, A, b, Aeq, beq, lb, ub, options)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag] = coneprog(problem)",
        inputs: &INPUTS_PROBLEM,
        outputs: &OUTPUT_X_FVAL_EXITFLAG,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, lambda] = coneprog(f, socConstraints)",
        inputs: &INPUTS_CORE,
        outputs: &OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, lambda] = coneprog(f, socConstraints, A, b)",
        inputs: &INPUTS_LINEAR,
        outputs: &OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, lambda] = coneprog(f, socConstraints, A, b, Aeq, beq, lb, ub, options)",
        inputs: &INPUTS_FULL,
        outputs: &OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor {
        label: "[x, fval, exitflag, output, lambda] = coneprog(problem)",
        inputs: &INPUTS_PROBLEM,
        outputs: &OUTPUT_ALL,
    },
    BuiltinSignatureDescriptor { label: "[x, fval, exitflag, output] = coneprog(f, socConstraints)", inputs: &INPUTS_CORE, outputs: &OUTPUT_DIAGNOSTICS },
    BuiltinSignatureDescriptor { label: "[x, fval, exitflag, output] = coneprog(f, socConstraints, A, b)", inputs: &INPUTS_LINEAR, outputs: &OUTPUT_DIAGNOSTICS },
    BuiltinSignatureDescriptor { label: "[x, fval, exitflag, output] = coneprog(f, socConstraints, A, b, Aeq, beq, lb, ub, options)", inputs: &INPUTS_FULL, outputs: &OUTPUT_DIAGNOSTICS },
    BuiltinSignatureDescriptor { label: "[x, fval, exitflag, output] = coneprog(problem)", inputs: &INPUTS_PROBLEM, outputs: &OUTPUT_DIAGNOSTICS },
];

pub const CONEPROG_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONEPROG.INVALID_ARGUMENT",
    identifier: Some("RunMat:coneprog:InvalidArgument"),
    when: "The argument count or optional argument grammar is invalid.",
    message: "coneprog: invalid argument",
};

pub const CONEPROG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONEPROG.INVALID_INPUT",
    identifier: Some("RunMat:coneprog:InvalidInput"),
    when: "Objective, cone, constraint, option, or bound dimensions/types are invalid.",
    message: "coneprog: invalid input",
};

pub const CONEPROG_ERRORS: [BuiltinErrorDescriptor; 2] = [
    CONEPROG_ERROR_INVALID_ARGUMENT,
    CONEPROG_ERROR_INVALID_INPUT,
];

pub const CONEPROG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONEPROG_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONEPROG_ERRORS,
};
