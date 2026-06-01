//! MATLAB-compatible `linprog` builtin for small and medium linear programs.

use nalgebra::{DMatrix, DVector};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    StructValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::math::optim::type_resolvers::linear_programming_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "linprog";
const ALGORITHM: &str = "active-set vertex enumeration";
const TOL: f64 = 1.0e-8;

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

const LINPROG_SIGNATURES: [BuiltinSignatureDescriptor; 12] = [
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
];

const LINPROG_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINPROG.INVALID_ARGUMENT",
    identifier: Some("RunMat:linprog:InvalidArgument"),
    when: "The argument count or optional argument grammar is invalid.",
    message: "linprog: invalid argument",
};

const LINPROG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
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

fn linprog_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.starts_with("linprog:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::linprog")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "linprog",
    op_kind: GpuOpKind::Custom("linear-programming"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host active-set LP solver. GPU-resident numeric inputs are gathered before solving.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::linprog")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "linprog",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Linear programming is a solver boundary and terminates fusion planning.",
};

#[runtime_builtin(
    name = "linprog",
    category = "math/optim",
    summary = "Solve a linear programming minimization problem with linear constraints and bounds.",
    keywords = "linprog,linear programming,optimization,linear constraints,bounds",
    accel = "sink",
    type_resolver(linear_programming_type),
    descriptor(crate::builtins::math::optim::linprog::LINPROG_DESCRIPTOR),
    builtin_path = "crate::builtins::math::optim::linprog"
)]
async fn linprog_builtin(f: Value, a: Value, b: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 4 {
        return Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_ARGUMENT,
            "too many input arguments",
        ));
    }

    let f = numeric_vector("f", f, FiniteMode::Finite).await?;
    if f.is_empty() {
        return Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_INPUT,
            "f must be a nonempty numeric vector",
        ));
    }
    let n = f.len();

    let (mut a_ineq, mut b_ineq) = parse_constraint_pair("A", a, "b", b, n).await?;
    let (a_eq, b_eq) = parse_optional_equality(rest.first(), rest.get(1), n).await?;
    let (lb, ub) = parse_bounds(rest.get(2), rest.get(3), n).await?;

    for i in 0..n {
        if lb[i] > ub[i] + TOL {
            return Ok(finalize(LinprogOutcome::infeasible(
                "No feasible point found: lower bound exceeds upper bound.",
            )));
        }
        if lb[i].is_finite() {
            let mut row = vec![0.0; n];
            row[i] = -1.0;
            a_ineq.push(row);
            b_ineq.push(-lb[i]);
        }
        if ub[i].is_finite() {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            a_ineq.push(row);
            b_ineq.push(ub[i]);
        }
    }

    let problem = LinearProgram {
        f,
        a_ineq,
        b_ineq,
        a_eq,
        b_eq,
    };
    Ok(finalize(solve_linprog(&problem)))
}

#[derive(Clone, Copy)]
enum FiniteMode {
    Finite,
    Bounds,
}

#[derive(Clone)]
struct MatrixInput {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl MatrixInput {
    fn row(&self, row: usize) -> Vec<f64> {
        (0..self.cols)
            .map(|col| self.data[row + col * self.rows])
            .collect()
    }
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(|err| linprog_error_with_detail(&LINPROG_ERROR_INVALID_INPUT, err.message()))
}

fn is_empty_value(value: &Value) -> bool {
    matches!(value, Value::Tensor(t) if t.data.is_empty())
}

async fn numeric_vector(
    label: &str,
    value: Value,
    finite_mode: FiniteMode,
) -> BuiltinResult<Vec<f64>> {
    let value = gather(value).await?;
    if is_empty_value(&value) {
        return Ok(Vec::new());
    }
    let data = match value {
        Value::Num(n) => vec![n],
        Value::Int(i) => vec![i.to_f64()],
        Value::Tensor(t) => {
            let dims = t.shape.len();
            if dims > 2 || (t.rows() != 1 && t.cols() != 1) {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    format!("{label} must be a vector"),
                ));
            }
            t.data
        }
        other => {
            return Err(linprog_error_with_detail(
                &LINPROG_ERROR_INVALID_INPUT,
                format!("{label} must be a real numeric vector, got {other:?}"),
            ))
        }
    };
    validate_numbers(label, &data, finite_mode)?;
    Ok(data)
}

async fn numeric_matrix(label: &str, value: Value) -> BuiltinResult<Option<MatrixInput>> {
    let value = gather(value).await?;
    if is_empty_value(&value) {
        return Ok(None);
    }
    match value {
        Value::Num(n) => {
            validate_numbers(label, &[n], FiniteMode::Finite)?;
            Ok(Some(MatrixInput {
                rows: 1,
                cols: 1,
                data: vec![n],
            }))
        }
        Value::Int(i) => {
            let value = i.to_f64();
            validate_numbers(label, &[value], FiniteMode::Finite)?;
            Ok(Some(MatrixInput {
                rows: 1,
                cols: 1,
                data: vec![value],
            }))
        }
        Value::Tensor(t) => {
            if t.shape.len() > 2 {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    format!("{label} must be a numeric matrix"),
                ));
            }
            validate_numbers(label, &t.data, FiniteMode::Finite)?;
            Ok(Some(MatrixInput {
                rows: t.rows(),
                cols: t.cols(),
                data: t.data,
            }))
        }
        other => Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_INPUT,
            format!("{label} must be a real numeric matrix, got {other:?}"),
        )),
    }
}

fn validate_numbers(label: &str, data: &[f64], finite_mode: FiniteMode) -> BuiltinResult<()> {
    for value in data {
        match finite_mode {
            FiniteMode::Finite if !value.is_finite() => {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    format!("{label} values must be finite"),
                ))
            }
            FiniteMode::Bounds if value.is_nan() => {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    format!("{label} bounds cannot be NaN"),
                ))
            }
            _ => {}
        }
    }
    Ok(())
}

async fn parse_constraint_pair(
    matrix_label: &str,
    matrix: Value,
    rhs_label: &str,
    rhs: Value,
    n: usize,
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let matrix = numeric_matrix(matrix_label, matrix).await?;
    let rhs = numeric_vector(rhs_label, rhs, FiniteMode::Finite).await?;
    match (matrix, rhs.is_empty()) {
        (None, true) => Ok((Vec::new(), Vec::new())),
        (None, false) => Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_INPUT,
            format!("{matrix_label} cannot be empty when {rhs_label} is nonempty"),
        )),
        (Some(matrix), _) => {
            if matrix.cols != n {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    format!("{matrix_label} must have one column per element of f"),
                ));
            }
            if rhs.len() != matrix.rows {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_INPUT,
                    format!("{rhs_label} length must match rows of {matrix_label}"),
                ));
            }
            let rows = (0..matrix.rows).map(|row| matrix.row(row)).collect();
            Ok((rows, rhs))
        }
    }
}

async fn parse_optional_equality(
    aeq: Option<&Value>,
    beq: Option<&Value>,
    n: usize,
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<f64>)> {
    match (aeq, beq) {
        (None, None) => Ok((Vec::new(), Vec::new())),
        (Some(aeq), None) if is_empty_value(aeq) => Ok((Vec::new(), Vec::new())),
        (Some(_), None) => Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_ARGUMENT,
            "Aeq requires a matching beq argument",
        )),
        (None, Some(_)) => Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_ARGUMENT,
            "beq requires a matching Aeq argument",
        )),
        (Some(aeq), Some(beq)) => {
            parse_constraint_pair("Aeq", aeq.clone(), "beq", beq.clone(), n).await
        }
    }
}

async fn parse_bounds(
    lb: Option<&Value>,
    ub: Option<&Value>,
    n: usize,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let lb = match lb {
        None => vec![f64::NEG_INFINITY; n],
        Some(value) if is_empty_value(value) => vec![f64::NEG_INFINITY; n],
        Some(value) => {
            let values = numeric_vector("lb", value.clone(), FiniteMode::Bounds).await?;
            normalize_bound("lb", values, n)?
        }
    };
    let ub = match ub {
        None => vec![f64::INFINITY; n],
        Some(value) if is_empty_value(value) => vec![f64::INFINITY; n],
        Some(value) => {
            let values = numeric_vector("ub", value.clone(), FiniteMode::Bounds).await?;
            normalize_bound("ub", values, n)?
        }
    };
    Ok((lb, ub))
}

fn normalize_bound(label: &str, values: Vec<f64>, n: usize) -> BuiltinResult<Vec<f64>> {
    if values.len() == n {
        Ok(values)
    } else {
        Err(linprog_error_with_detail(
            &LINPROG_ERROR_INVALID_INPUT,
            format!("{label} length must match f"),
        ))
    }
}

struct LinearProgram {
    f: Vec<f64>,
    a_ineq: Vec<Vec<f64>>,
    b_ineq: Vec<f64>,
    a_eq: Vec<Vec<f64>>,
    b_eq: Vec<f64>,
}

#[derive(Clone)]
struct LinprogOutcome {
    x: Option<Vec<f64>>,
    fval: Option<f64>,
    exitflag: i32,
    iterations: usize,
    constrviolation: f64,
    message: String,
}

impl LinprogOutcome {
    fn infeasible(message: &str) -> Self {
        Self {
            x: None,
            fval: None,
            exitflag: -2,
            iterations: 0,
            constrviolation: 0.0,
            message: message.to_string(),
        }
    }

    fn unbounded(iterations: usize) -> Self {
        Self {
            x: None,
            fval: None,
            exitflag: -3,
            iterations,
            constrviolation: 0.0,
            message: "Problem is unbounded.".to_string(),
        }
    }
}

fn solve_linprog(problem: &LinearProgram) -> LinprogOutcome {
    let n = problem.f.len();
    let Some(face) = equality_face(problem, n) else {
        return LinprogOutcome::infeasible("No feasible point found.");
    };
    let reduced = reduce_to_equality_face(problem, &face);
    let k = reduced.f.len();
    let mut candidates = Vec::new();
    let mut combinations = 0usize;

    enumerate_vertices(&reduced, |y| {
        combinations += 1;
        if is_feasible(&reduced, &y) {
            candidates.push(y);
        }
    });

    let feasible_fallback = if candidates.is_empty() {
        let y0 = vec![0.0; k];
        is_feasible(&reduced, &y0).then_some(y0)
    } else {
        None
    };
    let has_feasible_point = !candidates.is_empty() || feasible_fallback.is_some();
    if !has_feasible_point {
        return LinprogOutcome::infeasible("No feasible point found.");
    }
    if has_unbounded_descent_direction(&reduced) {
        return LinprogOutcome::unbounded(combinations);
    }

    if let Some(x) = feasible_fallback {
        candidates.push(x);
    }

    let mut best_y = candidates[0].clone();
    let mut best_fval = dot(&reduced.f, &best_y);
    for candidate in candidates.into_iter().skip(1) {
        let fval = dot(&reduced.f, &candidate);
        if fval < best_fval - TOL {
            best_y = candidate;
            best_fval = fval;
        }
    }

    let best = lift_from_equality_face(&face, &best_y);
    let best_fval = dot(&problem.f, &best);
    let constrviolation = constraint_violation(problem, &best);
    LinprogOutcome {
        x: Some(best),
        fval: Some(best_fval),
        exitflag: 1,
        iterations: combinations,
        constrviolation,
        message: "Optimal solution found.".to_string(),
    }
}

struct EqualityFace {
    x0: Vec<f64>,
    basis: Vec<Vec<f64>>,
}

fn equality_face(problem: &LinearProgram, n: usize) -> Option<EqualityFace> {
    let x0 = if problem.a_eq.is_empty() {
        vec![0.0; n]
    } else {
        pseudo_solve(&problem.a_eq, &problem.b_eq, n)?
    };
    Some(EqualityFace {
        x0,
        basis: nullspace_basis(&problem.a_eq, n),
    })
}

fn reduce_to_equality_face(problem: &LinearProgram, face: &EqualityFace) -> LinearProgram {
    LinearProgram {
        f: face
            .basis
            .iter()
            .map(|basis_vector| dot(&problem.f, basis_vector))
            .collect(),
        a_ineq: problem
            .a_ineq
            .iter()
            .map(|row| {
                face.basis
                    .iter()
                    .map(|basis_vector| dot(row, basis_vector))
                    .collect()
            })
            .collect(),
        b_ineq: problem
            .a_ineq
            .iter()
            .zip(&problem.b_ineq)
            .map(|(row, rhs)| rhs - dot(row, &face.x0))
            .collect(),
        a_eq: Vec::new(),
        b_eq: Vec::new(),
    }
}

fn lift_from_equality_face(face: &EqualityFace, y: &[f64]) -> Vec<f64> {
    let mut x = face.x0.clone();
    for (coeff, basis_vector) in y.iter().zip(&face.basis) {
        for (x_j, basis_j) in x.iter_mut().zip(basis_vector) {
            *x_j += coeff * basis_j;
        }
    }
    x
}

fn enumerate_vertices(problem: &LinearProgram, mut visit: impl FnMut(Vec<f64>)) {
    let n = problem.f.len();
    let max_active = problem.a_ineq.len().min(n);
    for active_count in 0..=max_active {
        enumerate_combinations(problem.a_ineq.len(), active_count, |active| {
            let mut rows = problem.a_eq.clone();
            let mut rhs = problem.b_eq.clone();
            for &idx in active {
                rows.push(problem.a_ineq[idx].clone());
                rhs.push(problem.b_ineq[idx]);
            }
            if let Some(x) = pseudo_solve(&rows, &rhs, n) {
                visit(x);
            }
        });
    }
}

fn has_unbounded_descent_direction(problem: &LinearProgram) -> bool {
    let n = problem.f.len();
    let max_active = problem.a_ineq.len().min(n.saturating_sub(1));
    for active_count in 0..=max_active {
        let mut found = false;
        enumerate_combinations(problem.a_ineq.len(), active_count, |active| {
            if found {
                return;
            }
            let mut rows = problem.a_eq.clone();
            for &idx in active {
                rows.push(problem.a_ineq[idx].clone());
            }
            for direction in candidate_nullspace_descent_directions(&rows, &problem.f, n) {
                if is_recession_direction(problem, &direction) && dot(&problem.f, &direction) < -TOL
                {
                    found = true;
                    return;
                }
            }
        });
        if found {
            return true;
        }
    }
    false
}

fn candidate_nullspace_descent_directions(rows: &[Vec<f64>], f: &[f64], n: usize) -> Vec<Vec<f64>> {
    let basis = nullspace_basis(rows, n);
    if basis.is_empty() {
        return Vec::new();
    }
    let mut directions = Vec::new();
    let mut projected = vec![0.0; n];
    for basis_vector in &basis {
        let coeff = -dot(f, basis_vector);
        for i in 0..n {
            projected[i] += coeff * basis_vector[i];
        }
        directions.push(basis_vector.clone());
        directions.push(basis_vector.iter().map(|v| -*v).collect());
    }
    if norm(&projected) > TOL {
        directions.push(projected);
    }
    directions
}

fn is_recession_direction(problem: &LinearProgram, direction: &[f64]) -> bool {
    norm(direction) > TOL
        && problem
            .a_eq
            .iter()
            .all(|row| dot(row, direction).abs() <= TOL)
        && problem.a_ineq.iter().all(|row| dot(row, direction) <= TOL)
}

fn is_feasible(problem: &LinearProgram, x: &[f64]) -> bool {
    constraint_violation(problem, x) <= 1.0e-7
}

fn constraint_violation(problem: &LinearProgram, x: &[f64]) -> f64 {
    let eq = problem
        .a_eq
        .iter()
        .zip(&problem.b_eq)
        .map(|(row, rhs)| (dot(row, x) - rhs).abs())
        .fold(0.0, f64::max);
    let ineq = problem
        .a_ineq
        .iter()
        .zip(&problem.b_ineq)
        .map(|(row, rhs)| (dot(row, x) - rhs).max(0.0))
        .fold(0.0, f64::max);
    eq.max(ineq)
}

fn nullspace_basis(rows: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    if n == 0 {
        return Vec::new();
    }
    if rows.is_empty() {
        return (0..n)
            .map(|i| {
                let mut basis = vec![0.0; n];
                basis[i] = 1.0;
                basis
            })
            .collect();
    }

    let (reduced, pivots) = rref(rows, n);
    let free_cols = (0..n)
        .filter(|col| !pivots.contains(col))
        .collect::<Vec<_>>();
    free_cols
        .into_iter()
        .filter_map(|free_col| {
            let mut basis = vec![0.0; n];
            basis[free_col] = 1.0;
            for (row, pivot_col) in pivots.iter().enumerate() {
                basis[*pivot_col] = -reduced[row][free_col];
            }
            let length = norm(&basis);
            (length > TOL).then(|| basis.into_iter().map(|value| value / length).collect())
        })
        .collect()
}

fn rref(rows: &[Vec<f64>], n: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut matrix = rows.to_vec();
    let mut pivots = Vec::new();
    let mut pivot_row = 0usize;

    for col in 0..n {
        let Some(best_row) = (pivot_row..matrix.len()).max_by(|&a, &b| {
            matrix[a][col]
                .abs()
                .partial_cmp(&matrix[b][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            break;
        };
        if matrix[best_row][col].abs() <= TOL {
            continue;
        }

        matrix.swap(pivot_row, best_row);
        let pivot = matrix[pivot_row][col];
        for value in &mut matrix[pivot_row] {
            *value /= pivot;
        }

        for row in 0..matrix.len() {
            if row == pivot_row {
                continue;
            }
            let factor = matrix[row][col];
            if factor.abs() <= TOL {
                continue;
            }
            for j in col..n {
                matrix[row][j] -= factor * matrix[pivot_row][j];
            }
        }

        pivots.push(col);
        pivot_row += 1;
        if pivot_row == matrix.len() {
            break;
        }
    }

    (matrix, pivots)
}

fn pseudo_solve(rows: &[Vec<f64>], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    if rows.is_empty() {
        return Some(vec![0.0; n]);
    }
    let matrix = dmatrix_from_rows(rows, n);
    let rhs_vec = DVector::from_column_slice(rhs);
    let svd = matrix.svd(true, true);
    let u = svd.u.as_ref()?;
    let v_t = svd.v_t.as_ref()?;
    let mut x = vec![0.0; n];
    for (i, sigma) in svd.singular_values.iter().enumerate() {
        if *sigma <= TOL {
            continue;
        }
        let coeff = (0..rows.len())
            .map(|row| u[(row, i)] * rhs_vec[row])
            .sum::<f64>()
            / sigma;
        for col in 0..n {
            x[col] += v_t[(i, col)] * coeff;
        }
    }
    let residual = rows
        .iter()
        .zip(rhs)
        .map(|(row, target)| (dot(row, &x) - target).abs())
        .fold(0.0, f64::max);
    (residual <= 1.0e-7).then_some(x)
}

fn dmatrix_from_rows(rows: &[Vec<f64>], n: usize) -> DMatrix<f64> {
    let data = rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    DMatrix::from_row_slice(rows.len(), n, &data)
}

fn enumerate_combinations(len: usize, choose: usize, mut visit: impl FnMut(&[usize])) {
    fn rec(
        len: usize,
        choose: usize,
        start: usize,
        current: &mut Vec<usize>,
        visit: &mut dyn FnMut(&[usize]),
    ) {
        if current.len() == choose {
            visit(current);
            return;
        }
        let remaining = choose - current.len();
        for idx in start..=len - remaining {
            current.push(idx);
            rec(len, choose, idx + 1, current, visit);
            current.pop();
        }
    }

    if choose > len {
        return;
    }
    let mut current = Vec::with_capacity(choose);
    rec(len, choose, 0, &mut current, &mut visit);
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

fn finalize(outcome: LinprogOutcome) -> Value {
    let x = outcome
        .x
        .clone()
        .map(vector_value)
        .unwrap_or_else(empty_double);
    let fval = outcome.fval.map(Value::Num).unwrap_or_else(empty_double);
    let exitflag = Value::Num(outcome.exitflag as f64);
    let output = Value::Struct(build_output_struct(&outcome));

    match crate::output_count::current_output_count() {
        None => x,
        Some(0) => Value::OutputList(Vec::new()),
        Some(1) => crate::output_count::output_list_with_padding(1, vec![x]),
        Some(2) => crate::output_count::output_list_with_padding(2, vec![x, fval]),
        Some(3) => crate::output_count::output_list_with_padding(3, vec![x, fval, exitflag]),
        Some(n) if n >= 4 => {
            crate::output_count::output_list_with_padding(n, vec![x, fval, exitflag, output])
        }
        Some(_) => x,
    }
}

fn vector_value(values: Vec<f64>) -> Value {
    let n = values.len();
    Tensor::new(values, vec![n, 1])
        .map(Value::Tensor)
        .unwrap_or_else(|_| empty_double())
}

fn empty_double() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

fn build_output_struct(outcome: &LinprogOutcome) -> StructValue {
    let mut fields = StructValue::new();
    fields.insert("iterations", Value::Num(outcome.iterations as f64));
    fields.insert("algorithm", Value::from(ALGORITHM));
    fields.insert("constrviolation", Value::Num(outcome.constrviolation));
    fields.insert("message", Value::from(outcome.message.clone()));
    fields
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::Value as V;

    fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> V {
        V::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
    }

    fn empty() -> V {
        V::Tensor(Tensor::zeros(vec![0, 0]))
    }

    fn run(f: V, a: V, b: V, rest: Vec<V>, outputs: usize) -> Vec<V> {
        let _guard = crate::output_count::push_output_count(Some(outputs));
        let value = block_on(linprog_builtin(f, a, b, rest)).expect("linprog");
        match value {
            V::OutputList(values) => values,
            other => vec![other],
        }
    }

    #[test]
    fn solves_bounded_feasible_problem() {
        let outputs = run(
            tensor(vec![-1.0, -2.0], 2, 1),
            tensor(vec![1.0, 1.0], 1, 2),
            V::Num(4.0),
            vec![empty(), empty(), tensor(vec![0.0, 0.0], 2, 1), empty()],
            3,
        );
        match &outputs[0] {
            V::Tensor(x) => {
                assert!((x.data[0] - 0.0).abs() < 1.0e-7, "{x:?}");
                assert!((x.data[1] - 4.0).abs() < 1.0e-7, "{x:?}");
            }
            other => panic!("unexpected x {other:?}"),
        }
        assert!(matches!(&outputs[1], V::Num(fval) if (*fval + 8.0).abs() < 1.0e-7));
        assert!(matches!(&outputs[2], V::Num(flag) if *flag == 1.0));
    }

    #[test]
    fn solves_equality_constrained_problem() {
        let outputs = run(
            tensor(vec![1.0, 2.0], 2, 1),
            empty(),
            empty(),
            vec![
                tensor(vec![1.0, 1.0], 1, 2),
                V::Num(3.0),
                tensor(vec![1.0, 0.0], 2, 1),
                empty(),
            ],
            2,
        );
        match &outputs[0] {
            V::Tensor(x) => {
                assert!((x.data[0] - 3.0).abs() < 1.0e-7, "{x:?}");
                assert!((x.data[1] - 0.0).abs() < 1.0e-7, "{x:?}");
            }
            other => panic!("unexpected x {other:?}"),
        }
        assert!(matches!(&outputs[1], V::Num(fval) if (*fval - 3.0).abs() < 1.0e-7));
    }

    #[test]
    fn reports_infeasible_bounds() {
        let outputs = run(
            V::Num(1.0),
            empty(),
            empty(),
            vec![empty(), empty(), V::Num(2.0), V::Num(1.0)],
            4,
        );
        assert!(matches!(&outputs[0], V::Tensor(t) if t.data.is_empty()));
        assert!(matches!(&outputs[1], V::Tensor(t) if t.data.is_empty()));
        assert!(matches!(&outputs[2], V::Num(flag) if *flag == -2.0));
        assert!(matches!(&outputs[3], V::Struct(s) if s.fields.contains_key("message")));
    }

    #[test]
    fn reports_unbounded_problem() {
        let outputs = run(V::Num(-1.0), empty(), empty(), Vec::new(), 3);
        assert!(matches!(&outputs[0], V::Tensor(t) if t.data.is_empty()));
        assert!(matches!(&outputs[1], V::Tensor(t) if t.data.is_empty()));
        assert!(matches!(&outputs[2], V::Num(flag) if *flag == -3.0));
    }

    #[test]
    fn accepts_empty_optional_placeholders() {
        let outputs = run(
            tensor(vec![1.0, 1.0], 2, 1),
            empty(),
            empty(),
            vec![empty(), empty(), tensor(vec![2.0, 3.0], 2, 1), empty()],
            2,
        );
        match &outputs[0] {
            V::Tensor(x) => {
                assert!((x.data[0] - 2.0).abs() < 1.0e-7, "{x:?}");
                assert!((x.data[1] - 3.0).abs() < 1.0e-7, "{x:?}");
            }
            other => panic!("unexpected x {other:?}"),
        }
        assert!(matches!(&outputs[1], V::Num(fval) if (*fval - 5.0).abs() < 1.0e-7));
    }

    #[test]
    fn solves_one_sided_bound_with_fewer_rows_than_variables() {
        let outputs = run(
            tensor(vec![1.0, 0.0], 2, 1),
            empty(),
            empty(),
            vec![
                empty(),
                empty(),
                tensor(vec![2.0, f64::NEG_INFINITY], 2, 1),
                empty(),
            ],
            3,
        );
        match &outputs[0] {
            V::Tensor(x) => {
                assert!((x.data[0] - 2.0).abs() < 1.0e-7, "{x:?}");
                assert!(x.data[1].abs() < 1.0e-7, "{x:?}");
            }
            other => panic!("unexpected x {other:?}"),
        }
        assert!(matches!(&outputs[1], V::Num(fval) if (*fval - 2.0).abs() < 1.0e-7));
        assert!(matches!(&outputs[2], V::Num(flag) if *flag == 1.0));
    }

    #[test]
    fn optimizes_along_equality_face_when_particular_solution_is_suboptimal() {
        let outputs = run(
            tensor(vec![-1.0, 0.0, 0.0], 3, 1),
            tensor(vec![1.0, 0.0, 0.0], 1, 3),
            V::Num(1.0),
            vec![
                tensor(vec![0.0, 0.0, 1.0], 1, 3),
                V::Num(0.0),
                empty(),
                empty(),
            ],
            3,
        );
        match &outputs[0] {
            V::Tensor(x) => {
                assert!((x.data[0] - 1.0).abs() < 1.0e-7, "{x:?}");
                assert!(x.data[1].abs() < 1.0e-7, "{x:?}");
                assert!(x.data[2].abs() < 1.0e-7, "{x:?}");
            }
            other => panic!("unexpected x {other:?}"),
        }
        assert!(matches!(&outputs[1], V::Num(fval) if (*fval + 1.0).abs() < 1.0e-7));
        assert!(matches!(&outputs[2], V::Num(flag) if *flag == 1.0));
    }

    #[test]
    fn validates_matrix_dimensions() {
        let err = block_on(linprog_builtin(
            tensor(vec![1.0, 1.0], 2, 1),
            tensor(vec![1.0, 1.0, 1.0], 1, 3),
            V::Num(1.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:linprog:InvalidInput"));
    }
}
