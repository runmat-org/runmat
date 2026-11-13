//! Public positional/problem grammar, original sparse rows and bound identity.
use super::{model, numeric, options, workspace};
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::BuiltinResult;
use clarabel::algebra::CscMatrix;
use runmat_value::Value;

pub(super) struct Problem {
    pub policy: numeric::Policy,
    pub objective: Value,
    pub constraints: Option<Value>,
    pub x: Vec<f64>,
    pub shape: Vec<usize>,
    pub scalar: bool,
    pub geometry: model::Geometry,
    pub controls: options::Controls,
}

fn empty() -> Value {
    super::empty()
}
pub(super) fn callable(value: &Value) -> bool {
    matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
            | Value::String(_)
            | Value::CharArray(_)
    )
}

pub(super) async fn problem(
    args: crate::call::BuiltinArguments,
    mut policy: numeric::Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Problem> {
    let values = if args.len() == 1 {
        let Value::Struct(value) = args.into_iter().next().unwrap() else {
            return Err(super::invalid(
                "single-argument form requires a scalar problem structure",
            ));
        };
        for field in ["objective", "x0", "solver", "options"] {
            if !value.fields.contains_key(field) {
                return Err(super::invalid(format!("problem requires {field}")));
            }
        }
        if runmat_builtins::OptimSolver::parse(&common::field_name(&value.fields["solver"])?)
            != Some(runmat_builtins::OptimSolver::Fmincon)
        {
            return Err(super::invalid("problem solver must be fmincon"));
        }
        [
            "objective",
            "x0",
            "Aineq",
            "bineq",
            "Aeq",
            "beq",
            "lb",
            "ub",
            "nonlcon",
            "options",
        ]
        .into_iter()
        .map(|name| value.fields.get(name).cloned().unwrap_or_else(empty))
        .collect::<Vec<_>>()
    } else {
        if !matches!(args.len(), 4 | 6 | 8 | 9 | 10) {
            return Err(super::invalid(
                "expected fun,x0,A,b, complete equality/bound pairs, optional nonlcon and options",
            ));
        }
        let mut values = args.into_iter().collect::<Vec<_>>();
        values.resize_with(10, empty);
        values
    };
    if !callable(&values[0]) {
        return Err(super::invalid(
            "objective must be a single function handle or function name",
        ));
    }
    let no_constraints = matches!(&values[8], Value::Tensor(value) if value.is_empty());
    if !no_constraints && !callable(&values[8]) {
        return Err(super::invalid(
            "nonlcon must be [] or a single function handle or function name",
        ));
    }
    let algorithm = options::Controls::selected_algorithm(&values[9])?;
    if algorithm == runmat_builtins::OptimAlgorithm::TrustRegionReflective && !no_constraints {
        return Err(super::invalid(
            "trust-region-reflective does not support nonlinear constraints",
        ));
    }
    policy.sparse = matches!(
        algorithm,
        runmat_builtins::OptimAlgorithm::Sqp | runmat_builtins::OptimAlgorithm::ActiveSet
    )
    .then_some(&runmat_builtins::FMINCON_SPARSE_EXTENSION);
    // Concrete role checks precede every input/control transfer. No closure,
    // object property or callback capture is traversed for numeric admission.
    for (index, role, geometry) in [
        (1, "x0", numeric::Geometry::Array),
        (2, "A", numeric::Geometry::MatrixAny),
        (3, "b", numeric::Geometry::Vector),
        (4, "Aeq", numeric::Geometry::MatrixAny),
        (5, "beq", numeric::Geometry::Vector),
        (6, "lb", numeric::Geometry::Array),
        (7, "ub", numeric::Geometry::Array),
    ] {
        if !numeric::preflight(
            &values[index],
            role,
            geometry,
            &policy,
            cooperation.runtime(),
        )? {
            return Err(super::invalid(format!("{role} must be real")));
        }
    }
    if algorithm == runmat_builtins::OptimAlgorithm::TrustRegionReflective
        && numeric::shape(&values[2]).is_some_and(|shape| shape.iter().all(|&axis| axis != 0))
    {
        return Err(super::invalid(
            "trust-region-reflective does not support linear inequalities",
        ));
    }
    let mut values = values.into_iter();
    let objective = values.next().unwrap();
    let x0 = values.next().unwrap();
    let a = values.next().unwrap();
    let b = values.next().unwrap();
    let aeq = values.next().unwrap();
    let beq = values.next().unwrap();
    let lower = values.next().unwrap();
    let upper = values.next().unwrap();
    let nonlinear = values.next().unwrap();
    let controls = values.next().unwrap();
    let controls = options::Controls::parse(controls, !no_constraints, cooperation).await?;
    let scalar = matches!(&x0, Value::Num(_) | Value::Int(_) | Value::Bool(_));
    let numeric::Numeric::Real { shape, values: x } =
        numeric::read(x0, "x0", numeric::Geometry::Array, &policy, cooperation).await?
    else {
        return Err(super::invalid("x0 must contain finite real values"));
    };
    if x.is_empty() {
        return Err(super::invalid("x0 must be nonempty"));
    }
    let inequality = pair(a, b, x.len(), "A/b", &policy, cooperation).await?;
    let equality = pair(aeq, beq, x.len(), "Aeq/beq", &policy, cooperation).await?;
    let lower = bound(
        lower,
        x.len(),
        f64::NEG_INFINITY,
        "lb",
        &policy,
        cooperation,
    )
    .await?;
    let upper = bound(upper, x.len(), f64::INFINITY, "ub", &policy, cooperation).await?;
    for (&lower, &upper) in lower.iter().zip(&upper) {
        if lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY {
            return Err(super::invalid(
                "bounds are inconsistent or have an invalid infinite side",
            ));
        }
        cooperation.completed(1).await?;
    }
    Ok(Problem {
        policy,
        objective,
        constraints: (!no_constraints).then_some(nonlinear),
        shape,
        scalar,
        x,
        geometry: model::Geometry {
            variables: lower.len(),
            inequality,
            equality,
            lower,
            upper,
        },
        controls,
    })
}

pub(super) async fn pair(
    a: Value,
    b: Value,
    n: usize,
    role: &str,
    policy: &numeric::Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<model::LinearConstraints> {
    let Some(a) =
        numeric::prepare(a, role, numeric::Geometry::MatrixAny, policy, cooperation).await?
    else {
        return Err(argument_error(policy, format!("{role} must be real")));
    };
    let matrix = common::conic::matrix::parse(
        a,
        common::conic::matrix::IntegerPolicy::Exact,
        cooperation,
        |detail| numeric::invalid(policy, role, detail),
    )
    .await?;
    let numeric::Numeric::Real { values: rhs, .. } =
        numeric::read(b, role, numeric::Geometry::Vector, policy, cooperation).await?
    else {
        return Err(argument_error(policy, format!("{role} must be finite")));
    };
    let matrix = match matrix {
        None if rhs.is_empty() => CscMatrix::zeros((0, n)),
        Some(matrix) if matrix.n == n && matrix.m == rhs.len() => matrix,
        _ => {
            return Err(argument_error(
                policy,
                format!("{role} row counts and variable dimensions must agree"),
            ))
        }
    };
    Ok(model::LinearConstraints { matrix, rhs })
}

pub(super) async fn bound(
    value: Value,
    n: usize,
    default: f64,
    role: &str,
    policy: &numeric::Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let Some(value) =
        numeric::prepare(value, role, numeric::Geometry::Array, policy, cooperation).await?
    else {
        return Err(argument_error(policy, format!("{role} must be real")));
    };
    let numeric::Numeric::Real {
        values: mut bound, ..
    } = numeric::materialize_raw(value, role, policy, cooperation).await?
    else {
        return Err(argument_error(policy, format!("{role} must be real")));
    };
    if bound.len() > n || bound.iter().any(|value| value.is_nan()) {
        return Err(argument_error(
            policy,
            format!("{role} must have at most numel(x0) elements and cannot contain NaN"),
        ));
    }
    if !bound.is_empty() && bound.len() < n {
        let _entry = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::console::record_console_line(crate::console::ConsoleStream::Stderr,
            format!("Warning: {}: {role} has fewer elements than x0; remaining components are unbounded.", policy.owner.name))?;
    }
    while bound.len() < n {
        workspace::push(&mut bound, default, policy.owner)?;
        cooperation.completed(1).await?;
    }
    Ok(bound)
}

fn argument_error(policy: &numeric::Policy, detail: impl AsRef<str>) -> crate::RuntimeError {
    crate::build_runtime_error(format!("{}: {}", policy.owner.name, detail.as_ref()))
        .with_builtin(policy.owner.name)
        .with_identifier(format!("RunMat:{}:InvalidArgument", policy.owner.name))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
