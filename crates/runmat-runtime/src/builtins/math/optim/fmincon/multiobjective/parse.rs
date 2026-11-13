//! The shared constrained data parser retains original public x geometry.
//! Only the numerical model receives the additional free attainment coordinate.
use super::super::{model, numeric, parse, workspace};
use crate::builtins::math::optim::common::{
    multiobjective::{
        controls::Controls,
        formulation::Specification,
        inputs::{Goals, Inputs},
        invalid, Kind,
    },
    ScalarCooperation,
};
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) struct Prepared {
    pub objective: Value,
    pub constraints: Option<Value>,
    pub x: Vec<f64>,
    pub shape: Vec<usize>,
    pub scalar: bool,
    pub geometry: model::Geometry,
    pub controls: Controls,
    pub specification: Specification,
    pub inconsistent_bounds: bool,
}
pub(super) async fn prepare(
    kind: Kind,
    input: Inputs,
    policy: &numeric::Policy,
    option_numeric: &'static runmat_builtins::BuiltinExtensionDescriptor,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Prepared> {
    if !parse::callable(&input.objective) {
        return Err(invalid(
            kind,
            "objective must be a single function handle or function name",
        ));
    }
    let nonlinear = !matches!(&input.nonlinear, Value::Tensor(value) if value.is_empty());
    if nonlinear && !parse::callable(&input.nonlinear) {
        return Err(invalid(kind, "nonlcon must be [] or a single callable"));
    }
    // All required data-role metadata precedes provider transfer. This never
    // derives facts from callbacks, closure captures or arbitrary structures.
    for (value, role, geometry) in [
        (&input.x0, "x0", numeric::Geometry::Array),
        (&input.a, "A", numeric::Geometry::MatrixAny),
        (&input.b, "b", numeric::Geometry::Vector),
        (&input.aeq, "Aeq", numeric::Geometry::MatrixAny),
        (&input.beq, "beq", numeric::Geometry::Vector),
        (&input.lower, "lb", numeric::Geometry::Array),
        (&input.upper, "ub", numeric::Geometry::Array),
    ] {
        if !numeric::preflight(value, role, geometry, policy, cooperation.runtime())? {
            return Err(invalid(kind, format!("{role} must be real")));
        }
    }
    if let Goals::Attainment { goal, weight } = &input.goals {
        for (value, role) in [(goal, "goal"), (weight, "weight")] {
            if !numeric::preflight(
                value,
                role,
                numeric::Geometry::Vector,
                policy,
                cooperation.runtime(),
            )? {
                return Err(invalid(kind, format!("{role} must be a real vector")));
            }
        }
    }
    let controls = Controls::parse(
        kind,
        input.options,
        nonlinear,
        option_numeric,
        policy.resident,
        cooperation,
    )
    .await?;
    let scalar = matches!(&input.x0, Value::Num(_) | Value::Int(_) | Value::Bool(_));
    let numeric::Numeric::Real {
        shape,
        values: mut x,
    } = numeric::read(
        input.x0,
        "x0",
        numeric::Geometry::Array,
        policy,
        cooperation,
    )
    .await?
    else {
        return Err(invalid(kind, "x0 must contain finite real values"));
    };
    if x.is_empty() {
        return Err(invalid(kind, "x0 must be nonempty"));
    }
    let specification = match input.goals {
        Goals::Minimax => Specification::Minimax {
            absolute: controls.first_objectives,
        },
        Goals::Attainment { goal, weight } => {
            let numeric::Numeric::Real { values: goals, .. } =
                numeric::read(goal, "goal", numeric::Geometry::Vector, policy, cooperation).await?
            else {
                return Err(invalid(kind, "goal must contain finite real values"));
            };
            let numeric::Numeric::Real {
                values: weights, ..
            } = numeric::read(
                weight,
                "weight",
                numeric::Geometry::Vector,
                policy,
                cooperation,
            )
            .await?
            else {
                return Err(invalid(kind, "weight must contain finite real values"));
            };
            if goals.is_empty() || goals.len() != weights.len() {
                return Err(invalid(
                    kind,
                    "goal and weight must be nonempty vectors of equal extent",
                ));
            }
            if controls.first_objectives > goals.len() {
                return Err(invalid(
                    kind,
                    "EqualityGoalCount exceeds the objective goal extent",
                ));
            }
            Specification::Goal {
                goals,
                weights,
                equal: controls.first_objectives,
            }
        }
    };
    let variables = x.len();
    let mut inequality =
        parse::pair(input.a, input.b, variables, "A/b", policy, cooperation).await?;
    let mut equality = parse::pair(
        input.aeq,
        input.beq,
        variables,
        "Aeq/beq",
        policy,
        cooperation,
    )
    .await?;
    let mut lower = parse::bound(
        input.lower,
        variables,
        f64::NEG_INFINITY,
        "lb",
        policy,
        cooperation,
    )
    .await?;
    let mut upper = parse::bound(
        input.upper,
        variables,
        f64::INFINITY,
        "ub",
        policy,
        cooperation,
    )
    .await?;
    let mut inconsistent_bounds = false;
    for (&lower, &upper) in lower.iter().zip(&upper) {
        if lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY {
            inconsistent_bounds = true;
        }
        cooperation.completed(1).await?;
    }
    let augmented = variables
        .checked_add(1)
        .ok_or_else(|| invalid(kind, "attainment variable extent overflow"))?;
    // Add one empty CSC column, not dense bounds or linear-constraint rows.
    for matrix in [&mut inequality.matrix, &mut equality.matrix] {
        workspace::push(&mut matrix.colptr, matrix.nzval.len(), policy.owner)?;
        matrix.n = augmented;
    }
    workspace::push(&mut x, 0.0, policy.owner)?;
    workspace::push(&mut lower, f64::NEG_INFINITY, policy.owner)?;
    workspace::push(&mut upper, f64::INFINITY, policy.owner)?;
    Ok(Prepared {
        objective: input.objective,
        constraints: nonlinear.then_some(input.nonlinear),
        x,
        shape,
        scalar,
        geometry: model::Geometry {
            variables: augmented,
            inequality,
            equality,
            lower,
            upper,
        },
        controls,
        specification,
        inconsistent_bounds,
    })
}
