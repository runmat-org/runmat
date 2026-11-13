//! One invocation-lifetime numerical phase; original controllers unchanged.
use super::*;
pub(super) async fn solve(
    problem: Problem,
    prepared: PreparedOptions,
    oracle: &mut dyn Oracle,
    policy: super::super::numeric::Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Outcome> {
    crate::context::host_allocation::require_unbounded_workspace(
        cooperation.runtime(),
        policy.owner.name,
        "modeled nonlinear evaluation and constrained factorization require unbounded host memory",
    )?;
    let PreparedOptions { controls } = prepared;
    let n = problem.x.len();
    if n == 0 || problem.lower.len() != n || problem.upper.len() != n {
        return Err(super::super::numeric::invalid(
            &policy,
            "modeled geometry",
            "inconsistent decision-coordinate extents",
        ));
    }
    for ((&x, &lower), &upper) in problem.x.iter().zip(&problem.lower).zip(&problem.upper) {
        if !x.is_finite()
            || lower.is_nan()
            || upper.is_nan()
            || lower > upper
            || lower == f64::INFINITY
            || upper == f64::NEG_INFINITY
        {
            return Err(super::super::numeric::invalid(
                &policy,
                "modeled geometry",
                "invalid initial coordinate or bounds",
            ));
        }
        cooperation.completed(1).await?;
    }
    controls.scaling.validate_variables(policy.owner.name, n)?;
    let limit = match controls.point_limit {
        Some(limit) => limit,
        None => n
            .checked_mul(100)
            .ok_or_else(|| super::super::invalid("evaluation limit overflow"))?,
    };
    let geometry = Geometry {
        variables: n,
        inequality: LinearConstraints {
            matrix: CscMatrix::zeros((0, n)),
            rhs: Vec::new(),
        },
        equality: LinearConstraints {
            matrix: CscMatrix::zeros((0, n)),
            rhs: Vec::new(),
        },
        lower: problem.lower,
        upper: problem.upper,
    };
    let parameters = Parameters {
        max_iterations: controls.max_iterations,
        feasibility: controls.feasibility,
        optimality: controls.optimality,
        step: controls.step,
        function: None,
        differences: controls.differences,
        callback: controls.callback(),
        display: controls.display,
    };
    let mut evaluator = Evaluator {
        internal: Some(oracle),
        attainment: None,
        semi_infinite: None,
        scaling: controls.scaling.clone(),
        cooperation,
        objective: Objective::Internal,
        feasibility_limit: None,
        selected_constraints: None,
        constraints: None,
        policy,
        shape: vec![n, 1],
        scalar: false,
        objective_gradient: false,
        constraint_gradient: false,
        budget: Budget::new(limit),
        constraint_extents: Some((problem.inequalities, problem.equalities)),
    };
    let numerical = super::algorithm::select(
        controls.algorithm,
        problem.x,
        &geometry,
        &parameters,
        &mut evaluator,
    )
    .await?;
    drop(evaluator);
    Ok(Outcome {
        numerical,
        variables: n,
    })
}
