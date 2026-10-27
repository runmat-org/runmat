//! Recognize once on the original Plan, then compile only selected affine
//! residual/constraint ancestry while retaining the full original layout/DAG.
use super::*;
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn compile(
    formulation: &formulation::Formulation,
    selected_solver: Option<runmat_builtins::OptimSolver>,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Prepared> {
    let source = evaluation::Plan::prepare_roots(&formulation.roots, policy, c).await?;
    let layout = layout::Layout::prepare(&source, policy, c).await?;
    if matches!(
        selected_solver,
        Some(runmat_builtins::OptimSolver::Fmincon | runmat_builtins::OptimSolver::Fminunc)
    ) {
        return Ok(Ok(Prepared::Nonlinear(
            nonlinear::Model::prepare(source, layout, formulation, None, policy, c).await?,
        )));
    }
    let classified = match &formulation.objective {
        Some(objective) => classify(&source, objective.root, policy, c).await?,
        None => Classification::Unrecognized,
    };
    let squared = match classified {
        Classification::Unrecognized => {
            if selected_solver == Some(runmat_builtins::OptimSolver::Lsqnonlin) {
                return Err(policy.invalid(
                    "lsqnonlin requires an original positive squared-residual objective",
                ));
            }
            return super::super::quadratic::prepare_or_affine(
                source,
                layout,
                formulation,
                policy,
                c,
            )
            .await;
        }
        Classification::Unrepresentable => return Err(super::super::run::unverified(
            policy,
            "Original squared-residual weights are not representable in solver working precision.",
        )),
        Classification::LeastSquares(value) => value,
    };
    if selected_solver == Some(runmat_builtins::OptimSolver::Lsqnonlin) {
        return Ok(Ok(Prepared::Nonlinear(
            nonlinear::Model::prepare(source, layout, formulation, Some(squared), policy, c)
                .await?,
        )));
    }
    let mut roots = Vec::new();
    for term in &squared.terms {
        roots.try_reserve(1).map_err(|e| policy.reserve(e))?;
        roots.push(term.residual);
        c.completed(1).await?;
    }
    for constraint in &formulation.constraints {
        roots.try_reserve(2).map_err(|e| policy.reserve(e))?;
        roots.extend([constraint.left, constraint.right]);
        c.completed(1).await?;
    }
    let selected = selection::Selection::prepare(&source, roots, policy, c).await?;
    let compiled = engine::prepare(
        source,
        layout,
        Some(selected),
        Some(squared),
        &[],
        formulation,
        policy,
        c,
    )
    .await?;
    let program = match compiled {
        Ok(Prepared::Algebraic(program)) => program,
        Ok(value @ Prepared::Nonlinear(_)) => return Ok(Ok(value)),
        Err(stop) => return Ok(Err(stop)),
    };
    let mut quadratic = formulation.objective.as_ref().is_some_and(|objective| {
        objective.sense == runmat_value::optimization_model::OptimizationObjectiveSense::Maximize
    });
    if let ModelObjective::LeastSquares(squares) = &program.objective {
        for term in &squares.terms {
            quadratic |= !term.weight.exact.is_positive();
            c.completed(1).await?;
        }
        if quadratic {
            return Ok(
                super::super::quadratic::from_squares(program, formulation, policy, c)
                    .await?
                    .map(Prepared::Algebraic),
            );
        }
    }
    Ok(Ok(Prepared::Algebraic(program)))
}
