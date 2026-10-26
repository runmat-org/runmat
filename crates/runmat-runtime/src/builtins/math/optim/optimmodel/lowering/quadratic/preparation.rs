//! Exact quadratic choice over the already prepared original Plan/Layout.
use super::*;
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn prepare_or_affine(
    source: evaluation::Plan,
    layout: layout::Layout,
    formulation: &formulation::Formulation,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<super::super::Prepared> {
    let selected = match &formulation.objective {
        Some(objective) => candidate(&source, objective.root, policy, c).await?,
        None => None,
    };
    let Some(selected) = selected else {
        return super::super::engine::prepare(
            source,
            layout,
            None,
            None,
            &formulation.roots,
            formulation,
            policy,
            c,
        )
        .await;
    };
    let compiled = match compile(&source, &layout, selected, None, policy, c).await? {
        Ok(value) => value,
        Err(stop) => {
            return super::super::engine::nonlinear_or_stop(
                source,
                layout,
                formulation,
                None,
                stop,
                policy,
                c,
            )
            .await
        }
    };
    let sense = formulation
        .objective
        .as_ref()
        .expect("selected quadratic objective")
        .sense;
    let Prepared {
        cost,
        evidence,
        facts,
    } = match compiled.assemble(layout.columns, sense, policy, c).await? {
        Ok(value) => value,
        Err(stop) => return Ok(Err(stop)),
    };
    let mut roots = Vec::new();
    for constraint in &formulation.constraints {
        roots.try_reserve(2).map_err(|e| policy.reserve(e))?;
        roots.extend([constraint.left, constraint.right]);
        c.completed(1).await?;
    }
    let selected = selection::Selection::prepare(&source, roots, policy, c).await?;
    let mut program = match super::super::engine::prepare(
        source,
        layout,
        Some(selected),
        None,
        &[],
        formulation,
        policy,
        c,
    )
    .await?
    {
        Ok(super::super::Prepared::Algebraic(value)) => value,
        Ok(value @ super::super::Prepared::Nonlinear(_)) => return Ok(Ok(value)),
        Err(stop) => return Ok(Err(stop)),
    };
    program
        .facts
        .try_reserve(facts.len())
        .map_err(|e| policy.reserve(e))?;
    for (handle, fact) in facts {
        // The affine constraint fact is conservative for a shared source row;
        // otherwise move the exact quadratic dependence/exponent evidence.
        program.facts.entry(handle).or_insert(fact);
        c.completed(1).await?;
    }
    program.objective = objective::ModelObjective::Quadratic(
        crate::builtins::math::optim::common::retained_control::Retained::new(
            Model {
                cost: Some(cost),
                evidence,
            },
            c,
            crate::builtins::math::optim::common::retained_control::Policy {
                identifier: "RunMat:OptimModel:Allocation",
                operation: "quadratic model control allocation",
            },
        )?,
    );
    Ok(Ok(super::super::Prepared::Algebraic(program)))
}
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn from_squares(
    mut program: Program,
    formulation: &formulation::Formulation,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Program> {
    let previous = std::mem::replace(&mut program.objective, objective::ModelObjective::Affine);
    let squares = match previous {
        objective::ModelObjective::LeastSquares(value)
        | objective::ModelObjective::ConstantSquares(value) => value,
        _ => {
            return Err(
                policy.invalid("quadratic squared objective lost its original residual role")
            )
        }
    };
    let mut roots =
        runmat_value::exact_capacity_vec(squares.terms.len()).map_err(|e| policy.storage(e))?;
    for term in &squares.terms {
        roots.push(term.residual);
        c.completed(1).await?;
    }
    let selected = selection::Selection::prepare(&program.source, roots, policy, c).await?;
    let compiled = match compile(
        &program.source,
        &program.layout,
        selected,
        Some(squares),
        policy,
        c,
    )
    .await?
    {
        Ok(value) => value,
        Err(stop) => return Ok(Err(stop)),
    };
    let sense = formulation
        .objective
        .as_ref()
        .expect("original squared objective")
        .sense;
    let Prepared {
        cost,
        evidence,
        facts,
    } = match compiled
        .assemble(program.layout.columns, sense, policy, c)
        .await?
    {
        Ok(value) => value,
        Err(stop) => return Ok(Err(stop)),
    };
    let mut constraint_roots = std::collections::HashSet::new();
    for constraint in &formulation.constraints {
        constraint_roots
            .try_reserve(2)
            .map_err(|e| policy.reserve(e))?;
        constraint_roots.insert(constraint.left);
        constraint_roots.insert(constraint.right);
        c.completed(1).await?;
    }
    if let Projection::Squares(squares) = &evidence.projection {
        for term in &squares.terms {
            if !constraint_roots.contains(&term.residual) {
                program.arrays.remove(&term.residual);
            }
            c.completed(1).await?;
        }
    }
    program
        .facts
        .try_reserve(facts.len())
        .map_err(|e| policy.reserve(e))?;
    for (handle, fact) in facts {
        program.facts.entry(handle).or_insert(fact);
        c.completed(1).await?;
    }
    program.objective = objective::ModelObjective::Quadratic(
        crate::builtins::math::optim::common::retained_control::Retained::new(
            Model {
                cost: Some(cost),
                evidence,
            },
            c,
            crate::builtins::math::optim::common::retained_control::Policy {
                identifier: "RunMat:OptimModel:Allocation",
                operation: "quadratic model control allocation",
            },
        )?,
    );
    Ok(Ok(program))
}
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn force(
    mut program: Program,
    formulation: &formulation::Formulation,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Program> {
    match &program.objective {
        objective::ModelObjective::Quadratic(_) => return Ok(Ok(program)),
        objective::ModelObjective::LeastSquares(_)
        | objective::ModelObjective::ConstantSquares(_) => {
            return from_squares(program, formulation, policy, c).await;
        }
        objective::ModelObjective::Affine => {}
        objective::ModelObjective::Equations(_) => {
            return Err(policy.invalid(
                "an equation residual program cannot be used as a scalar quadratic objective",
            ));
        }
    }
    let mut roots = Vec::new();
    if let Some(objective) = &formulation.objective {
        roots.try_reserve(1).map_err(|e| policy.reserve(e))?;
        roots.push(objective.root);
    }
    let selection = selection::Selection::prepare(&program.source, roots, policy, c).await?;
    let sense = formulation.objective.as_ref().map_or(
        runmat_value::optimization_model::OptimizationObjectiveSense::Minimize,
        |objective| objective.sense,
    );
    let scale = affine_scale(&program, &selection, sense, policy, c).await?;
    program.objective = objective::ModelObjective::Quadratic(
        crate::builtins::math::optim::common::retained_control::Retained::new(
            Model {
                cost: None,
                evidence: Evidence {
                    restricted: None,
                    atoms: super::super::norm_square::Plan::default(),
                    selection,
                    curvature: Curvature::Convex,
                    projection: Projection::Affine,
                    scale,
                },
            },
            c,
            crate::builtins::math::optim::common::retained_control::Policy {
                identifier: "RunMat:OptimModel:Allocation",
                operation: "quadratic model control allocation",
            },
        )?,
    );
    Ok(Ok(program))
}
