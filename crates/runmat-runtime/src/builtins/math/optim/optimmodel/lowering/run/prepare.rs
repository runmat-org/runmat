//! Original model compilation and assembly before a numerical child exists.
use super::*;
pub(super) async fn run<'a>(
    model: &runmat_value::optimization_model::OptimizationProblemData,
    options: Option<&'a Value>,
    products: Products,
    selected: Option<OptimSolver>,
    starting: Option<&'a Value>,
    policy: Policy,
    cooperation: &'a mut ScalarCooperation,
) -> BuiltinResult<Prepared<'a>> {
    let formulation = formulation::Formulation::prepare(model, policy, cooperation).await?;
    let compiled = objective::compile(&formulation, selected, policy, cooperation)
        .await?
        .map_err(|stop| stop.error(policy))?;
    let mut program = match compiled {
        super::super::Prepared::Algebraic(program) => program,
        super::super::Prepared::Nonlinear(model) => {
            let solver =
                nonlinear::select(&model, &formulation, selected, policy, cooperation).await?;
            return Ok(Prepared {
                family: Either::Right(NonlinearPrepared {
                    model,
                    formulation,
                    solver,
                }),
                invocation: Invocation {
                    options,
                    starting,
                    products,
                    policy,
                    cooperation,
                },
            });
        }
    };
    if selected == Some(OptimSolver::Quadprog) {
        program = quadratic::force(program, &formulation, policy, cooperation)
            .await?
            .map_err(|stop| stop.error(policy))?;
    }
    if program.layout.columns == 0 {
        return Err(policy.invalid("solve requires at least one original optimization variable"));
    }
    let mut integer = false;
    for variable in &program.layout.variables {
        integer |= variable.domain
            != runmat_value::optimization_variable::OptimizationVariableDomain::Continuous;
        cooperation.completed(1).await?;
    }
    let affine = matches!(
        &program.objective,
        objective::ModelObjective::Affine | objective::ModelObjective::ConstantSquares(_)
    );
    if integer && !affine {
        return Err(policy.invalid("integer model variables require the supported affine objective and constraint consumer"));
    }
    let solver = if integer || selected == Some(OptimSolver::Intlinprog) {
        if !affine {
            return Err(policy.invalid("intlinprog requires an affine model objective"));
        }
        OptimSolver::Intlinprog
    } else if program.objective.quadratic().is_some() {
        OptimSolver::Quadprog
    } else if matches!(
        &program.objective,
        objective::ModelObjective::LeastSquares(_)
    ) {
        OptimSolver::Lsqlin
    } else {
        OptimSolver::Linprog
    };
    if selected.is_some_and(|selected| selected != solver) {
        return Err(
            policy.invalid("selected solver does not match the original model objective form")
        );
    }
    if starting.is_some() && solver == OptimSolver::Linprog {
        return Err(policy.invalid("a starting point is not consumed by this linear model solver"));
    }
    let quadratic_cost = program.objective.take_quadratic_cost();
    let original_convexity = program
        .objective
        .quadratic()
        .is_none_or(|evidence| evidence.curvature == quadratic::Curvature::Convex);
    let kind = if solver == OptimSolver::Quadprog {
        if quadratic_cost.is_some() {
            assembly::ObjectiveKind::Quadratic
        } else {
            assembly::ObjectiveKind::Affine
        }
    } else if solver == OptimSolver::Lsqlin {
        assembly::ObjectiveKind::LeastSquares
    } else {
        match &program.objective {
            objective::ModelObjective::ConstantSquares(_) => assembly::ObjectiveKind::Constant,
            _ => assembly::ObjectiveKind::Affine,
        }
    };
    let assembled = if solver == OptimSolver::Intlinprog {
        assembly::assemble_domains(
            &program,
            &formulation,
            kind,
            assembly::Domains::Integer,
            policy,
            cooperation,
        )
        .await?
    } else {
        assembly::assemble(&program, &formulation, kind, policy, cooperation).await?
    }
    .map_err(|stop| stop.error(policy))?;
    let rows = assembled.rows;
    let assembled = assembly::Assembled {
        rows: Vec::new(),
        ..assembled
    };

    Ok(Prepared {
        family: Either::Left(Algebraic {
            program,
            formulation,
            rows,
            assembled,
            solver,
            quadratic_cost,
            original_convexity,
        }),
        invocation: Invocation {
            options,
            starting,
            products,
            policy,
            cooperation,
        },
    })
}
