//! One selected original numerical body; other solver futures do not exist here.
use super::super::*;
pub(super) async fn run<'a>(
    prepared: Algebraic,
    invocation: Invocation<'a>,
) -> BuiltinResult<Completed<'a>> {
    let Algebraic {
        program,
        formulation,
        rows,
        assembled,
        quadratic_cost,
        original_convexity,
        ..
    } = prepared;
    let Invocation {
        options,
        products,
        policy,
        cooperation,
        starting,
        ..
    } = invocation;
    let mut validation =
        validation::Validator::new(&program, &formulation, &rows, policy, cooperation).await?;
    let mut outcome = {
        let timing = if products != Products::Primal {
            Some(SolverTiming::start(cooperation.runtime())?)
        } else {
            None
        };
        let quadratic::Cost { hessian, linear } = match quadratic_cost {
            Some(cost) => {
                if !matches!(assembled.cost, assembly::Cost::Quadratic) {
                    return Err(
                        policy.invalid("quadratic objective has inconsistent assembly role")
                    );
                }
                cost
            }
            None => {
                let assembly::Cost::Affine(linear) = assembled.cost else {
                    return Err(policy.invalid("affine quadprog cost is unavailable"));
                };
                let hessian = assembly::matrix(
                    Vec::new(),
                    program.layout.columns,
                    program.layout.columns,
                    policy,
                    cooperation,
                )
                .await?
                .map_err(|stop| stop.error(policy))?;
                quadratic::Cost { hessian, linear }
            }
        };
        let prepared = qp::prepare(
            qp::Problem {
                p: hessian,
                f: linear,
                a_ineq: assembled.a_ineq,
                b_ineq: assembled.b_ineq,
                a_eq: assembled.a_eq,
                b_eq: assembled.b_eq,
                lb: assembled.lb,
                ub: assembled.ub,
            },
            options,
            timing.as_ref(),
            cooperation,
        )
        .await?;
        let initial = if prepared.algorithm() == runmat_builtins::OptimAlgorithm::ActiveSet {
            match starting {
                Some(value) => {
                    layout::initial_point(
                        &program.source,
                        &program.layout,
                        value,
                        policy,
                        cooperation,
                    )
                    .await?
                }
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let mut outcome = qp::solve(
            prepared,
            initial,
            if products == Products::WithMultipliers {
                qp::Products::WithMultipliers
            } else {
                qp::Products::Primal
            },
            cooperation,
            &mut validation,
        )
        .await?;
        if original_convexity && outcome.flag == -6 {
            outcome.flag = -8;
            outcome.message="Rounded quadratic Hessian failed numerical convexity admission although the original objective is convex.";
            if matches!(outcome.method, qp::Method::ActiveSet(_)) {
                outcome.method = qp::Method::ActiveSet(qp::ActiveTermination::Unverified);
            }
        }
        report::Outcome::Quadratic { outcome, timing }
    };
    let (retained, stopped) = validation.into_stopped();
    if let report::Outcome::Quadratic {
        outcome: result, ..
    } = &outcome
    {
        if let qp::Method::ActiveSet(reason) = result.method {
            if reason == qp::ActiveTermination::Unverified {
                return Err(unverified(policy, result.message));
            }
            if !matches!(
                reason,
                qp::ActiveTermination::Optimal
                    | qp::ActiveTermination::Infeasible
                    | qp::ActiveTermination::Unbounded
                    | qp::ActiveTermination::Nonconvex
            ) {
                let candidate = stopped.ok_or_else(|| {
                    unverified(policy, "active QP stop has no original-model candidate")
                })?;
                let numerical = result.x.as_deref().ok_or_else(|| {
                    unverified(policy, "active QP stop has no numerical candidate")
                })?;
                if !candidate
                    .matches(&program.layout, numerical, policy, cooperation)
                    .await?
                {
                    return Err(unverified(
                        policy,
                        "active QP stop differs from its original-model candidate",
                    ));
                }
                return Ok(Completed {
                    family: Either::Right(Either::Right(AlgebraicStopped {
                        program,
                        formulation,
                        rows,
                        candidate,
                        outcome,
                    })),
                    policy,
                    cooperation,
                });
            }
        }
    }
    if let report::Outcome::Quadratic {
        outcome: result, ..
    } = &mut outcome
    {
        if matches!(
            result.method,
            qp::Method::ActiveSet(qp::ActiveTermination::Infeasible)
        ) {
            // The hard certificate does not make a supplied x0 feasible.
            result.x = None;
            result.fval = None;
            result.lambda = None;
        }
    }
    Ok(Completed {
        family: Either::Left(AlgebraicCompleted {
            program,
            formulation,
            rows,
            retained,
            outcome,
        }),
        policy,
        cooperation,
    })
}
