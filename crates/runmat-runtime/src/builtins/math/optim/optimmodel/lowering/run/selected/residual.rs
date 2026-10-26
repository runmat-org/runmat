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
        ..
    } = prepared;
    let Invocation {
        options,
        products,
        policy,
        starting,
        cooperation,
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
        let problem = objective::assemble(
            &program,
            assembly::Assembled {
                rows: Vec::new(),
                ..assembled
            },
            policy,
            cooperation,
        )
        .await?
        .map_err(|stop| stop.error(policy))?;
        let prepared = lsq::prepare(problem, options, cooperation).await?;
        let initial = if matches!(
            prepared.algorithm(),
            runmat_builtins::OptimAlgorithm::TrustRegionReflective
                | runmat_builtins::OptimAlgorithm::ActiveSet
        ) {
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
            // Inactive x0 remains rooted by execute and is not inspected.
            Vec::new()
        };
        report::Outcome::LeastSquares {
            outcome: lsq::solve(
                prepared,
                initial,
                if products == Products::WithMultipliers {
                    lsq::Products::WithMultipliers
                } else {
                    lsq::Products::Primal
                },
                lsq::Units::OriginalSquared,
                cooperation,
                &mut validation,
            )
            .await?,
            timing,
        }
    };
    let (retained, stopped) = validation.into_stopped();
    let stopped_reason = match &outcome {
        report::Outcome::LeastSquares { outcome, .. } => match outcome.method {
            lsq::Method::ActiveSet { termination } => Some(termination),
            _ => None,
        },
        _ => None,
    };
    if let Some(reason) = stopped_reason {
        if reason == lsq::ActiveTermination::Unverified {
            return Err(unverified(
                policy,
                "active-set candidate lacks original-model evidence",
            ));
        }
        if !matches!(
            reason,
            lsq::ActiveTermination::Optimal | lsq::ActiveTermination::Infeasible
        ) {
            let candidate = stopped.ok_or_else(|| {
                unverified(policy, "active-set stop has no original-model candidate")
            })?;
            let numerical = match &outcome {
                report::Outcome::LeastSquares { outcome, .. } => outcome.x.as_deref(),
                _ => None,
            }
            .ok_or_else(|| unverified(policy, "active-set stop has no numerical candidate"))?;
            if !candidate
                .matches(&program.layout, numerical, policy, cooperation)
                .await?
            {
                return Err(unverified(
                    policy,
                    "active-set stop differs from original-model candidate",
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
        if reason == lsq::ActiveTermination::Infeasible {
            // The exact certificate does not confer feasibility on a raw x0.
            // Problem-based infeasibility publishes the ordinary empty solution.
            if let report::Outcome::LeastSquares { outcome, .. } = &mut outcome {
                outcome.x = None;
                outcome.residual = None;
                outcome.resnorm = None;
            }
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
