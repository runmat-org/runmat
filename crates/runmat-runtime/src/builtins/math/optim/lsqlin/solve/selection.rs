//! Domain data select one actual numerical algorithm after common bound checks.
use super::*;
use futures::future::Either;
pub(super) struct Prepared<'a, V> {
    problem: Problem,
    products: compiled::Products,
    units: objective::Units,
    cooperation: &'a mut ScalarCooperation,
    validation: &'a mut V,
    conflict: Option<(usize, f64, f64)>,
}
pub(super) async fn prepare<'a, V: compiled::Validation>(
    problem: Problem,
    products: compiled::Products,
    units: objective::Units,
    cooperation: &'a mut ScalarCooperation,
    validation: &'a mut V,
) -> BuiltinResult<Prepared<'a, V>> {
    let mut conflict = None;
    for (column, (&lower, &upper)) in problem.lb.iter().zip(&problem.ub).enumerate() {
        if lower > upper || lower == f64::INFINITY || upper == f64::NEG_INFINITY {
            conflict = Some((column, lower, upper));
            break;
        }
        cooperation.completed(1).await?;
    }
    Ok(Prepared {
        problem,
        products,
        units,
        cooperation,
        validation,
        conflict,
    })
}
pub(super) fn execute<'a, V: compiled::Validation + 'a>(
    prepared: Prepared<'a, V>,
) -> impl std::future::Future<Output = BuiltinResult<Outcome>> + 'a {
    if prepared.conflict.is_some() {
        Either::Left(bounds(prepared))
    } else {
        let Prepared {
            problem,
            products,
            units,
            cooperation,
            validation,
            conflict: _,
        } = prepared;
        Either::Right(
            if problem.controls.algorithm == runmat_builtins::OptimAlgorithm::TrustRegionReflective
            {
                Either::Left(super::super::reflective::solve(
                    problem,
                    products,
                    units,
                    cooperation,
                    validation,
                ))
            } else if problem.controls.algorithm == runmat_builtins::OptimAlgorithm::ActiveSet {
                Either::Right(Either::Left(super::super::active_set::solve(
                    problem,
                    products,
                    units,
                    cooperation,
                    validation,
                )))
            } else {
                Either::Right(Either::Right(super::interior::solve(
                    problem,
                    products,
                    units,
                    cooperation,
                    validation,
                )))
            },
        )
    }
}
async fn bounds<V: compiled::Validation>(prepared: Prepared<'_, V>) -> BuiltinResult<Outcome> {
    let Prepared {
        problem,
        cooperation,
        validation,
        conflict,
        ..
    } = prepared;
    let (column, lower, upper) = conflict.expect("selected original bound conflict");
    let valid = validation
        .certificate(
            compiled::Certificate::InconsistentBounds {
                column,
                lower,
                upper,
            },
            cooperation,
        )
        .await?;
    let mut out = if valid {
        let mut out = Outcome::failure(
            -2,
            0,
            "Inconsistent bounds; no finite feasible point exists.",
        );
        if !problem.x0.is_empty() {
            out.x = Some(problem.x0);
        }
        out
    } else {
        Outcome::failure(
            -8,
            0,
            "Assembled bound conflict failed original model validation.",
        )
    };
    if problem.controls.algorithm == runmat_builtins::OptimAlgorithm::TrustRegionReflective {
        out.method = model::Method::Reflective { cg_iterations: 0 };
    }
    if problem.controls.algorithm == runmat_builtins::OptimAlgorithm::ActiveSet {
        out.method = model::Method::ActiveSet {
            termination: if valid {
                model::ActiveTermination::Infeasible
            } else {
                model::ActiveTermination::Unverified
            },
        };
    }
    problem.controls.summary(&out, cooperation.runtime())?;
    Ok(out)
}
