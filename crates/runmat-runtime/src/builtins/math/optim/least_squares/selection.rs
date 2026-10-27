//! Actual controller selection consumes one accepted original residual seed.
use super::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Method {
    LevenbergMarquardt,
    Reflective,
    NotStarted(runmat_builtins::OptimAlgorithm),
}
impl Method {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::LevenbergMarquardt => "levenberg-marquardt",
            Self::Reflective => "trust-region-reflective",
            Self::NotStarted(_) => "not-started",
        }
    }
}

pub(crate) fn solve_selected<'a, E: LeastSquaresEvaluator + 'a>(
    name: &'a str,
    evaluator: &'a mut E,
    x: Vec<f64>,
    bounds: &'a LeastSquaresBounds,
    options: &'a LeastSquaresOptions,
    requested: Method,
    display: runmat_builtins::OptimDisplay,
) -> impl Future<Output = BuiltinResult<LeastSquaresResult>> + 'a {
    solve_controlled(
        name, evaluator, x, bounds, options, requested, display, None,
    )
}

pub(crate) fn solve_controlled<'a, E: LeastSquaresEvaluator + 'a>(
    name: &'a str,
    evaluator: &'a mut E,
    x: Vec<f64>,
    bounds: &'a LeastSquaresBounds,
    options: &'a LeastSquaresOptions,
    requested: Method,
    display: runmat_builtins::OptimDisplay,
    source: Option<&'a runmat_value::StructValue>,
) -> impl Future<Output = BuiltinResult<LeastSquaresResult>> + 'a {
    let placement = if requested == Method::Reflective {
        initial::Placement::Interior
    } else {
        initial::Placement::Project
    };
    crate::execution::after_result(
        initial::invocation(name, evaluator, x, bounds, options, placement),
        move |prepared| {
            let initial::Prepared {
                name,
                evaluator,
                seed,
                bounds,
                options,
            } = prepared;
            let seed = match seed {
                initial::Initial::Supplied(x) => {
                    return futures::future::Either::Left(std::future::ready(Ok(not_started(
                        x, requested,
                    ))))
                }
                initial::Initial::Evaluated(seed) => seed,
            };
            futures::future::Either::Right(
                if requested == Method::Reflective && seed.residual.len() >= seed.x.len() {
                    futures::future::Either::Left(reflective::solve(
                        name, evaluator, seed, bounds, options, display, source,
                    ))
                } else {
                    let warning = if requested == Method::Reflective {
                        let runtime = evaluator.cooperation().and_then(|c| c.runtime());
                        crate::context::with_original_scope(runtime, || {
                            crate::warnings::emit(crate::warnings::WarningRequest{
     builtin:runmat_builtins::LSQNONLIN_CATALOG_ENTRY.identity.name,
     identifier:"RunMat:lsqnonlin:UnderdeterminedReflective",
     message:"The residual has fewer elements than variables; using Levenberg-Marquardt with the already evaluated starting point.",
     show_identifier:true,
    })
                        })
                    } else {
                        Ok(())
                    };
                    futures::future::Either::Right(match warning {
                        Err(error) => futures::future::Either::Left(std::future::ready(Err(error))),
                        Ok(()) => {
                            let runtime = if matches!(
                                display,
                                runmat_builtins::OptimDisplay::Iter
                                    | runmat_builtins::OptimDisplay::Final
                            ) {
                                evaluator.cooperation().and_then(|c| c.runtime()).cloned()
                            } else {
                                None
                            };
                            futures::future::Either::Right(crate::execution::after_result(
                                lm::solve(name, evaluator, seed, bounds, options, display),
                                move |result| {
                                    std::future::ready(super::display::finish(
                                        runtime, display, result,
                                    ))
                                },
                            ))
                        }
                    })
                },
            )
        },
    )
}

fn not_started(x: Vec<f64>, requested: Method) -> LeastSquaresResult {
    let requested = match requested {
        Method::Reflective => runmat_builtins::OptimAlgorithm::TrustRegionReflective,
        Method::LevenbergMarquardt => runmat_builtins::OptimAlgorithm::LevenbergMarquardt,
        Method::NotStarted(requested) => requested,
    };
    LeastSquaresResult {
        point_evidence: PointEvidence::Supplied,
        jacobian_source: JacobianSource::FiniteDifference,
        method: Method::NotStarted(requested),
        variable_len: x.len(), residual_len: 0, x,
        residual: Vec::new(), jacobian: PublishedJacobian::RowMajor(Vec::new()), gradient: None,
        resnorm: f64::NAN, first_order_optimality: f64::NAN,
        exitflag: 0, iterations: 0, func_count: 0, cg_iterations: 0, step_size: 0.,
        message: "Function-evaluation budget is zero. The projected supplied point is unevaluated; no residual, Jacobian, feasibility or stationarity is asserted.".to_owned(),
    }
}
