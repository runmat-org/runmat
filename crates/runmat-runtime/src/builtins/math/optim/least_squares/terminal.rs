//! Terminal numerical products keep their accepted derivative source until publication.
use super::*;
pub(super) fn result<'a, E: LeastSquaresEvaluator + 'a>(
    evaluator: &'a mut E,
    bounds: &'a LeastSquaresBounds,
    options: &'a LeastSquaresOptions,
    x: Vec<f64>,
    residual: Vec<f64>,
    jacobian: Vec<f64>,
    exitflag: i32,
    iterations: usize,
    func_count: usize,
    first_order_optimality: f64,
    step_size: f64,
    message: &'a str,
) -> impl Future<Output = BuiltinResult<LeastSquaresResult>> + 'a {
    finish(
        evaluator,
        bounds,
        options,
        x,
        residual,
        DerivativeResult::Workspace(jacobian),
        exitflag,
        iterations,
        func_count,
        first_order_optimality,
        step_size,
        message,
    )
}

pub(super) enum DerivativeResult {
    Workspace(Vec<f64>),
    Retained(Jacobian),
    RetainedWithWorkspace(Jacobian, Vec<f64>),
}
pub(super) async fn finish<E: LeastSquaresEvaluator>(
    evaluator: &mut E,
    bounds: &LeastSquaresBounds,
    options: &LeastSquaresOptions,
    x: Vec<f64>,
    residual: Vec<f64>,
    jacobian: DerivativeResult,
    exitflag: i32,
    iterations: usize,
    func_count: usize,
    first_order_optimality: f64,
    step_size: f64,
    message: &str,
) -> BuiltinResult<LeastSquaresResult> {
    let mut numerical = None;
    let derivative = match jacobian {
        DerivativeResult::Workspace(values) => match evaluator.take_jacobian() {
            Some(source) => {
                numerical = Some(values);
                DerivativeResult::Retained(source)
            }
            None => DerivativeResult::Workspace(values),
        },
        DerivativeResult::RetainedWithWorkspace(source, values) => {
            numerical = Some(values);
            DerivativeResult::Retained(source)
        }
        source => source,
    };
    let (jacobian, gradient, first_order_optimality, jacobian_source) = match derivative {
        DerivativeResult::Retained(source) => {
            if source.dimensions() != (residual.len(), x.len()) {
                return Err(optim_error(
                    "lsqnonlin",
                    "accepted Jacobian disagrees with returned point",
                ));
            }
            let c = evaluator.cooperation().ok_or_else(|| {
                optim_error("lsqnonlin", "supplied Jacobian lost original cooperation")
            })?;
            let gradient = if options.final_gradient {
                let mut gradient = runmat_value::exact_capacity_vec(x.len()).map_err(|source| {
                    crate::build_runtime_error("Jacobian gradient allocation failed")
                        .with_source(source)
                        .build()
                })?;
                while gradient.len() < x.len() {
                    let end = gradient.len().saturating_add(1024).min(x.len());
                    let count = end - gradient.len();
                    gradient.resize(end, 0.);
                    c.completed(count as u64).await?;
                }
                let finite = {
                    let mut products = source.products(c).await?;
                    products.apply(
                        crate::builtins::math::linalg::ops::linear_operator::ProductOrientation::Adjoint,
                        &residual, &mut gradient, c).await?.is_some()
                };
                if !finite {
                    for chunk in gradient.chunks_mut(1024) {
                        chunk.fill(f64::NAN);
                        c.completed(chunk.len() as u64).await?;
                    }
                }
                Some(gradient)
            } else {
                None
            };
            let optimality = match &gradient {
                Some(gradient) => criteria::from_gradient(gradient, &x, bounds, c).await?,
                None => first_order_optimality,
            };
            let values = if options.final_jacobian {
                source.into_published(numerical, c).await?
            } else {
                PublishedJacobian::RowMajor(Vec::new())
            };
            (values, gradient, optimality, JacobianSource::Supplied)
        }
        DerivativeResult::RetainedWithWorkspace(_, _) => {
            unreachable!("normalized retained derivative")
        }
        DerivativeResult::Workspace(jacobian) => {
            if let Some(current) = evaluator.take_analytic_jacobian() {
                if current.len() != checked_jacobian_work("optim", residual.len(), x.len())? {
                    return Err(optim_error(
                        "optim",
                        "accepted Jacobian dimensions do not match the returned iterate",
                    ));
                }
                let optimality =
                    traversal::first_order_optimality(&current, &residual, x.len(), evaluator)
                        .await?;
                (
                    PublishedJacobian::RowMajor(current),
                    None,
                    optimality,
                    JacobianSource::Supplied,
                )
            } else {
                (
                    PublishedJacobian::RowMajor(jacobian),
                    None,
                    first_order_optimality,
                    if evaluator.supplies_jacobian() {
                        JacobianSource::Supplied
                    } else {
                        JacobianSource::FiniteDifference
                    },
                )
            }
        }
    };

    let mut jacobian = jacobian;
    let unavailable = matches!(&jacobian, PublishedJacobian::RowMajor(values) if values.is_empty())
        && !x.is_empty()
        && !residual.is_empty();
    let first_order_optimality =
        if unavailable && gradient.is_none() && jacobian_source == JacobianSource::FiniteDifference
        {
            f64::NAN
        } else {
            first_order_optimality
        };
    if unavailable && options.final_jacobian {
        let count = checked_jacobian_work("optim", residual.len(), x.len())?;
        let PublishedJacobian::RowMajor(jacobian) = &mut jacobian else {
            unreachable!("missing dense Jacobian");
        };
        jacobian.try_reserve(count).map_err(|source| {
            crate::build_runtime_error("final Jacobian allocation failed")
                .with_source(source)
                .build()
        })?;
        while jacobian.len() < count {
            let end = jacobian.len().saturating_add(4096).min(count);
            let completed = end - jacobian.len();
            let first = jacobian.len();
            jacobian.resize(end, f64::NAN);
            for index in first..end {
                let column = index % x.len();
                if bounds.lower[column] == bounds.upper[column] {
                    jacobian[index] = 0.0;
                }
            }
            checkpoint(evaluator, completed).await?;
        }
    }
    let message = if unavailable && options.final_jacobian {
        format!("{message} Final Jacobian unavailable within the function-evaluation budget; unevaluated coefficients are NaN.")
    } else {
        message.to_string()
    };
    let resnorm = norm2_squared(&residual, evaluator).await?;
    Ok(LeastSquaresResult {
        point_evidence: PointEvidence::Evaluated,
        jacobian_source,
        gradient,
        cg_iterations: 0,
        method: Method::LevenbergMarquardt,
        variable_len: x.len(),
        residual_len: residual.len(),
        x,
        residual,
        jacobian,
        resnorm,
        exitflag,
        iterations,
        func_count,
        first_order_optimality,
        step_size,
        message,
    })
}
