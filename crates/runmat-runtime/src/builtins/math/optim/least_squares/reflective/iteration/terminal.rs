//! Final derivatives belong only to the accepted original point.
use super::*;
pub(super) struct State {
    pub point: initial::Seed,
    pub jacobian: Option<Jacobian>,
    pub iterations: usize,
    pub count: usize,
    pub cg_iterations: usize,
    pub optimality: f64,
    pub step: f64,
}
#[derive(Clone, Copy)]
pub(super) enum Stop {
    Optimality,
    Step,
    Function,
    Limit,
}
impl Stop {
    fn flag(self) -> i32 {
        match self {
            Self::Optimality => 1,
            Self::Step => 2,
            Self::Function => 3,
            Self::Limit => 0,
        }
    }
    fn message(self) -> &'static str {
        match self{
  Self::Optimality=>"Reflective first-order optimality is within tolerance.",
  Self::Step=>"Reflective step displacement is within StepTolerance; stationarity is not asserted.",
  Self::Function=>"Reflective squared-residual change is within FunctionTolerance; stationarity is not asserted.",
  Self::Limit=>"Reflective iteration or function-evaluation limit reached.",
 }
    }
}
pub(super) struct Terminal<'a> {
    pub name: &'a str,
    pub geometry: &'a Geometry,
    pub bounds: &'a LeastSquaresBounds,
    pub options: &'a LeastSquaresOptions,
    pub display: OptimDisplay,
}
impl Terminal<'_> {
    pub(super) async fn finish<E: LeastSquaresEvaluator>(
        &self,
        evaluator: &mut E,
        state: State,
        stop: Stop,
    ) -> BuiltinResult<LeastSquaresResult> {
        let Self {
            name,
            geometry,
            bounds,
            options,
            display,
        } = *self;
        let State {
            point,
            jacobian,
            iterations,
            count,
            cg_iterations,
            optimality,
            step,
        } = state;
        let source = jacobian.or_else(|| evaluator.take_jacobian());
        let mut result = match source {
            Some(source @ (Jacobian::Matrix { .. } | Jacobian::Opaque(_))) => {
                finish(
                    evaluator,
                    bounds,
                    options,
                    point.x,
                    point.residual,
                    DerivativeResult::Retained(source),
                    stop.flag(),
                    iterations,
                    count,
                    optimality,
                    step,
                    stop.message(),
                )
                .await?
            }
            source => {
                let values = match source {
                    Some(Jacobian::RowMajor { values, .. }) => values,
                    None => Vec::new(),
                    Some(Jacobian::Matrix { .. } | Jacobian::Opaque(_)) => {
                        unreachable!("retained source selected")
                    }
                };
                if matches!(stop, Stop::Step | Stop::Function) {
                    final_result(
                        name,
                        evaluator,
                        point.x,
                        point.residual,
                        values,
                        bounds,
                        options,
                        stop.flag(),
                        iterations,
                        count,
                        optimality,
                        step,
                        stop.message(),
                    )
                    .await?
                } else {
                    result(
                        evaluator,
                        bounds,
                        options,
                        point.x,
                        point.residual,
                        values,
                        stop.flag(),
                        iterations,
                        count,
                        optimality,
                        step,
                        stop.message(),
                    )
                    .await?
                }
            }
        };
        result.cg_iterations = cg_iterations;
        result.method = Method::Reflective;
        result.first_order_optimality = if geometry.axes.is_empty() {
            0.
        } else if let Some(gradient) = &result.gradient {
            match geometry
                .frame_from_gradient(&result.x, gradient, cooperation(evaluator)?)
                .await?
            {
                Some(frame) => frame.optimality,
                None => f64::NAN,
            }
        } else if let Some(values) = result.jacobian.row_major() {
            if result.residual_len.checked_mul(result.variable_len) == Some(values.len()) {
                let mut products = derivative::Products::RowMajor {
                    values,
                    rows: result.residual_len,
                    columns: result.variable_len,
                };
                match geometry
                    .frame(
                        &result.x,
                        &result.residual,
                        &mut products,
                        cooperation(evaluator)?,
                    )
                    .await?
                {
                    Some(frame) => frame.optimality,
                    None => f64::NAN,
                }
            } else if values.is_empty() && result.jacobian_source == JacobianSource::Supplied {
                result.first_order_optimality
            } else {
                f64::NAN
            }
        } else {
            result.first_order_optimality
        };
        original_derivative_columns(name, &mut result, bounds, evaluator.cooperation()).await?;
        emit(evaluator, display, Event::Final, || {
            format!(
                "lsqnonlin: {} iterations={} funcCount={}",
                stop.message(),
                result.iterations,
                result.func_count
            )
        })?;
        Ok(result)
    }
}
