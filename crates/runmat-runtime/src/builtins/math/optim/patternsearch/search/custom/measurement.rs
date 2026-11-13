//! The issued merit callable uses only the original point/constraint budget.
use super::*;
use crate::builtins::math::optim::common::constrained_numeric as numeric;
use crate::execution::{scoped_callable::Request, RootedValueSequence};
use numeric::{Geometry, Numeric};

pub(in crate::builtins::math::optim::patternsearch::search) async fn measure<S>(
    request: Request,
    point_axis: super::super::super::evaluation::PointAxis,
    base: &Point,
    initial: f64,
    reuse: super::super::merit::Reuse,
    geometry: &Linear,
    tolerance: f64,
    stop: impl Fn() -> Option<S>,
    budget: S,
    best: &mut Option<Best>,
    merit: &mut impl for<'a> std::ops::AsyncFnMut(
        &'a Point,
        &'a mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>>,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<ControlFlow<S, RootedValueSequence>> {
    if request.requested_outputs > 1 || request.arguments.len() != 1 {
        return Err(invalid(
            "custom search merit expects one point argument and at most one output",
        ));
    }
    evaluator.cooperation.completed(0).await?;
    if let Some(stop) = stop() {
        return Ok(ControlFlow::Break(stop));
    }
    let view = request.arguments.view();
    let value = view.get(0).expect("one actual merit input");
    let input_geometry = if evaluator.execution().vectorized {
        Geometry::MatrixAny
    } else {
        Geometry::Vector
    };
    if !numeric::preflight(
        value,
        "custom search point",
        input_geometry,
        &evaluator.context.policy,
        evaluator.cooperation.runtime(),
    )? {
        return Err(invalid("custom search point must be real numeric"));
    }
    // The original argument root owner remains live across provider awaits.
    let Numeric::Real { values, shape } = numeric::read(
        value.clone(),
        "custom search point",
        input_geometry,
        &evaluator.context.policy,
        evaluator.cooperation,
    )
    .await?
    else {
        return Err(invalid("custom search point must be finite and real"));
    };
    let n = base.x.len();
    let points = if evaluator.execution().vectorized {
        point_axis
            .point_count(&shape, n)
            .ok_or_else(|| invalid("custom search point dimensions or orientation changed"))?
    } else if values.len() == n {
        1
    } else {
        return Err(invalid("custom search point dimensions changed"));
    };
    if points.checked_mul(n) != Some(values.len()) {
        return Err(invalid("custom search point extent overflow"));
    }
    let admission = super::super::merit::Admission {
        base,
        initial,
        geometry,
        tolerance,
        reuse,
    };
    let mut transaction = super::super::merit::Transaction::new(points)?;
    for point in 0..points {
        let mut x = super::super::super::buffer(n)?;
        for axis in 0..n {
            x.push(if !evaluator.execution().vectorized {
                values[axis]
            } else {
                values[point_axis.offset(point, axis, points, n)]
            });
            evaluator.cooperation.completed(1).await?;
        }
        transaction.push(x, &admission, evaluator).await?;
    }
    let scores = match transaction
        .finish(best, initial, merit, evaluator, stop, budget)
        .await?
    {
        ControlFlow::Continue(scores) => scores,
        ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
    };
    if request.requested_outputs == 0 {
        return Ok(ControlFlow::Continue(RootedValueSequence::empty()));
    }
    let result = if evaluator.execution().vectorized {
        common::vector_to_value(
            "patternsearch",
            scores,
            &point_axis.result_shape(points),
            false,
        )?
    } else {
        Value::Num(scores[0])
    };
    let sequence = runmat_value::ValueSequence::single(result)
        .map_err(crate::sequence::sequence_error_to_runtime)?;
    let returned = match evaluator.cooperation.runtime() {
        Some(runtime) => RootedValueSequence::in_context(sequence, runtime)?,
        None => RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error())?,
    };
    Ok(ControlFlow::Continue(returned))
}
