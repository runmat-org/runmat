use super::super::common::OdeInput;
use super::dense::{dormand_prince_correction, RungeKuttaStep, StepInterpolant};
use super::math::lincomb;
use super::rhs::eval_rhs;
use crate::BuiltinResult;
use runmat_value::Value;

pub(super) async fn step_ode45(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    h: f64,
    input: &OdeInput,
    dense_output: bool,
) -> BuiltinResult<RungeKuttaStep> {
    let k1 = eval_rhs(name, function, t, y, input).await?;
    let y2 = lincomb(y, h, &[(&k1, 1.0 / 5.0)]);
    let k2 = eval_rhs(name, function, t + h * (1.0 / 5.0), &y2, input).await?;

    let y3 = lincomb(y, h, &[(&k1, 3.0 / 40.0), (&k2, 9.0 / 40.0)]);
    let k3 = eval_rhs(name, function, t + h * (3.0 / 10.0), &y3, input).await?;

    let y4 = lincomb(
        y,
        h,
        &[(&k1, 44.0 / 45.0), (&k2, -56.0 / 15.0), (&k3, 32.0 / 9.0)],
    );
    let k4 = eval_rhs(name, function, t + h * (4.0 / 5.0), &y4, input).await?;

    let y5 = lincomb(
        y,
        h,
        &[
            (&k1, 19372.0 / 6561.0),
            (&k2, -25360.0 / 2187.0),
            (&k3, 64448.0 / 6561.0),
            (&k4, -212.0 / 729.0),
        ],
    );
    let k5 = eval_rhs(name, function, t + h * (8.0 / 9.0), &y5, input).await?;

    let y6 = lincomb(
        y,
        h,
        &[
            (&k1, 9017.0 / 3168.0),
            (&k2, -355.0 / 33.0),
            (&k3, 46732.0 / 5247.0),
            (&k4, 49.0 / 176.0),
            (&k5, -5103.0 / 18656.0),
        ],
    );
    let k6 = eval_rhs(name, function, t + h, &y6, input).await?;

    let y_high = lincomb(
        y,
        h,
        &[
            (&k1, 35.0 / 384.0),
            (&k3, 500.0 / 1113.0),
            (&k4, 125.0 / 192.0),
            (&k5, -2187.0 / 6784.0),
            (&k6, 11.0 / 84.0),
        ],
    );
    let k7 = eval_rhs(name, function, t + h, &y_high, input).await?;

    let y_low = lincomb(
        y,
        h,
        &[
            (&k1, 5179.0 / 57600.0),
            (&k3, 7571.0 / 16695.0),
            (&k4, 393.0 / 640.0),
            (&k5, -92097.0 / 339200.0),
            (&k6, 187.0 / 2100.0),
            (&k7, 1.0 / 40.0),
        ],
    );
    let err = y_high
        .iter()
        .zip(y_low.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    let interpolant = if dense_output {
        let correction = dormand_prince_correction(
            name,
            input.runtime.as_ref(),
            h,
            [&k1, &k3, &k4, &k5, &k6, &k7],
        )
        .await?;
        Some(StepInterpolant::DormandPrince {
            first: k1,
            last: k7,
            correction,
        })
    } else {
        None
    };
    Ok(RungeKuttaStep {
        y_next: y_high,
        error: err,
        interpolant,
    })
}

pub(super) async fn step_ode23(
    name: &str,
    function: &Value,
    t: f64,
    y: &[f64],
    h: f64,
    input: &OdeInput,
    dense_output: bool,
) -> BuiltinResult<RungeKuttaStep> {
    let k1 = eval_rhs(name, function, t, y, input).await?;
    let y2 = lincomb(y, h, &[(&k1, 0.5)]);
    let k2 = eval_rhs(name, function, t + h * 0.5, &y2, input).await?;

    let y3 = lincomb(y, h, &[(&k2, 3.0 / 4.0)]);
    let k3 = eval_rhs(name, function, t + h * (3.0 / 4.0), &y3, input).await?;

    let y_high = lincomb(
        y,
        h,
        &[(&k1, 2.0 / 9.0), (&k2, 1.0 / 3.0), (&k3, 4.0 / 9.0)],
    );
    let k4 = eval_rhs(name, function, t + h, &y_high, input).await?;

    let y_low = lincomb(
        y,
        h,
        &[
            (&k1, 7.0 / 24.0),
            (&k2, 1.0 / 4.0),
            (&k3, 1.0 / 3.0),
            (&k4, 1.0 / 8.0),
        ],
    );
    let err = y_high
        .iter()
        .zip(y_low.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    let interpolant = dense_output.then_some(StepInterpolant::BogackiShampine {
        first: k1,
        last: k4,
    });
    Ok(RungeKuttaStep {
        y_next: y_high,
        error: err,
        interpolant,
    })
}
