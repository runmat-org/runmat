//! Continuous extensions use the accepted RK step's original stage derivatives.
//! No derivative callback or integration-controller decision is added here.
//!
//! DP45 quartic correction weights: Shampine (1986), also published in Hairer
//! and Wanner's DOPRI5 reference: https://www.unige.ch/~hairer/prog/nonstiff/dopri5.f
//! The cubic endpoint/derivative Hermite basis is also the BS23 extension.

use crate::context::host_allocation::{host_allocation_runtime_error, HostAllocationError};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

use super::super::common::check_cancellation;

pub(super) enum StepInterpolant {
    DormandPrince {
        first: Vec<f64>,
        last: Vec<f64>,
        correction: Vec<f64>,
    },
    BogackiShampine {
        first: Vec<f64>,
        last: Vec<f64>,
    },
}

pub(super) struct RungeKuttaStep {
    pub y_next: Vec<f64>,
    pub error: Vec<f64>,
    pub interpolant: Option<StepInterpolant>,
}

pub(super) fn output_vector(
    name: &str,
    runtime: Option<&RuntimeContext>,
    count: usize,
) -> BuiltinResult<Vec<f64>> {
    check_cancellation(name, runtime)?;
    runmat_value::exact_capacity_vec(count).map_err(|source| {
        let mut error = host_allocation_runtime_error(
            runtime,
            "RunMat:Placement:ResourceLimit",
            format!("{name}: output storage allocation failed"),
            HostAllocationError::Numeric(source),
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}

/// The six nonzero DP45 dense weights multiply already-computed stages.
/// The correction is scaled by h and adds theta²(1-theta)² to the cubic.
pub(super) async fn dormand_prince_correction(
    name: &str,
    runtime: Option<&RuntimeContext>,
    h: f64,
    stages: [&[f64]; 6],
) -> BuiltinResult<Vec<f64>> {
    check_cancellation(name, runtime)?;
    const WEIGHTS: [f64; 6] = [
        -12715105075.0 / 11282082432.0,
        87487479700.0 / 32700410799.0,
        -10690763975.0 / 1880347072.0,
        701980252875.0 / 199316789632.0,
        -1453857185.0 / 822651844.0,
        69997945.0 / 29380423.0,
    ];
    let count = stages[0].len();
    let mut correction = output_vector(name, runtime, count)?;
    let mut cooperation = runtime.map(RuntimeContext::cpu_cooperation);
    for start in (0..count).step_by(1024) {
        check_cancellation(name, runtime)?;
        let end = start.saturating_add(1024).min(count);
        for component in start..end {
            let weighted = stages
                .iter()
                .zip(WEIGHTS)
                .map(|(stage, weight)| weight * stage[component])
                .sum::<f64>();
            correction.push(h * weighted);
        }
        if let Some(cooperation) = &mut cooperation {
            cooperation
                .checkpoint((end - start) as u64)
                .await
                .map_err(|source| source.into_runtime_error_in_context(runtime))?;
        }
    }
    Ok(correction)
}

impl StepInterpolant {
    pub(super) async fn evaluate(
        &self,
        name: &str,
        runtime: Option<&RuntimeContext>,
        y_start: &[f64],
        y_end: &[f64],
        h: f64,
        theta: f64,
    ) -> BuiltinResult<Vec<f64>> {
        check_cancellation(name, runtime)?;
        let (first, last, correction) = match self {
            Self::DormandPrince {
                first,
                last,
                correction,
            } => (first, last, Some(correction)),
            Self::BogackiShampine { first, last } => (first, last, None),
        };
        let mut result = output_vector(name, runtime, y_start.len())?;
        let mut cooperation = runtime.map(RuntimeContext::cpu_cooperation);
        let theta2 = theta * theta;
        let complement = 1.0 - theta;
        let bump = theta2 * complement * complement;
        for start in (0..y_start.len()).step_by(1024) {
            check_cancellation(name, runtime)?;
            let end = start.saturating_add(1024).min(y_start.len());
            for component in start..end {
                let delta = y_end[component] - y_start[component];
                let start_slope = h * first[component];
                let end_slope = h * last[component];
                let value = y_start[component]
                    + theta * start_slope
                    + theta2 * (3.0 * delta - 2.0 * start_slope - end_slope)
                    + theta2 * theta * (-2.0 * delta + start_slope + end_slope)
                    + correction.map_or(0.0, |values| bump * values[component]);
                result.push(value);
            }
            if let Some(cooperation) = &mut cooperation {
                cooperation
                    .checkpoint((end - start) as u64)
                    .await
                    .map_err(|source| source.into_runtime_error_in_context(runtime))?;
            }
        }
        Ok(result)
    }
}
