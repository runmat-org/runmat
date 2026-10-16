use super::admission::Method;
use super::input::HostInput;
use super::*;
use crate::builtins::stats::summary::distribution_math::student_t_inv_upper_tail;
use nalgebra::DMatrix;
use runmat_value::{ExactHostTensor2dAllocation, NumericDType, NumericStorage, Tensor};

pub(super) struct Cooperation<'a> {
    cancellation: Option<runmat_execution::cancellation::CancellationToken>,
    budget: Option<crate::execution::CpuCooperationBudget>,
    runtime: Option<&'a crate::context::RuntimeContext>,
    receipt: Option<&'a crate::context::host_allocation::HostAllocationReceipt>,
}

impl<'a> Cooperation<'a> {
    pub(super) fn new(
        runtime: Option<&'a crate::context::RuntimeContext>,
        receipt: Option<&'a crate::context::host_allocation::HostAllocationReceipt>,
    ) -> Self {
        let cancellation = runtime.map(crate::context::RuntimeContext::cancellation);
        let budget = runtime.zip(cancellation.as_ref()).map(|(runtime, token)| {
            crate::execution::CpuCooperationBudget::new(
                runtime.execution().clone(),
                (*token).clone(),
            )
        });
        Self {
            cancellation,
            budget,
            runtime,
            receipt,
        }
    }

    pub(super) fn check(&self) -> BuiltinResult<()> {
        if let (Some(runtime), Some(receipt)) = (self.runtime, self.receipt) {
            receipt.checkpoint(runtime).map_err(allocation_error)?;
        }
        if self
            .cancellation
            .as_ref()
            .is_some_and(runmat_execution::cancellation::CancellationToken::is_cancelled)
        {
            return Err(crate::builtins::common::map_control_flow_with_builtin(
                crate::execution::ExecutionServiceError::Cancelled.into_runtime_error(),
                NAME,
            ));
        }
        Ok(())
    }

    async fn completed(&mut self, units: u64) -> BuiltinResult<()> {
        self.check()?;
        if let Some(budget) = self.budget.as_mut() {
            let result = budget.checkpoint(units).await;
            self.check()?;
            result.map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
        }
        Ok(())
    }

    fn retained_output(
        &self,
        dtype: NumericDType,
        estimates: &[f64],
        standard_errors: &[f64],
        critical: f64,
    ) -> BuiltinResult<Option<Value>> {
        let (Some(runtime), Some(receipt)) = (self.runtime, self.receipt) else {
            return Ok(None);
        };
        self.check()?;
        let p = estimates.len();
        let tensor = receipt
            .with_output_allocation(runtime, |permit| {
                let mut output = ExactHostTensor2dAllocation::new(dtype, p, 2, permit)?;
                match dtype {
                    NumericDType::F32 => {
                        let values = output.as_f32_slice_mut().expect("admitted single output");
                        fill_intervals(values, estimates, standard_errors, critical, |v| v as f32);
                    }
                    NumericDType::F64 => {
                        let values = output.as_f64_slice_mut().expect("admitted double output");
                        fill_intervals(values, estimates, standard_errors, critical, |v| v);
                    }
                    _ => unreachable!("admitted beta is single or double"),
                }
                output.into_tensor()
            })
            .map_err(allocation_error)?;
        self.check()?;
        Ok(Some(Value::Tensor(tensor)))
    }
}

fn fill_intervals<T: Copy>(
    output: &mut [T],
    estimates: &[f64],
    standard_errors: &[f64],
    critical: f64,
    cast: impl Fn(f64) -> T,
) {
    for (slot, value) in
        output
            .iter_mut()
            .zip(interval_endpoints(estimates, standard_errors, critical))
    {
        *slot = cast(value);
    }
}

fn interval_endpoints<'a>(
    estimates: &'a [f64],
    standard_errors: &'a [f64],
    critical: f64,
) -> impl Iterator<Item = f64> + 'a {
    let intervals = move || {
        estimates
            .iter()
            .zip(standard_errors)
            .map(move |(&estimate, &se)| (estimate, critical * se))
    };
    intervals()
        .map(|(estimate, half_width)| estimate - half_width)
        .chain(intervals().map(|(estimate, half_width)| estimate + half_width))
}

fn interval_values<T>(
    estimates: &[f64],
    standard_errors: &[f64],
    critical: f64,
    cast: impl Fn(f64) -> T,
) -> BuiltinResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(
            estimates
                .len()
                .checked_mul(2)
                .ok_or_else(|| invalid("output dimensions overflow"))?,
        )
        .map_err(|err| numerical(format!("cannot allocate confidence intervals: {err}")))?;
    values.extend(interval_endpoints(estimates, standard_errors, critical).map(cast));
    Ok(values)
}

pub(super) async fn intervals(
    cooperation: &mut Cooperation<'_>,
    method: Method,
    beta: &HostInput,
    residual: &HostInput,
    source: &HostInput,
    alpha: f64,
    produces_output: bool,
) -> BuiltinResult<Option<Value>> {
    let p = beta.len();
    let n = residual.len();
    let beta_values = beta.values_f64_cow();
    let residual_values = residual.values_f64_cow();
    if beta_values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("beta must contain finite values"));
    }
    let source_values = source.values_f64_cow();
    cooperation.check()?;
    let standard_errors = match method {
        Method::Covariance => {
            covariance_variances(&source_values, &residual_values, p, cooperation).await?
        }
        Method::Jacobian => {
            jacobian_variances(&source_values, &residual_values, n, p, cooperation).await?
        }
    };
    let df = standard_errors.0;
    let critical = student_t_inv_upper_tail(alpha / 2.0, df as f64);
    if !critical.is_finite() || critical < 0.0 {
        return Err(numerical(
            "unable to compute a finite Student-t critical value",
        ));
    }
    cooperation.completed(p as u64).await?;
    if !produces_output {
        return Ok(None);
    }
    if let Some(output) = cooperation.retained_output(
        beta.numeric_dtype(),
        &beta_values,
        &standard_errors.1,
        critical,
    )? {
        return Ok(Some(output));
    }
    let storage = match beta.numeric_dtype() {
        NumericDType::F32 => NumericStorage::F32(interval_values(
            &beta_values,
            &standard_errors.1,
            critical,
            |value| value as f32,
        )?),
        NumericDType::F64 => NumericStorage::F64(interval_values(
            &beta_values,
            &standard_errors.1,
            critical,
            |value| value,
        )?),
        _ => return Err(invalid("beta must be single or double")),
    };
    Tensor::from_numeric_storage(storage, vec![p, 2])
        .map(Value::Tensor)
        .map(Some)
        .map_err(|err| numerical(format!("cannot construct confidence intervals: {err}")))
}

async fn covariance_variances(
    covar: &[f64],
    residual: &[f64],
    p: usize,
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<(usize, Vec<f64>)> {
    let mut valid = 0usize;
    for &value in residual {
        if value.is_infinite() {
            return Err(invalid("residuals must not contain Inf"));
        }
        valid += usize::from(!value.is_nan());
        cooperation.completed(1).await?;
    }
    let df = valid.checked_sub(p).filter(|df| *df > 0).ok_or_else(|| {
        numerical("the number of complete residuals must exceed the number of coefficients")
    })?;
    let mut standard_errors = Vec::new();
    standard_errors
        .try_reserve_exact(p)
        .map_err(|err| numerical(format!("cannot allocate covariance diagonal: {err}")))?;
    for column in 0..p {
        for row in 0..p {
            if !covar[row + column * p].is_finite() {
                return Err(invalid("Covar must contain finite entries"));
            }
        }
        let variance = covar[column + column * p];
        if variance < 0.0 {
            return Err(invalid("Covar must have nonnegative diagonal variances"));
        }
        standard_errors.push(variance.sqrt());
        cooperation.completed(p as u64).await?;
    }
    Ok((df, standard_errors))
}

async fn jacobian_variances(
    jacobian: &[f64],
    residual: &[f64],
    n: usize,
    p: usize,
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<(usize, Vec<f64>)> {
    let mut complete = Vec::new();
    complete
        .try_reserve_exact(n)
        .map_err(|err| numerical(format!("cannot allocate observation mask: {err}")))?;
    let mut sumsq_scale = 0.0f64;
    let mut sumsq = 0.0f64;
    for (row, &value) in residual.iter().enumerate() {
        if value.is_infinite() {
            return Err(invalid("residuals must not contain Inf"));
        }
        let mut retained = !value.is_nan();
        for column in 0..p {
            let derivative = jacobian[row + column * n];
            if derivative.is_infinite() {
                return Err(invalid("Jacobian must not contain Inf"));
            }
            retained &= !derivative.is_nan();
        }
        if retained {
            let magnitude = value.abs();
            if magnitude > 0.0 {
                if sumsq_scale < magnitude {
                    sumsq = 1.0 + sumsq * (sumsq_scale / magnitude).powi(2);
                    sumsq_scale = magnitude;
                } else {
                    sumsq += (magnitude / sumsq_scale).powi(2);
                }
            }
            complete.push(row);
        }
        cooperation.completed(p as u64 + 1).await?;
    }
    let rows = complete.len();
    let df = rows.checked_sub(p).filter(|df| *df > 0).ok_or_else(|| {
        numerical("the number of complete observations must exceed the number of coefficients")
    })?;
    let scale = complete
        .iter()
        .flat_map(|&row| (0..p).map(move |column| jacobian[row + column * n].abs()))
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(numerical("Jacobian must have full column rank"));
    }
    let cells = rows
        .checked_mul(p)
        .ok_or_else(|| invalid("Jacobian dimensions overflow"))?;
    let mut data = Vec::new();
    data.try_reserve_exact(cells)
        .map_err(|err| numerical(format!("cannot allocate complete Jacobian: {err}")))?;
    for column in 0..p {
        for &row in &complete {
            data.push(jacobian[row + column * n] / scale);
        }
        cooperation.completed(rows as u64).await?;
    }
    let svd = DMatrix::from_vec(rows, p, data).svd(false, true);
    cooperation.completed(cells as u64).await?;
    let v_t = svd
        .v_t
        .ok_or_else(|| numerical("Jacobian SVD did not produce right singular vectors"))?;
    let largest = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = (rows.max(p) as f64) * f64::EPSILON * largest;
    if largest == 0.0
        || svd
            .singular_values
            .iter()
            .any(|&singular| !singular.is_finite() || singular <= tolerance)
    {
        return Err(numerical("Jacobian must have full column rank"));
    }
    let rmse = sumsq_scale * (sumsq / df as f64).sqrt();
    let mut standard_errors = Vec::new();
    standard_errors
        .try_reserve_exact(p)
        .map_err(|err| numerical(format!("cannot allocate interval variances: {err}")))?;
    for parameter in 0..p {
        let mut norm = 0.0f64;
        for component in 0..p {
            norm = norm.hypot(v_t[(component, parameter)] / svd.singular_values[component]);
        }
        standard_errors.push((rmse / scale) * norm);
        cooperation.completed(p as u64).await?;
    }
    Ok((df, standard_errors))
}
