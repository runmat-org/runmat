//! Shared admission and cooperative scans for one-way inferential statistics.
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use crate::{BuiltinResult, RuntimeError};
use runmat_types::{NumericClass, NumericDomain, NumericFact, ValueKindFact};
use runmat_value::{Tensor, Value};

pub(super) fn invalid(name: &'static str, message: impl std::fmt::Display) -> RuntimeError {
    crate::build_runtime_error(format!("{name}: {message}"))
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:InvalidArgument"))
        .build()
}

pub(super) fn real_sample(name: &'static str, value: &Value) -> BuiltinResult<()> {
    if matches!(
        crate::value_fact::value_fact(value).kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double | NumericClass::Single,
            domain: NumericDomain::Real
        })
    ) {
        Ok(())
    } else {
        Err(invalid(
            name,
            "sample data must be real single or double arrays",
        ))
    }
}

pub(super) async fn gather(
    name: &'static str,
    value: &Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Tensor> {
    admit_automatic_host_value(name, value)?;
    real_sample(name, value)?;
    let value = crate::dispatcher::gather_if_needed_in_context_async(value, runtime).await?;
    crate::builtins::common::tensor::value_into_tensor_for(name, value)
        .map_err(|error| invalid(name, error))
}

/// Host-only statistics may undo automatic placement, not explicit device
/// intent. Preflight the whole call before transferring even an earlier input.
pub(super) fn admit_automatic_host_arguments(
    name: &'static str,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    for value in args {
        admit_automatic_host_value(name, value)?;
    }
    Ok(())
}
fn admit_automatic_host_value(name: &'static str, value: &Value) -> BuiltinResult<()> {
    use crate::builtins::common::residency::{host_gather_intent, HostGatherIntent};
    if host_gather_intent(value)? == HostGatherIntent::ExplicitOrUnproven {
        let mut error=invalid(name,"host fallback requires automatic residency; explicit gpuArray or unproven device intent is unsupported (use gather explicitly)");
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        Err(error)
    } else {
        Ok(())
    }
}

pub(super) struct Cooperation<'a> {
    runtime: Option<&'a RuntimeContext>,
    budget: Option<CpuCooperationBudget>,
}
impl<'a> Cooperation<'a> {
    pub(super) fn new(runtime: Option<&'a RuntimeContext>) -> Self {
        Self {
            runtime,
            budget: runtime.map(RuntimeContext::cpu_cooperation),
        }
    }
    pub(super) async fn completed(&mut self, work: usize) -> BuiltinResult<()> {
        if let Some(budget) = self.budget.as_mut() {
            budget
                .checkpoint(work as u64)
                .await
                .map_err(|error| error.into_runtime_error_in_context(self.runtime))?;
        }
        Ok(())
    }
}

pub(super) fn numeric(
    name: &'static str,
    data: Vec<f64>,
    shape: Vec<usize>,
) -> BuiltinResult<Value> {
    if data.len() == 1 {
        return Ok(Value::Num(data[0]));
    }
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(|error| invalid(name, error))
}

/// Shift before accumulation; scaling avoids squaring large deviations before
/// division. Nonfinite observations deliberately produce undefined statistics.
pub(super) struct Moments {
    pub(super) count: usize,
    pub(super) mean: f64,
    pub(super) scale: f64,
    pub(super) normalized_squares: f64,
}
pub(super) async fn moments(
    values: &[f64],
    cooperation: &mut Cooperation<'_>,
) -> BuiltinResult<Moments> {
    if values.is_empty() {
        return Ok(Moments {
            count: 0,
            mean: f64::NAN,
            scale: f64::NAN,
            normalized_squares: f64::NAN,
        });
    }
    let origin = values[0];
    let mut shifted = 0.0;
    let mut correction = 0.0;
    let mut all_finite = true;
    let mut value_scale = 0.0f64;
    for chunk in values.chunks(1024) {
        for &value in chunk {
            all_finite &= value.is_finite();
            value_scale = value_scale.max(value.abs());
            let difference = value - origin;
            let contribution =
                if difference.is_infinite() && value.is_finite() && origin.is_finite() {
                    value / values.len() as f64 - origin / values.len() as f64
                } else {
                    difference / values.len() as f64
                };
            let next = contribution - correction;
            let sum = shifted + next;
            correction = (sum - shifted) - next;
            shifted = sum;
        }
        cooperation.completed(chunk.len()).await?;
    }
    let mut mean = origin + shifted;
    if !mean.is_finite() && all_finite {
        let scale = value_scale;
        let mut normalized = 0.0;
        let mut correction = 0.0;
        for chunk in values.chunks(1024) {
            for &value in chunk {
                let next = (value / scale) / values.len() as f64 - correction;
                let sum = normalized + next;
                correction = (sum - normalized) - next;
                normalized = sum;
            }
            cooperation.completed(chunk.len()).await?;
        }
        mean = scale * normalized.clamp(-1.0, 1.0);
    }
    let mut scale = 0.0f64;
    let mut sum = 1.0;
    for chunk in values.chunks(1024) {
        for &value in chunk {
            let deviation = (value - mean).abs();
            if !value.is_finite() || !mean.is_finite() {
                return Ok(Moments {
                    count: values.len(),
                    mean,
                    scale: f64::NAN,
                    normalized_squares: f64::NAN,
                });
            }
            if deviation.is_infinite() {
                let normalized = (value / f64::MAX - mean / f64::MAX).abs();
                sum = sum * (scale / f64::MAX).powi(2) + normalized * normalized;
                scale = f64::MAX;
                continue;
            }
            if deviation != 0.0 {
                if scale < deviation {
                    sum = 1.0 + sum * (scale / deviation).powi(2);
                    scale = deviation;
                } else {
                    sum += (deviation / scale).powi(2);
                }
            }
        }
        cooperation.completed(chunk.len()).await?;
    }
    Ok(Moments {
        count: values.len(),
        mean,
        scale,
        normalized_squares: sum,
    })
}

#[cfg(test)]
mod tests;
