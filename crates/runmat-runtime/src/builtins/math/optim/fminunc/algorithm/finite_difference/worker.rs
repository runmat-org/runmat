//! Identical difference arithmetic over bounded ordered worker callbacks.
use super::*;

impl ObjectiveEvaluator<'_> {
    pub(super) async fn worker_gradient(
        &mut self,
        x: &[f64],
        f0: f64,
        central: bool,
    ) -> BuiltinResult<Vec<f64>> {
        let count = x
            .len()
            .checked_mul(if central { 2 } else { 1 })
            .ok_or_else(|| difference_error("finite-difference evaluation count overflows"))?;
        if count > self.remaining_func_evals() {
            return Err(difference_error(
                "MaxFunEvals cannot admit the complete difference sweep",
            ));
        }
        // Admit every actual point before publishing any wave. The immutable
        // axis scales are shared; only one compact point pair is retained per coordinate.
        let mut points = runmat_value::exact_capacity_vec(x.len()).map_err(|source| {
            crate::build_runtime_error("finite-difference point allocation failed")
                .with_source(source)
                .build()
        })?;
        for (index, &value) in x.iter().enumerate() {
            points.push(self.scaling.unbounded(NAME, value, index, central)?);
            self.cooperation.completed(1).await?;
        }
        let width = self
            .parallel
            .as_ref()
            .expect("selected worker route")
            .width();
        let mut gradient = vectors::zeroed_like(x, self.cooperation).await?;
        let mut plus_value = 0.0;
        let mut start = 0;
        while start < count {
            let end = start.saturating_add(width).min(count);
            let mut arguments = Vec::with_capacity(end - start);
            for ordinal in start..end {
                let index = if central { ordinal / 2 } else { ordinal };
                let mut point = vectors::copied(x, self.cooperation).await?;
                if central {
                    let plus = points[index].first;
                    let minus = points[index].second.expect("preflighted central pair");
                    point[index] = if ordinal % 2 == 0 { plus } else { minus };
                } else {
                    point[index] = points[index].first;
                }
                arguments.push(vec![vector_to_value(
                    NAME,
                    point,
                    &self.shape,
                    self.scalar,
                )?]);
            }
            let values = {
                let (_, budget) = self.cooperation.runtime_budget();
                self.parallel
                    .as_ref()
                    .expect("selected worker route")
                    .execute(
                        arguments,
                        1,
                        budget.expect("selected parallel route has original cooperation budget"),
                    )?
                    .await?
            };
            self.func_count += values.len();
            if values.len() != end - start {
                return Err(difference_error(
                    "worker batch did not return its admitted output cardinality",
                ));
            }
            for (ordinal, sequence) in (start..end).zip(values) {
                let value = {
                    let _entry = self
                        .cooperation
                        .runtime()
                        .map(crate::context::RuntimeContext::enter);
                    sequence.try_into_single().map_err(|mut error| {
                        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                        error
                    })?
                };
                let value = self.scalar_result(value).await?;
                if central {
                    if ordinal % 2 == 0 {
                        plus_value = value;
                        continue;
                    }
                    let index = ordinal / 2;
                    let plus = points[index].first;
                    let minus = points[index].second.expect("preflighted central pair");
                    let delta = plus_value - value;
                    gradient[index] =
                        if delta.is_infinite() && plus_value.is_finite() && value.is_finite() {
                            plus_value / (plus - minus) - value / (plus - minus)
                        } else {
                            delta / (plus - minus)
                        };
                    if !gradient[index].is_finite() {
                        return Err(difference_error(
                            "central finite-difference gradient is not finite",
                        ));
                    }
                } else {
                    gradient[ordinal] = (value - f0) / (points[ordinal].first - x[ordinal]);
                }
            }
            start = end;
        }
        Ok(gradient)
    }
}
fn difference_error(detail: &str) -> RuntimeError {
    let mut error = fminunc_error_with_detail(&FMINUNC_ERROR_INVALID_INPUT, detail);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
