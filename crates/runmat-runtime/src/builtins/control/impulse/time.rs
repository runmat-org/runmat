//! Impulse-response time validation and grid construction.
use super::*;
use crate::builtins::common::tensor;
use crate::builtins::common::validation::integer_is_exact_f64;
use runmat_value::{IntValue, Tensor};

fn ensure_vector(label: &str, shape: &[usize]) -> BuiltinResult<()> {
    let non_unit = shape.iter().copied().filter(|&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(impulse_error_with_detail(
            &IMPULSE_ERROR_INVALID_TIME,
            format!("{label} must be a vector"),
        ))
    }
}

fn exact_integer_as_f64(value: &IntValue, label: &str) -> BuiltinResult<f64> {
    if !integer_is_exact_f64(value) {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_INVALID_ARGUMENT,
            format!("{label} integer values must be exactly representable as double"),
        ));
    }
    Ok(value.to_f64())
}

fn ensure_exact_integer_tensor(tensor: &Tensor, label: &str) -> BuiltinResult<()> {
    if tensor.integer_storage().is_some_and(|storage| {
        storage
            .exact_values()
            .iter()
            .any(|value| !integer_is_exact_f64(value))
    }) {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_INVALID_ARGUMENT,
            format!("{label} integer values must be exactly representable as double"),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub(super) enum TimeSpec {
    Values(Vec<f64>),
}

impl TimeSpec {
    pub(super) fn preflight_host(rest: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
        if rest.len() == 1 {
            let value = rest.first().expect("one impulse time argument");
            if !matches!(value, Value::GpuTensor(_)) {
                if let Some(final_time) = scalar_time_from_value(value)? {
                    if !final_time.is_finite() || final_time < 0.0 {
                        return Err(impulse_error_with_detail(
                            &IMPULSE_ERROR_INVALID_TIME,
                            "final time must be finite and nonnegative",
                        ));
                    }
                } else {
                    let vector = time_vector_from_value(value.clone())?;
                    if vector
                        .iter()
                        .any(|number| !number.is_finite() || *number < 0.0)
                        || vector.windows(2).any(|pair| pair[1] <= pair[0])
                    {
                        return Err(impulse_error_with_detail(
                            &IMPULSE_ERROR_INVALID_TIME,
                            "time vector must be finite, nonnegative and strictly increasing",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    pub(super) async fn parse(
        system: &TfSystem,
        rest: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Self> {
        match rest.len() {
            0 => Ok(Self::Values(default_time_vector(system))),
            1 => {
                let value = rest.first().expect("one impulse time argument");
                let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
                if let Some(final_time) = scalar_time_from_value(&gathered)? {
                    return Ok(Self::Values(time_vector_from_final_time(
                        system, final_time,
                    )?));
                }
                let vector = time_vector_from_value(gathered)?;
                validate_time_vector(system, &vector)?;
                Ok(Self::Values(vector))
            }
            _ => Err(impulse_error_with_detail(
                &IMPULSE_ERROR_INVALID_ARGUMENT,
                "expected impulse(sys), impulse(sys, tFinal), or impulse(sys, t)",
            )),
        }
    }
}

fn scalar_time_from_value(value: &Value) -> BuiltinResult<Option<f64>> {
    match value {
        Value::Num(n) => Ok(Some(*n)),
        Value::Int(i) => Ok(Some(exact_integer_as_f64(i, "time")?)),
        Value::Bool(b) => Ok(Some(if *b { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            ensure_exact_integer_tensor(tensor, "time")?;
            Ok(Some(tensor::tensor_value_f64(tensor, 0)))
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(Some(if logical.data[0] == 0 { 0.0 } else { 1.0 }))
        }
        Value::Tensor(_) | Value::LogicalArray(_) => Ok(None),
        _ => Ok(None),
    }
}

fn default_time_vector(system: &TfSystem) -> Vec<f64> {
    if system.is_discrete() {
        (0..DEFAULT_POINTS)
            .map(|idx| idx as f64 * system.sample_time)
            .collect()
    } else {
        linspace(0.0, 10.0, DEFAULT_POINTS)
    }
}

fn time_vector_from_final_time(system: &TfSystem, final_time: f64) -> BuiltinResult<Vec<f64>> {
    if !final_time.is_finite() || final_time < 0.0 {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_INVALID_TIME,
            "final time must be a finite non-negative scalar",
        ));
    }
    if system.is_discrete() {
        let count = checked_discrete_sample_count(system, final_time)?;
        Ok((0..count)
            .map(|idx| idx as f64 * system.sample_time)
            .collect())
    } else if final_time == 0.0 {
        Ok(vec![0.0])
    } else {
        Ok(linspace(0.0, final_time, DEFAULT_POINTS))
    }
}

fn checked_discrete_sample_count(system: &TfSystem, final_time: f64) -> BuiltinResult<usize> {
    let samples = final_time / system.sample_time;
    if !samples.is_finite() {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_DISCRETE_LIMIT,
            "discrete sample count exceeds platform limits",
        ));
    }

    let count = samples.floor() + 1.0;
    if count > usize::MAX as f64 || count > MAX_DISCRETE_SAMPLES as f64 {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_DISCRETE_LIMIT,
            format!("discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"),
        ));
    }
    Ok(count as usize)
}

pub(super) fn checked_discrete_sample_index(system: &TfSystem, time: f64) -> BuiltinResult<usize> {
    let samples = time / system.sample_time;
    let index = samples.round();
    if !index.is_finite() || index > usize::MAX as f64 {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_DISCRETE_LIMIT,
            "discrete sample index exceeds platform limits",
        ));
    }
    if index >= MAX_DISCRETE_SAMPLES as f64 {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_DISCRETE_LIMIT,
            format!("discrete response would require more than {MAX_DISCRETE_SAMPLES} samples"),
        ));
    }
    Ok(index as usize)
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (stop - start) / ((count - 1) as f64);
    (0..count).map(|idx| start + idx as f64 * step).collect()
}

pub(super) fn time_vector_from_value(value: Value) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value).map_err(|err| {
        impulse_error_with_detail(
            &IMPULSE_ERROR_INVALID_TIME,
            format!("time vector must be numeric: {err}"),
        )
    })?;
    ensure_vector("time", tensor.shape())?;
    ensure_exact_integer_tensor(&tensor, "time")?;
    let values = tensor::tensor_into_values_f64(tensor);
    if values.is_empty() {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_INVALID_TIME,
            "time vector cannot be empty",
        ));
    }
    Ok(values)
}

fn validate_time_vector(system: &TfSystem, values: &[f64]) -> BuiltinResult<()> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_INVALID_TIME,
            "time vector values must be finite and non-negative",
        ));
    }
    if values.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_INVALID_TIME,
            "time vector values must be strictly increasing",
        ));
    }
    if system.is_discrete() {
        for &value in values {
            let samples = value / system.sample_time;
            if (samples - samples.round()).abs() > 1.0e-8 {
                return Err(impulse_error_with_detail(
                    &IMPULSE_ERROR_INVALID_TIME,
                    "discrete-time vectors must use integer multiples of the sample time",
                ));
            }
        }
    }
    Ok(())
}
