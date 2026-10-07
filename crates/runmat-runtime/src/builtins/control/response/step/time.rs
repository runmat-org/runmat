//! Validation and construction of step-response time grids.
use super::*;
use crate::builtins::common::tensor;

#[derive(Clone, Debug)]
pub(crate) enum TimeSpec {
    Auto,
    FinalTime(f64),
    Vector(Vec<f64>),
}

impl TimeSpec {
    pub(super) fn preflight_host(value: Option<&Value>) -> SimulationResult<()> {
        if let Some(value) = value.filter(|value| !matches!(value, Value::GpuTensor(_))) {
            Self::parse(Some(value), 0.0)?;
        }
        Ok(())
    }
    pub(super) fn parse(value: Option<&Value>, sample_time: f64) -> SimulationResult<Self> {
        let Some(value) = value else {
            return Ok(Self::Auto);
        };
        match value {
            Value::Num(n) => Self::final_time(*n),
            Value::Int(i) => Self::final_time(i.to_f64()),
            Value::Tensor(tensor) => {
                ensure_time_vector_shape(tensor.shape())?;
                let data = tensor::tensor_values_f64(tensor);
                if tensor::is_scalar_tensor(tensor) {
                    return Self::final_time(tensor::tensor_value_f64(tensor, 0));
                }
                Self::vector(data, sample_time)
            }
            other => Err(simulation_error(
                SimulationErrorKind::InvalidTime,
                format!("time input must be a scalar final time or numeric vector, got {other:?}"),
            )),
        }
    }

    fn final_time(value: f64) -> SimulationResult<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(simulation_error(
                SimulationErrorKind::InvalidTime,
                "final time must be a positive finite scalar",
            ));
        }
        Ok(Self::FinalTime(value))
    }

    fn vector(values: Vec<f64>, sample_time: f64) -> SimulationResult<Self> {
        validate_time_vector(&values)?;
        if sample_time > 0.0 {
            for &t in &values {
                let k = (t / sample_time).round();
                if (t - k * sample_time).abs() > 1.0e-8 * sample_time.max(1.0) {
                    return Err(simulation_error(
                        SimulationErrorKind::InvalidTime,
                        "discrete-time sample vector must align with the model sample time",
                    ));
                }
            }
        }
        Ok(Self::Vector(values))
    }
}

fn ensure_time_vector_shape(shape: &[usize]) -> SimulationResult<()> {
    let non_unit = shape.iter().copied().filter(|&dim| dim > 1).count();
    if non_unit <= 1 {
        Ok(())
    } else {
        Err(simulation_error(
            SimulationErrorKind::InvalidTime,
            "time input must be a vector",
        ))
    }
}

fn validate_time_vector(values: &[f64]) -> SimulationResult<()> {
    if values.is_empty() {
        return Err(simulation_error(
            SimulationErrorKind::InvalidTime,
            "time vector must not be empty",
        ));
    }
    let mut previous = None;
    for &value in values {
        if !value.is_finite() || value < 0.0 {
            return Err(simulation_error(
                SimulationErrorKind::InvalidTime,
                "time vector values must be finite and non-negative",
            ));
        }
        if let Some(prev) = previous {
            if value < prev {
                return Err(simulation_error(
                    SimulationErrorKind::InvalidTime,
                    "time vector must be nondecreasing",
                ));
            }
        }
        previous = Some(value);
    }
    Ok(())
}
