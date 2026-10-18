//! Shared SISO transfer-function frequency evaluation.
//!
//! This is the one point evaluator for `evalfr`, `freqresp`, `bode`, Nyquist,
//! and the transfer-function `bandwidth` overload. The existing Nyquist
//! frontend will delegate to this owner when this leaf is linked. All model
//! and result allocation happens only after a real host-policy checkpoint.

mod bandwidth;

use num_complex::Complex64;
use runmat_value::{Tensor, Value};

use crate::builtins::control::tf_model::{
    control_error, polynomial_roots, PhysicalPolynomials, TfModel, TfSampleDomain,
};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

pub(crate) use bandwidth::bandwidth;

const DEFAULT_GRID_POINTS: usize = 200;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrequencyApi {
    Evalfr,
    Freqresp,
    Bode,
    Bandwidth,
    Nyquist,
}

impl FrequencyApi {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Evalfr => "evalfr",
            Self::Freqresp => "freqresp",
            Self::Bode => "bode",
            Self::Bandwidth => "bandwidth",
            Self::Nyquist => "nyquist",
        }
    }
    const fn argument_identifier(self) -> &'static str {
        match self {
            Self::Evalfr => "RunMat:evalfr:InvalidArgument",
            Self::Freqresp => "RunMat:freqresp:InvalidArgument",
            Self::Bode => "RunMat:bode:InvalidArgument",
            Self::Bandwidth => "RunMat:bandwidth:InvalidArgument",
            Self::Nyquist => "RunMat:nyquist:InvalidArgument",
        }
    }
    const fn unsupported_identifier(self) -> &'static str {
        match self {
            Self::Evalfr => "RunMat:evalfr:UnsupportedModel",
            Self::Freqresp => "RunMat:freqresp:UnsupportedModel",
            Self::Bode => "RunMat:bode:UnsupportedModel",
            Self::Bandwidth => "RunMat:bandwidth:UnsupportedModel",
            Self::Nyquist => "RunMat:nyquist:UnsupportedModel",
        }
    }
}

impl std::fmt::Display for FrequencyApi {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

fn argument(name: FrequencyApi, message: impl Into<String>) -> RuntimeError {
    control_error(name.as_str(), name.argument_identifier(), message)
}

fn unsupported(name: FrequencyApi, message: impl Into<String>) -> RuntimeError {
    control_error(name.as_str(), name.unsupported_identifier(), message)
}

fn invalid_grid(name: FrequencyApi, message: impl Into<String>) -> RuntimeError {
    if name == FrequencyApi::Nyquist {
        control_error(name.as_str(), "RunMat:nyquist:InvalidModel", message)
    } else {
        unsupported(name, message)
    }
}

fn admission(
    name: FrequencyApi,
    runtime: Option<&RuntimeContext>,
    error: HostAllocationError,
) -> RuntimeError {
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    crate::builtin::builtin_control_error(cancellation.as_ref(), name.as_str(), error)
}

/// Current numerical forms use existing unbounded host result owners. A
/// managed caller must not start gathering model properties or frequency
/// inputs until a complete output/metadata owner is added to this family.
pub(super) fn require_host_owner(
    name: FrequencyApi,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    match HostAllocationReceipt::policy(runtime).map_err(|error| admission(name, runtime, error))? {
        HostAllocationPolicy::Unbounded => Ok(()),
        HostAllocationPolicy::Managed => Err(admission(
            name,
            runtime,
            HostAllocationError::MissingManagedOwner,
        )),
    }
}

pub(super) async fn model(
    name: FrequencyApi,
    value: Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<TfModel> {
    require_host_owner(name, runtime)?;
    let model = TfModel::from_value_async(value, name.as_str()).await?;
    model.ensure_zero_delays(name.as_str())?;
    Ok(model)
}

/// Evaluate the already-validated canonical model; no parallel polynomial
/// implementation or delay approximation is introduced here.
pub(super) fn at_point(
    name: FrequencyApi,
    model: &TfModel,
    point: Complex64,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Complex64> {
    require_host_owner(name, runtime)?;
    if !point.re.is_finite() || !point.im.is_finite() {
        return Err(argument(
            name,
            format!("{name}: evaluation point must be finite"),
        ));
    }
    let physical = model.physical_polynomials();
    checked_at_point(name, &physical, point)
}

fn checked_at_point(
    name: FrequencyApi,
    physical: &PhysicalPolynomials<'_>,
    point: Complex64,
) -> BuiltinResult<Complex64> {
    let value = physical.evaluate(point).ok_or_else(|| {
        argument(
            name,
            format!("{name}: response is singular at the requested point"),
        )
    })?;
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(argument(
            name,
            format!("{name}: finite response cannot be represented at the requested point"),
        ));
    }
    Ok(value)
}

pub(super) fn point_at_frequency(
    name: FrequencyApi,
    model: &TfModel,
    frequency: f64,
) -> BuiltinResult<Complex64> {
    if !frequency.is_finite() || frequency < 0.0 {
        return Err(argument(
            name,
            format!("{name}: frequencies must be finite and nonnegative"),
        ));
    }
    match model.sample_domain() {
        TfSampleDomain::Continuous => Ok(Complex64::new(0.0, frequency)),
        TfSampleDomain::Discrete(period) => {
            let phase = frequency * period.value();
            if !phase.is_finite() {
                return Err(argument(
                    name,
                    format!("{name}: frequency and sample time overflow the phase range"),
                ));
            }
            Ok(Complex64::new(phase.cos(), phase.sin()))
        }
        TfSampleDomain::DiscreteUnspecified => Err(unsupported(
            name,
            format!("{name}: discrete sample time is unspecified"),
        )),
    }
}

pub(super) fn evaluate_grid(
    name: FrequencyApi,
    model: &TfModel,
    frequencies: &[f64],
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<Complex64>> {
    require_host_owner(name, runtime)?;
    let physical = model.physical_polynomials();
    let mut response = Vec::with_capacity(frequencies.len());
    for (index, &frequency) in frequencies.iter().enumerate() {
        if index % 64 == 0 {
            require_host_owner(name, runtime)?;
        }
        let point = point_at_frequency(name, model, frequency)?;
        response.push(checked_at_point(name, &physical, point)?);
    }
    Ok(response)
}

pub(super) fn point_argument(name: FrequencyApi, value: &Value) -> BuiltinResult<Complex64> {
    let point = match value {
        Value::Num(real) => Complex64::new(*real, 0.0),
        Value::Complex(real, imaginary) => Complex64::new(*real, *imaginary),
        Value::Tensor(tensor) if tensor.len() == 1 && tensor.integer_storage().is_none() => {
            match (tensor.as_f64_slice(), tensor.as_f32_slice()) {
                (Some(values), _) => Complex64::new(values[0], 0.0),
                (_, Some(values)) => Complex64::new(f64::from(values[0]), 0.0),
                _ => return Err(argument(name, format!("{name}: expected a numeric scalar"))),
            }
        }
        Value::ComplexTensor(tensor) if tensor.len() == 1 && tensor.integer_storage().is_none() => {
            match (tensor.as_f64_slice(), tensor.as_f32_slice()) {
                (Some(values), _) => Complex64::new(values[0].0, values[0].1),
                (_, Some(values)) => Complex64::new(f64::from(values[0].0), f64::from(values[0].1)),
                _ => return Err(argument(name, format!("{name}: expected a numeric scalar"))),
            }
        }
        _ => return Err(argument(name, format!("{name}: expected a numeric scalar"))),
    };
    if !point.re.is_finite() || !point.im.is_finite() {
        return Err(argument(name, format!("{name}: point must be finite")));
    }
    Ok(point)
}

fn validate_grid(
    name: FrequencyApi,
    value: &Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    let invalid = || {
        argument(
            name,
            format!("{name}: expected a nonempty real host frequency vector"),
        )
    };
    match value {
        Value::Num(value) if value.is_finite() && *value >= 0.0 => Ok(()),
        Value::Tensor(tensor)
            if tensor.shape().iter().filter(|&&extent| extent > 1).count() <= 1
                && tensor.integer_storage().is_none() =>
        {
            let valid = match (tensor.as_f64_slice(), tensor.as_f32_slice()) {
                (Some(values), _) => {
                    if values.is_empty() {
                        false
                    } else {
                        for chunk in values.chunks(4096) {
                            require_host_owner(name, runtime)?;
                            if chunk.iter().any(|v| !v.is_finite() || *v < 0.0) {
                                return Err(invalid());
                            }
                        }
                        true
                    }
                }
                (_, Some(values)) => {
                    if values.is_empty() {
                        false
                    } else {
                        for chunk in values.chunks(4096) {
                            require_host_owner(name, runtime)?;
                            if chunk.iter().any(|v| !v.is_finite() || *v < 0.0) {
                                return Err(invalid());
                            }
                        }
                        true
                    }
                }
                _ => false,
            };
            if valid {
                Ok(())
            } else {
                Err(invalid())
            }
        }
        _ => Err(invalid()),
    }
}

pub(super) fn explicit_grid(
    name: FrequencyApi,
    value: &Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<f64>> {
    require_host_owner(name, runtime)?;
    validate_grid(name, value, runtime)?;
    match value {
        Value::Num(value) => Ok(vec![*value]),
        Value::Tensor(tensor) => match (tensor.as_f64_slice(), tensor.as_f32_slice()) {
            (Some(values), _) => {
                let mut result = Vec::with_capacity(values.len());
                for chunk in values.chunks(4096) {
                    require_host_owner(name, runtime)?;
                    result.extend_from_slice(chunk);
                }
                Ok(result)
            }
            (_, Some(values)) => {
                let mut result = Vec::with_capacity(values.len());
                for chunk in values.chunks(4096) {
                    require_host_owner(name, runtime)?;
                    result.extend(chunk.iter().copied().map(f64::from));
                }
                Ok(result)
            }
            _ => unreachable!("validated frequency tensor has a real floating storage"),
        },
        _ => unreachable!("validated frequency input is a real host scalar or tensor"),
    }
}

pub(super) fn default_grid(name: FrequencyApi, model: &TfModel) -> BuiltinResult<Vec<f64>> {
    match model.sample_domain() {
        TfSampleDomain::Discrete(period) => {
            let stop = std::f64::consts::PI / period.value();
            if !stop.is_finite() {
                return Err(invalid_grid(
                    name,
                    format!("{name}: sample time has no finite Nyquist grid"),
                ));
            }
            return Ok((0..DEFAULT_GRID_POINTS)
                .map(|index| stop * (index + 1) as f64 / (DEFAULT_GRID_POINTS + 1) as f64)
                .collect());
        }
        TfSampleDomain::DiscreteUnspecified => {
            return Err(unsupported(
                name,
                format!("{name}: discrete sample time is unspecified"),
            ));
        }
        TfSampleDomain::Continuous => {}
    }
    let mut breakpoints = Vec::new();
    let physical = model.physical_polynomials();
    for coefficients in [physical.numerator(), physical.denominator()] {
        breakpoints.extend(
            polynomial_roots(coefficients, name.as_str())?
                .into_iter()
                .map(|root| root.norm())
                .filter(|value| value.is_finite() && *value > 0.0),
        );
    }
    let (start, stop) = if breakpoints.is_empty() {
        (1.0e-2, 1.0e2)
    } else {
        let min = breakpoints.iter().copied().fold(f64::INFINITY, f64::min);
        let max = breakpoints.iter().copied().fold(0.0, f64::max);
        let start = (min / 100.0).max(1.0e-4);
        let stop = (max * 100.0).max(start * 10.0);
        (start, stop)
    };
    if !start.is_finite() || !stop.is_finite() {
        return Err(invalid_grid(
            name,
            format!("{name}: no finite default frequency grid"),
        ));
    }
    let first = start.log10();
    let span = stop.log10() - first;
    Ok((0..DEFAULT_GRID_POINTS)
        .map(|index| 10.0_f64.powf(first + span * index as f64 / (DEFAULT_GRID_POINTS - 1) as f64))
        .collect())
}

pub(super) fn frequency_column(name: FrequencyApi, values: Vec<f64>) -> BuiltinResult<Value> {
    let count = values.len();
    Tensor::new(values, vec![count, 1])
        .map(Value::Tensor)
        .map_err(|error| argument(name, format!("{name}: frequency output failed: {error}")))
}
