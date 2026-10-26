use runmat_analysis_core::{
    DirectionalCombination, GeneratorDomain, ModalCombination, ModalTruncationPolicy,
    PhysicalQuantityKind, ResponseSpectrum, ResponseSpectrumId, ShockResponseSpectrumMethod,
    SignalShape, SpectrumAbscissa, SpectrumConstruction, SpectrumDamping, SpectrumDirection,
    SpectrumEnvelope, SpectrumEvidence, SpectrumExtrapolation, SpectrumInterpolation,
    SpectrumResponse,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::Serialize;

use crate::execution::{
    AnalysisExecutionError, AnalysisExecutionResult, CanonicalSignalRealization,
    CanonicalSignalValues,
};

use matrix::Matrix2;

mod matrix;

const PROVENANCE_PREFIX: &[u8] = b"runmat-shock-response-spectrum\0";
const PROVENANCE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(16 * 1024 * 1024, 1_000_000, 128, 16);
const PEAK_PHASE_SEGMENTS: f64 = 16.0;
const MAXIMUM_PEAK_SUBDIVISIONS: usize = 1_000_000;

/// Typed policy for deriving one shock-response spectrum from an admitted acceleration history.
///
/// The source samples and their identity live in [`CanonicalSignalRealization`]. Consequently this
/// request contains only spectrum semantics; callers cannot provide or forge construction
/// evidence independently of the numerical operation.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShockResponseSpectrumRequest {
    pub spectrum_id: ResponseSpectrumId,
    pub abscissa: SpectrumAbscissa,
    pub damping: SpectrumDamping,
    pub response: SpectrumResponse,
    pub envelope: SpectrumEnvelope,
    pub interpolation: SpectrumInterpolation,
    pub extrapolation: SpectrumExtrapolation,
    pub direction: SpectrumDirection,
    pub modal_combination: ModalCombination,
    pub directional_combination: DirectionalCombination,
    pub truncation: ModalTruncationPolicy,
    pub rigid_response: runmat_analysis_core::RigidResponsePolicy,
    pub method: ShockResponseSpectrumMethod,
}

/// Derives a shock-response spectrum using an exact state transition for each piecewise-linear
/// acceleration interval and a bounded stationary-point search inside every interval.
///
/// Discontinuous acceleration knots retain their authored left and right limits. Displacement and
/// velocity remain continuous across a jump; acceleration response is evaluated on both sides.
/// The currently admitted construction method is `ExactPiecewiseLinear`. The recursive-filter
/// vocabulary remains authored-only until it has an independently qualified implementation.
pub fn construct_shock_response_spectrum(
    source: &CanonicalSignalRealization,
    request: ShockResponseSpectrumRequest,
) -> AnalysisExecutionResult<ResponseSpectrum> {
    source.validate()?;
    validate_source(source)?;
    if request.method != ShockResponseSpectrumMethod::ExactPiecewiseLinear {
        return Err(invalid(
            "recursive-filter SRS construction is not implemented; use exact piecewise-linear construction",
        ));
    }

    let frequencies = frequencies_hz(&request.abscissa)?;
    let damping_ratio = damping_ratio(request.damping)?;
    let history = AccelerationHistory::from_realization(source)?;
    let ordinates = frequencies
        .iter()
        .map(|frequency| {
            oscillator_peak(
                &history,
                *frequency,
                damping_ratio,
                request.response,
                request.envelope,
            )
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;

    let maximum_interval = history
        .times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(0.0_f64, f64::max);
    let maximum_frequency = frequencies.iter().copied().fold(0.0_f64, f64::max);
    let samples_per_cycle = 1.0 / (maximum_interval * maximum_frequency);
    let sampling_adequacy_ratio = (samples_per_cycle / 20.0).min(1.0);
    let max_interpolation_error = source.evidence.max_absolute_error;
    let provenance_digest = spectrum_provenance(source, &request, &ordinates)?;
    let spectrum = ResponseSpectrum {
        spectrum_id: request.spectrum_id,
        abscissa: request.abscissa,
        ordinates: runmat_analysis_core::NumericalDataSource::Authored {
            values: ordinates,
            provenance_digest,
        },
        damping: request.damping,
        response: request.response,
        envelope: request.envelope,
        interpolation: request.interpolation,
        extrapolation: request.extrapolation,
        direction: request.direction,
        modal_combination: request.modal_combination,
        directional_combination: request.directional_combination,
        truncation: request.truncation,
        rigid_response: request.rigid_response,
        construction: SpectrumConstruction::DerivedFromAccelerationHistory {
            source_signal_id: source.signal_id.clone(),
            method: request.method,
        },
        evidence: SpectrumEvidence {
            sampling_adequacy_ratio,
            max_interpolation_error,
        },
    };
    spectrum.validate_intrinsic().map_err(invalid)?;
    Ok(spectrum)
}

fn validate_source(source: &CanonicalSignalRealization) -> AnalysisExecutionResult<()> {
    if source.quantity != PhysicalQuantityKind::Acceleration
        || source.shape != SignalShape::Scalar
        || source.domain != GeneratorDomain::TimeSeconds
        || !matches!(source.values, CanonicalSignalValues::Real { .. })
    {
        return Err(invalid(
            "SRS construction requires a scalar, real, time-domain acceleration realization",
        ));
    }
    Ok(())
}

fn frequencies_hz(abscissa: &SpectrumAbscissa) -> AnalysisExecutionResult<Vec<f64>> {
    let frequencies = match abscissa {
        SpectrumAbscissa::FrequencyHertz(values) => values.canonical_values().clone(),
        SpectrumAbscissa::PeriodSeconds(values) => values
            .canonical_values()
            .iter()
            .map(|period| 1.0 / period)
            .collect(),
    };
    if frequencies.is_empty()
        || frequencies
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return Err(invalid("SRS frequencies must be finite and positive"));
    }
    Ok(frequencies)
}

fn damping_ratio(damping: SpectrumDamping) -> AnalysisExecutionResult<f64> {
    let ratio = match damping {
        SpectrumDamping::Ratio(value) => value,
        SpectrumDamping::QualityFactor(value) => 1.0 / (2.0 * value),
    };
    if !ratio.is_finite() || !(0.0..1.0).contains(&ratio) {
        return Err(invalid(
            "SRS construction requires a damping ratio in [0, 1); quality factor must imply the same range",
        ));
    }
    Ok(ratio)
}

#[derive(Clone, Debug)]
struct AccelerationHistory {
    times: Vec<f64>,
    left: Vec<f64>,
    right: Vec<f64>,
}

impl AccelerationHistory {
    fn from_realization(source: &CanonicalSignalRealization) -> AnalysisExecutionResult<Self> {
        let CanonicalSignalValues::Real { values } = &source.values else {
            return Err(invalid("SRS acceleration values must be real"));
        };
        let mut cursor = 0_usize;
        let mut times = Vec::with_capacity(source.knots.len());
        let mut left = Vec::with_capacity(source.knots.len());
        let mut right = Vec::with_capacity(source.knots.len());
        for knot in &source.knots {
            times.push(knot.abscissa());
            let first = values
                .get(cursor)
                .copied()
                .map(f64::from_bits)
                .ok_or_else(|| invalid("SRS acceleration realization is truncated"))?;
            cursor += 1;
            if knot.continuity.is_some() {
                let second = values
                    .get(cursor)
                    .copied()
                    .map(f64::from_bits)
                    .ok_or_else(|| invalid("SRS jump acceleration realization is truncated"))?;
                cursor += 1;
                left.push(first);
                right.push(second);
            } else {
                left.push(first);
                right.push(first);
            }
        }
        if cursor != values.len() {
            return Err(invalid(
                "SRS acceleration realization has unrelated trailing values",
            ));
        }
        Ok(Self { times, left, right })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OscillatorState {
    displacement: f64,
    velocity: f64,
}

fn oscillator_peak(
    history: &AccelerationHistory,
    frequency_hz: f64,
    damping_ratio: f64,
    response: SpectrumResponse,
    envelope: SpectrumEnvelope,
) -> AnalysisExecutionResult<f64> {
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;
    let mut state = OscillatorState::default();
    let mut peak = SignedPeak::default();
    peak.observe(response_value(
        state,
        history.right[0],
        omega,
        damping_ratio,
        response,
    ));

    for interval in 0..history.times.len() - 1 {
        let duration = history.times[interval + 1] - history.times[interval];
        let start_acceleration = history.right[interval];
        let end_acceleration = history.left[interval + 1];
        let slope = (end_acceleration - start_acceleration) / duration;
        let subdivisions = ((omega * duration * PEAK_PHASE_SEGMENTS / std::f64::consts::PI).ceil()
            as usize)
            .max(1);
        if subdivisions > MAXIMUM_PEAK_SUBDIVISIONS {
            return Err(AnalysisExecutionError::Limit(
                "shock-response stationary-point subdivisions",
            ));
        }
        let evaluate = |time: f64| -> AnalysisExecutionResult<(f64, f64)> {
            let point = advance_piecewise_linear(
                state,
                start_acceleration,
                slope,
                time,
                omega,
                damping_ratio,
            )?;
            let acceleration = start_acceleration + slope * time;
            Ok((
                response_value(point, acceleration, omega, damping_ratio, response),
                response_derivative(point, acceleration, slope, omega, damping_ratio, response),
            ))
        };

        let mut prior_time = 0.0;
        let (mut prior_value, mut prior_derivative) = evaluate(0.0)?;
        peak.observe(prior_value);
        for subdivision in 1..=subdivisions {
            let time = duration * subdivision as f64 / subdivisions as f64;
            let (value, derivative) = evaluate(time)?;
            peak.observe(value);
            if prior_derivative == 0.0 {
                peak.observe(prior_value);
            } else if derivative == 0.0
                || prior_derivative.is_sign_positive() != derivative.is_sign_positive()
            {
                let root = stationary_point(&evaluate, prior_time, time, prior_derivative)?;
                peak.observe(evaluate(root)?.0);
            }
            prior_time = time;
            prior_value = value;
            prior_derivative = derivative;
        }
        state = advance_piecewise_linear(
            state,
            start_acceleration,
            slope,
            duration,
            omega,
            damping_ratio,
        )?;
        peak.observe(response_value(
            state,
            history.right[interval + 1],
            omega,
            damping_ratio,
            response,
        ));
    }
    Ok(peak.ordinate(envelope))
}

fn stationary_point(
    evaluate: &impl Fn(f64) -> AnalysisExecutionResult<(f64, f64)>,
    mut left: f64,
    mut right: f64,
    mut left_derivative: f64,
) -> AnalysisExecutionResult<f64> {
    for _ in 0..64 {
        let middle = 0.5 * (left + right);
        let derivative = evaluate(middle)?.1;
        if derivative == 0.0 || right - left <= 8.0 * f64::EPSILON * right.abs().max(1.0) {
            return Ok(middle);
        }
        if derivative.is_sign_positive() == left_derivative.is_sign_positive() {
            left = middle;
            left_derivative = derivative;
        } else {
            right = middle;
        }
    }
    Ok(0.5 * (left + right))
}

fn advance_piecewise_linear(
    state: OscillatorState,
    start_acceleration: f64,
    slope: f64,
    duration: f64,
    omega: f64,
    damping_ratio: f64,
) -> AnalysisExecutionResult<OscillatorState> {
    let a = Matrix2::new(0.0, 1.0, -omega * omega, -2.0 * damping_ratio * omega);
    let transition = (a * duration).exp()?;
    let inverse = a.inverse()?;
    let identity = Matrix2::identity();
    let constant_integral = inverse * (transition - identity);
    let linear_integral = inverse * (constant_integral - identity * duration);
    let homogeneous = transition.apply([state.displacement, state.velocity]);
    let input = [0.0, -1.0];
    let constant = constant_integral.apply(input);
    let linear = linear_integral.apply(input);
    let displacement = homogeneous[0] + constant[0] * start_acceleration + linear[0] * slope;
    let velocity = homogeneous[1] + constant[1] * start_acceleration + linear[1] * slope;
    if !displacement.is_finite() || !velocity.is_finite() {
        return Err(invalid("SRS oscillator state is non-finite"));
    }
    Ok(OscillatorState {
        displacement,
        velocity,
    })
}

fn response_value(
    state: OscillatorState,
    base_acceleration: f64,
    omega: f64,
    damping_ratio: f64,
    response: SpectrumResponse,
) -> f64 {
    let relative_acceleration = -2.0 * damping_ratio * omega * state.velocity
        - omega * omega * state.displacement
        - base_acceleration;
    match response {
        SpectrumResponse::RelativeDisplacement => state.displacement,
        SpectrumResponse::RelativeVelocity => state.velocity,
        SpectrumResponse::RelativeAcceleration => relative_acceleration,
        SpectrumResponse::AbsoluteAcceleration => relative_acceleration + base_acceleration,
        SpectrumResponse::PseudoVelocity => omega * state.displacement,
        SpectrumResponse::PseudoAcceleration => omega * omega * state.displacement,
    }
}

fn response_derivative(
    state: OscillatorState,
    base_acceleration: f64,
    base_acceleration_slope: f64,
    omega: f64,
    damping_ratio: f64,
    response: SpectrumResponse,
) -> f64 {
    let relative_acceleration = -2.0 * damping_ratio * omega * state.velocity
        - omega * omega * state.displacement
        - base_acceleration;
    let relative_jerk = -2.0 * damping_ratio * omega * relative_acceleration
        - omega * omega * state.velocity
        - base_acceleration_slope;
    match response {
        SpectrumResponse::RelativeDisplacement => state.velocity,
        SpectrumResponse::RelativeVelocity => relative_acceleration,
        SpectrumResponse::RelativeAcceleration => relative_jerk,
        SpectrumResponse::AbsoluteAcceleration => relative_jerk + base_acceleration_slope,
        SpectrumResponse::PseudoVelocity => omega * state.velocity,
        SpectrumResponse::PseudoAcceleration => omega * omega * state.velocity,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SignedPeak {
    positive: f64,
    negative: f64,
}

impl SignedPeak {
    fn observe(&mut self, value: f64) {
        self.positive = self.positive.max(value);
        self.negative = self.negative.max(-value);
    }

    fn ordinate(self, envelope: SpectrumEnvelope) -> f64 {
        match envelope {
            SpectrumEnvelope::Positive => self.positive,
            SpectrumEnvelope::Negative => self.negative,
            SpectrumEnvelope::Maximax => self.positive.max(self.negative),
        }
    }
}

fn spectrum_provenance(
    source: &CanonicalSignalRealization,
    request: &ShockResponseSpectrumRequest,
    ordinates: &[f64],
) -> AnalysisExecutionResult<String> {
    encode_contract(
        PROVENANCE_PREFIX,
        "shock-response-spectrum-provenance",
        &(source.identity, request, ordinates),
        PROVENANCE_LIMITS,
    )
    .map(Digest::sha256)
    .map(|digest| digest.to_string())
    .map_err(Into::into)
}

fn invalid(detail: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(detail.into())
}

#[cfg(test)]
mod tests;
