use runmat_analysis_core::{
    AnalysisModel, Direction, PhysicalSignal, ProcedureStep, Signal, SignalExtrapolation,
    SignalInterpolation, SignalKnot, SignalKnotValue, SignalOrdinate, SignalRepresentation,
    SignalSampleApproach, TabulatedSignalDomain,
};

use super::{
    model::{ComplexScalar, ComplexVector3},
    realize::RealizationError,
};

mod point;
pub(crate) use point::initial_signal_sample_point;
mod prescribed;
pub(crate) use prescribed::{
    sample_prescribed_current, sample_prescribed_heat_flux, sample_prescribed_potential,
    sample_prescribed_pressure, sample_prescribed_temperature,
    sample_prescribed_temperature_difference, sample_prescribed_velocity,
};
mod static_table;
mod values;

pub(crate) use values::ComplexTensor;
use values::{interpolate_value, ordinate_value, real_value, zero_value};

use static_table::sample_static_table;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ExcitationSamplePoint<'a> {
    Static,
    TimeSeconds {
        value: f64,
        approach: SignalSampleApproach,
    },
    FrequencyHertz {
        value: f64,
        approach: SignalSampleApproach,
    },
    StudyParameter {
        parameter_id: &'a str,
        canonical_value: f64,
        approach: SignalSampleApproach,
    },
}

impl ExcitationSamplePoint<'_> {
    pub(crate) fn time(value: f64) -> Self {
        Self::TimeSeconds {
            value,
            approach: SignalSampleApproach::At,
        }
    }

    pub(crate) fn frequency(value: f64) -> Self {
        Self::FrequencyHertz {
            value,
            approach: SignalSampleApproach::At,
        }
    }
}

mod time_events;
pub(crate) use time_events::{time_breakpoints, time_breakpoints_profiled};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SampledValue {
    Scalar(ComplexScalar),
    Vector3(ComplexVector3),
    Tensor(ComplexTensor),
}

pub(crate) fn sample_signal<Q>(
    signal: &Signal<Q>,
    sample: ExcitationSamplePoint<'_>,
    component: impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    match &signal.representation {
        SignalRepresentation::Constant(value) => real_value(value, &component),
        SignalRepresentation::Tabulated(table) => {
            let axis = table.coordinate_field(None).map_err(RealizationError::new)?;
            if matches!(sample, ExcitationSamplePoint::Static) {
                return sample_static_table(&table.knots, &component);
            }
            let (abscissa, approach) = sample_abscissa(sample, &table.domain)?;
            sample_table(
                &table.knots,
                &axis,
                abscissa,
                approach,
                table.interpolation,
                table.extrapolation,
                component,
            )
        }
        SignalRepresentation::Waveform(waveform) => match sample {
            ExcitationSamplePoint::TimeSeconds { value, approach } => {
                super::waveform::sample(waveform, value, approach, component)
            }
            _ => Err(RealizationError::new(
                "waveform signals require a time-domain sample point",
            )),
        },
        SignalRepresentation::Artifact(_) | SignalRepresentation::Generator(_) => Err(
            RealizationError::new(
                "artifact and generator signals must be materialized before FEA excitation realization",
            ),
        ),
    }
}

pub(crate) fn expect_scalar(value: SampledValue) -> Result<ComplexScalar, RealizationError> {
    match value {
        SampledValue::Scalar(value) if value.real.is_finite() && value.imaginary.is_finite() => {
            Ok(value)
        }
        SampledValue::Scalar(_) => Err(RealizationError::new(
            "scalar excitation requires finite real and imaginary components",
        )),
        SampledValue::Vector3(_) | SampledValue::Tensor(_) => {
            Err(RealizationError::new("excitation requires scalar signal"))
        }
    }
}

pub(crate) fn expect_vector3(value: SampledValue) -> Result<ComplexVector3, RealizationError> {
    match value {
        SampledValue::Vector3(value)
            if value
                .real
                .iter()
                .chain(&value.imaginary)
                .all(|v| v.is_finite()) =>
        {
            Ok(value)
        }
        SampledValue::Vector3(_) => Err(RealizationError::new("nonfinite vector3 excitation")),
        SampledValue::Scalar(_) | SampledValue::Tensor(_) => {
            Err(RealizationError::new("excitation requires vector3 signal"))
        }
    }
}

pub(crate) fn expect_vector3_or_direction(
    value: SampledValue,
    direction: Option<Direction>,
) -> Result<ComplexVector3, RealizationError> {
    let vector = match value {
        SampledValue::Vector3(value) => Ok(value),
        SampledValue::Tensor(_) => Err(RealizationError::new(
            "directional excitation requires scalar or vector3 signal",
        )),
        SampledValue::Scalar(value) => {
            let direction = direction.ok_or_else(|| {
                RealizationError::new("scalar directional excitation requires a direction")
            })?;
            Ok(ComplexVector3 {
                real: direction.vector.map(|component| component * value.real),
                imaginary: direction
                    .vector
                    .map(|component| component * value.imaginary),
            })
        }
    }?;
    expect_vector3(SampledValue::Vector3(vector))
}

fn sample_abscissa(
    sample: ExcitationSamplePoint<'_>,
    domain: &TabulatedSignalDomain,
) -> Result<
    (
        runmat_types::physical::DimensionedScalar,
        SignalSampleApproach,
    ),
    RealizationError,
> {
    let (value, approach) = match (sample, domain) {
        (
            ExcitationSamplePoint::TimeSeconds { value, approach },
            TabulatedSignalDomain::TimeSeconds,
        ) => (value, approach),
        (
            ExcitationSamplePoint::FrequencyHertz { value, approach },
            TabulatedSignalDomain::FrequencyHertz,
        ) => (value, approach),
        (
            ExcitationSamplePoint::StudyParameter {
                parameter_id,
                canonical_value,
                approach,
            },
            TabulatedSignalDomain::StudyParameter {
                parameter_id: axis_parameter,
                ..
            },
        ) if parameter_id == axis_parameter => (canonical_value, approach),
        _ => {
            return Err(RealizationError::new(
                "excitation sample point does not match the signal domain",
            ));
        }
    };
    if value.is_finite() {
        let descriptor = domain
            .axis_descriptor(None)
            .map_err(RealizationError::new)?;
        let value =
            runmat_types::physical::DimensionedScalar::try_from_descriptor(descriptor, value)
                .map_err(|e| RealizationError::new(e.to_string()))?;
        Ok((value, approach))
    } else {
        Err(RealizationError::new(
            "excitation sample point must be finite",
        ))
    }
}

fn sample_table<Q>(
    knots: &[SignalKnot<Q>],
    axis: &runmat_types::physical::Quantity<
        runmat_types::physical::DynamicDimension,
        runmat_analysis_core::problem::signal::SignalAxisCoordinates<'_, Q>,
    >,
    x: runmat_types::physical::DimensionedScalar,
    approach: SignalSampleApproach,
    interpolation: SignalInterpolation,
    extrapolation: SignalExtrapolation,
    component: impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    if axis.scalar_count() != knots.len() || !axis.descriptor().is_compatible_with(x.descriptor()) {
        return Err(RealizationError::new(
            "sample coordinate units do not match the signal axis",
        ));
    }
    // Generic interpolation now runs under the retained typed axis header.
    let x = x.canonical_value();
    if knots.is_empty()
        || knots.iter().any(|knot| !knot.abscissa.is_finite())
        || knots
            .windows(2)
            .any(|pair| pair[0].abscissa >= pair[1].abscissa)
    {
        return Err(RealizationError::new(
            "tabulated knots must have finite, strictly increasing abscissae",
        ));
    }
    let first = &knots[0];
    let last = &knots[knots.len() - 1];
    if x < first.abscissa || x > last.abscissa {
        return match extrapolation {
            SignalExtrapolation::Reject => {
                Err(RealizationError::new("sample lies outside signal domain"))
            }
            SignalExtrapolation::Zero => zero_value(point_ordinate(first), &component),
            SignalExtrapolation::HoldEndpoint => ordinate_value(
                point_ordinate(if x < first.abscissa { first } else { last }),
                &component,
            ),
        };
    }
    match knots.binary_search_by(|knot| knot.abscissa.total_cmp(&x)) {
        Ok(index) => sample_exact_knot(knots, index, approach, &component),
        Err(right) => {
            if approach != SignalSampleApproach::At {
                return Err(RealizationError::new(
                    "one-sided signal sampling requires an exact knot abscissa",
                ));
            }
            let left = right - 1;
            let left_value = ordinate_value(outgoing_ordinate(&knots[left]), &component)?;
            let right_value = ordinate_value(incoming_ordinate(&knots[right]), &component)?;
            let weight = match interpolation {
                SignalInterpolation::LeftEndpoint => 0.0,
                SignalInterpolation::RightEndpoint => 1.0,
                SignalInterpolation::Linear => {
                    (x - knots[left].abscissa) / (knots[right].abscissa - knots[left].abscissa)
                }
            };
            interpolate_value(left_value, right_value, weight)
        }
    }
}

fn point_ordinate<Q>(knot: &SignalKnot<Q>) -> &SignalOrdinate<Q> {
    match &knot.value {
        SignalKnotValue::Continuous(value) => value,
        SignalKnotValue::Jump {
            left_limit,
            right_limit,
            continuity,
        } => match continuity {
            runmat_analysis_core::SignalPointContinuity::Left => left_limit,
            runmat_analysis_core::SignalPointContinuity::Right => right_limit,
        },
    }
}

fn incoming_ordinate<Q>(knot: &SignalKnot<Q>) -> &SignalOrdinate<Q> {
    match &knot.value {
        SignalKnotValue::Continuous(value) => value,
        SignalKnotValue::Jump { left_limit, .. } => left_limit,
    }
}

fn outgoing_ordinate<Q>(knot: &SignalKnot<Q>) -> &SignalOrdinate<Q> {
    match &knot.value {
        SignalKnotValue::Continuous(value) => value,
        SignalKnotValue::Jump { right_limit, .. } => right_limit,
    }
}

fn sample_exact_knot<Q>(
    knots: &[SignalKnot<Q>],
    index: usize,
    approach: SignalSampleApproach,
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    if (approach == SignalSampleApproach::LeftLimit && index == 0)
        || (approach == SignalSampleApproach::RightLimit && index + 1 == knots.len())
    {
        return Err(RealizationError::new(
            "requested one-sided signal limit does not exist at the domain endpoint",
        ));
    }
    let ordinate = match approach {
        SignalSampleApproach::At => point_ordinate(&knots[index]),
        SignalSampleApproach::LeftLimit => incoming_ordinate(&knots[index]),
        SignalSampleApproach::RightLimit => outgoing_ordinate(&knots[index]),
    };
    ordinate_value(ordinate, component)
}
