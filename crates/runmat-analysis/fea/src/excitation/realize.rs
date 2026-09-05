use std::collections::{HashMap, HashSet};
use std::fmt;

use runmat_analysis_core::{
    AnalysisModel, ExcitationAction, ExcitationCaseActivation, ExcitationTarget, PhysicalSignal,
    Signal, SignalAxis, SignalExtrapolation, SignalInterpolation, SignalOrdinate,
    SignalRepresentation, SignalValue,
};

use super::model::{
    ComplexScalar, ComplexVector3, RealizedExcitation, RealizedExcitationKind,
    RealizedExcitationTarget, RealizedExcitations,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ExcitationSamplePoint {
    Initial,
    FrequencyHertz(f64),
}

pub(crate) fn realize_model_excitations(
    model: &AnalysisModel,
) -> Result<RealizedExcitations, RealizationError> {
    let sample = model
        .electromagnetic
        .as_ref()
        .map(|domain| ExcitationSamplePoint::FrequencyHertz(domain.reference_frequency.hertz))
        .unwrap_or(ExcitationSamplePoint::Initial);
    let step_id = model
        .sequence
        .steps
        .first()
        .map(|step| step.step_id.as_str());
    realize_excitations(model, step_id, sample)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RealizationError {
    message: String,
}

impl RealizationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RealizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl std::error::Error for RealizationError {}

pub(crate) fn realize_excitations(
    model: &AnalysisModel,
    step_id: Option<&str>,
    sample: ExcitationSamplePoint,
) -> Result<RealizedExcitations, RealizationError> {
    let signals = model
        .signals
        .iter()
        .map(|signal| (signal.signal_id().0.as_str(), signal))
        .collect::<HashMap<_, _>>();
    let active_cases = active_case_ids(model, step_id);
    let mut entries = Vec::new();
    for action in &model.excitations {
        let excitation = action.excitation();
        if !active_cases.contains(excitation.case_id.0.as_str())
            || (!excitation.active_step_ids.is_empty()
                && step_id.is_none_or(|step| {
                    !excitation
                        .active_step_ids
                        .iter()
                        .any(|active| active == step)
                }))
        {
            continue;
        }
        let signal = signals
            .get(excitation.signal_id.0.as_str())
            .ok_or_else(|| {
                RealizationError::new(format!(
                    "excitation {} references missing signal {}",
                    excitation.excitation_id.0, excitation.signal_id.0
                ))
            })?;
        let target = realize_target(&excitation.target);
        let kind = realize_kind(action, signal, sample)?;
        entries.push(RealizedExcitation {
            excitation_id: excitation.excitation_id.clone(),
            case_id: excitation.case_id.clone(),
            target,
            kind,
        });
    }
    Ok(RealizedExcitations::new(coalesce_wrenches(entries)))
}

fn active_case_ids<'a>(model: &'a AnalysisModel, step_id: Option<&str>) -> HashSet<&'a str> {
    model
        .excitation_cases
        .iter()
        .filter(|case| match &case.activation {
            ExcitationCaseActivation::Always => true,
            ExcitationCaseActivation::Steps(steps) => {
                step_id.is_some_and(|step| steps.iter().any(|active| active == step))
            }
            ExcitationCaseActivation::MutuallyExclusiveGroup { .. } => false,
        })
        .map(|case| case.case_id.0.as_str())
        .collect()
}

fn realize_target(target: &ExcitationTarget) -> RealizedExcitationTarget {
    match target {
        ExcitationTarget::Region(target) => {
            RealizedExcitationTarget::Region(target.region_id.clone())
        }
        ExcitationTarget::ReferencePoint(point) => RealizedExcitationTarget::ReferencePoint {
            reference_point_id: point.reference_point_id.clone(),
            position: point.position,
        },
        ExcitationTarget::Port { port_id } => RealizedExcitationTarget::Port(port_id.clone()),
        ExcitationTarget::Interface { interface_id } => {
            RealizedExcitationTarget::Interface(interface_id.clone())
        }
        ExcitationTarget::WholeModel => RealizedExcitationTarget::WholeModel,
    }
}

macro_rules! scalar_action {
    ($signal:expr, $variant:path, $field:ident, $sample:expr) => {
        match $signal {
            $variant(signal) => {
                sample_signal(signal, $sample, |value| value.$field).and_then(expect_scalar)
            }
            _ => Err(quantity_mismatch()),
        }
    };
}

macro_rules! vector_action {
    ($signal:expr, $variant:path, $field:ident, $sample:expr) => {
        match $signal {
            $variant(signal) => {
                sample_signal(signal, $sample, |value| value.$field).and_then(expect_vector3)
            }
            _ => Err(quantity_mismatch()),
        }
    };
}

macro_rules! directional_vector_action {
    ($action:expr, $signal:expr, $variant:path, $field:ident, $sample:expr) => {
        match $signal {
            $variant(signal) => {
                sample_signal(signal, $sample, |value| value.$field).and_then(|value| {
                    expect_vector3_or_direction(value, $action.excitation().direction)
                })
            }
            _ => Err(quantity_mismatch()),
        }
    };
}

fn realize_kind(
    action: &ExcitationAction,
    signal: &PhysicalSignal,
    sample: ExcitationSamplePoint,
) -> Result<RealizedExcitationKind, RealizationError> {
    Ok(match action {
        ExcitationAction::Force(_) => RealizedExcitationKind::Force(directional_vector_action!(
            action,
            signal,
            PhysicalSignal::Force,
            newtons,
            sample
        )?),
        ExcitationAction::Moment(_) => RealizedExcitationKind::Moment(directional_vector_action!(
            action,
            signal,
            PhysicalSignal::Moment,
            newton_meters,
            sample
        )?),
        ExcitationAction::Traction(_) => RealizedExcitationKind::Traction(vector_action!(
            signal,
            PhysicalSignal::Traction,
            pascals,
            sample
        )?),
        ExcitationAction::Pressure(_) => RealizedExcitationKind::Pressure(scalar_action!(
            signal,
            PhysicalSignal::Pressure,
            pascals,
            sample
        )?),
        ExcitationAction::BodyForce(_) => RealizedExcitationKind::BodyForce(vector_action!(
            signal,
            PhysicalSignal::BodyForceDensity,
            newtons_per_cubic_meter,
            sample
        )?),
        ExcitationAction::BaseDisplacement(_) => {
            RealizedExcitationKind::BaseDisplacement(directional_vector_action!(
                action,
                signal,
                PhysicalSignal::Displacement,
                meters,
                sample
            )?)
        }
        ExcitationAction::BaseVelocity(_) => {
            RealizedExcitationKind::BaseVelocity(directional_vector_action!(
                action,
                signal,
                PhysicalSignal::Velocity,
                meters_per_second,
                sample
            )?)
        }
        ExcitationAction::BaseAcceleration(_) => {
            RealizedExcitationKind::BaseAcceleration(directional_vector_action!(
                action,
                signal,
                PhysicalSignal::Acceleration,
                meters_per_second_squared,
                sample
            )?)
        }
        ExcitationAction::HeatGeneration(_) => {
            RealizedExcitationKind::HeatGeneration(scalar_action!(
                signal,
                PhysicalSignal::HeatGenerationDensity,
                watts_per_cubic_meter,
                sample
            )?)
        }
        ExcitationAction::HeatFlux(_) => RealizedExcitationKind::HeatFlux(scalar_action!(
            signal,
            PhysicalSignal::HeatFlux,
            watts_per_square_meter,
            sample
        )?),
        ExcitationAction::TemperatureField(_) => RealizedExcitationKind::TemperatureField(
            scalar_action!(signal, PhysicalSignal::Temperature, kelvin, sample)?,
        ),
        ExcitationAction::AmbientTemperature(_) => RealizedExcitationKind::AmbientTemperature(
            scalar_action!(signal, PhysicalSignal::Temperature, kelvin, sample)?,
        ),
        ExcitationAction::TerminalVoltage(_) => RealizedExcitationKind::TerminalVoltage(
            scalar_action!(signal, PhysicalSignal::ElectricPotential, volts, sample)?,
        ),
        ExcitationAction::ElectricCurrent(_) => RealizedExcitationKind::ElectricCurrent(
            scalar_action!(signal, PhysicalSignal::ElectricCurrent, amperes, sample)?,
        ),
        ExcitationAction::ElectricCharge(_) => RealizedExcitationKind::ElectricCharge(
            scalar_action!(signal, PhysicalSignal::ElectricCharge, coulombs, sample)?,
        ),
        ExcitationAction::CurrentDensity(_) => {
            RealizedExcitationKind::CurrentDensity(vector_action!(
                signal,
                PhysicalSignal::ElectricCurrentDensity,
                amperes_per_square_meter,
                sample
            )?)
        }
        ExcitationAction::MagneticVectorPotential(_) => {
            RealizedExcitationKind::MagneticVectorPotential(vector_action!(
                signal,
                PhysicalSignal::MagneticVectorPotential,
                webers_per_meter,
                sample
            )?)
        }
        ExcitationAction::AcousticPressure(_) => RealizedExcitationKind::AcousticPressure(
            scalar_action!(signal, PhysicalSignal::Pressure, pascals, sample)?,
        ),
        ExcitationAction::NormalVelocity(_) => RealizedExcitationKind::NormalVelocity(
            scalar_action!(signal, PhysicalSignal::Velocity, meters_per_second, sample)?,
        ),
        ExcitationAction::InletVelocity(_) => {
            RealizedExcitationKind::InletVelocity(directional_vector_action!(
                action,
                signal,
                PhysicalSignal::Velocity,
                meters_per_second,
                sample
            )?)
        }
        ExcitationAction::MassFlow(_) => RealizedExcitationKind::MassFlow(scalar_action!(
            signal,
            PhysicalSignal::MassFlowRate,
            kilograms_per_second,
            sample
        )?),
        ExcitationAction::WallMotion(_) => {
            RealizedExcitationKind::WallMotion(directional_vector_action!(
                action,
                signal,
                PhysicalSignal::Velocity,
                meters_per_second,
                sample
            )?)
        }
    })
}

#[derive(Debug, Clone, PartialEq)]
enum SampledValue {
    Scalar(ComplexScalar),
    Vector3(ComplexVector3),
}

fn sample_signal<Q>(
    signal: &Signal<Q>,
    sample: ExcitationSamplePoint,
    component: impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    match &signal.representation {
        SignalRepresentation::Constant(value) => Ok(real_value(value, &component)),
        SignalRepresentation::Tabulated(table) => {
            let abscissa = sample_abscissa(sample, &table.axis)?;
            sample_table(
                axis_values(&table.axis),
                &table.values,
                abscissa,
                table.interpolation,
                table.extrapolation,
                component,
            )
        }
        SignalRepresentation::Waveform(_) => Err(RealizationError::new(
            "waveform signals must be materialized before FEA excitation realization",
        )),
        SignalRepresentation::Artifact(_) | SignalRepresentation::Generator(_) => Err(
            RealizationError::new(
                "artifact and generator signals must be materialized before FEA excitation realization",
            ),
        ),
    }
}

fn axis_values(axis: &SignalAxis) -> &[f64] {
    match axis {
        SignalAxis::TimeSeconds(values) | SignalAxis::FrequencyHertz(values) => values,
        SignalAxis::StudyParameter {
            canonical_values, ..
        } => canonical_values,
    }
}

fn sample_abscissa(
    sample: ExcitationSamplePoint,
    axis: &SignalAxis,
) -> Result<f64, RealizationError> {
    match (sample, axis) {
        (ExcitationSamplePoint::FrequencyHertz(value), SignalAxis::FrequencyHertz(_)) => Ok(value),
        (ExcitationSamplePoint::Initial, axis) => axis_values(axis)
            .first()
            .copied()
            .ok_or_else(|| RealizationError::new("tabulated signal has no samples")),
        _ => Err(RealizationError::new(
            "excitation sample point does not match the signal domain",
        )),
    }
}

fn real_value<Q>(value: &SignalValue<Q>, component: &impl Fn(&Q) -> f64) -> SampledValue {
    match value {
        SignalValue::Scalar(value) => SampledValue::Scalar(ComplexScalar::real(component(value))),
        SignalValue::Vector3(values) => SampledValue::Vector3(ComplexVector3::real([
            component(&values[0]),
            component(&values[1]),
            component(&values[2]),
        ])),
        SignalValue::Tensor { .. } => unreachable!("FEA actions reject tensor signal values"),
    }
}

fn sample_table<Q>(
    axis: &[f64],
    values: &[SignalOrdinate<Q>],
    x: f64,
    interpolation: SignalInterpolation,
    extrapolation: SignalExtrapolation,
    component: impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    let (left, right, weight) = sample_interval(axis, x, interpolation, extrapolation)?;
    let left = ordinate_value(&values[left], &component)?;
    let right = ordinate_value(&values[right], &component)?;
    interpolate_value(left, right, weight)
}

fn sample_interval(
    axis: &[f64],
    x: f64,
    interpolation: SignalInterpolation,
    extrapolation: SignalExtrapolation,
) -> Result<(usize, usize, f64), RealizationError> {
    if axis.is_empty() {
        return Err(RealizationError::new("tabulated signal has no samples"));
    }
    if axis.len() == 1 {
        return if x == axis[0] || extrapolation == SignalExtrapolation::HoldEndpoint {
            Ok((0, 0, 0.0))
        } else {
            Err(RealizationError::new("sample lies outside signal domain"))
        };
    }
    if x < axis[0] || x > axis[axis.len() - 1] {
        return match extrapolation {
            SignalExtrapolation::Reject => {
                Err(RealizationError::new("sample lies outside signal domain"))
            }
            SignalExtrapolation::HoldEndpoint => {
                let index = usize::from(x > axis[axis.len() - 1]) * (axis.len() - 1);
                Ok((index, index, 0.0))
            }
            SignalExtrapolation::Zero => Ok((0, 0, f64::NAN)),
        };
    }
    let right = axis.partition_point(|value| *value < x).min(axis.len() - 1);
    let left = right.saturating_sub(1);
    let weight = match interpolation {
        SignalInterpolation::StepLeft => 0.0,
        SignalInterpolation::StepRight => 1.0,
        SignalInterpolation::Linear => {
            if left == right {
                0.0
            } else {
                (x - axis[left]) / (axis[right] - axis[left])
            }
        }
        SignalInterpolation::CubicHermite => {
            return Err(RealizationError::new(
                "cubic Hermite excitation signals require derivative materialization",
            ))
        }
    };
    Ok((left, right, weight))
}

fn ordinate_value<Q>(
    ordinate: &SignalOrdinate<Q>,
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    Ok(match ordinate {
        SignalOrdinate::Real(value) => real_value(value, component),
        SignalOrdinate::Complex { real, imaginary } => complex_value(real, imaginary, component)?,
    })
}

fn complex_value<Q>(
    real: &SignalValue<Q>,
    imaginary: &SignalValue<Q>,
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    Ok(match (real, imaginary) {
        (SignalValue::Scalar(real), SignalValue::Scalar(imaginary)) => {
            SampledValue::Scalar(ComplexScalar {
                real: component(real),
                imaginary: component(imaginary),
            })
        }
        (SignalValue::Vector3(real), SignalValue::Vector3(imaginary)) => {
            SampledValue::Vector3(ComplexVector3 {
                real: [
                    component(&real[0]),
                    component(&real[1]),
                    component(&real[2]),
                ],
                imaginary: [
                    component(&imaginary[0]),
                    component(&imaginary[1]),
                    component(&imaginary[2]),
                ],
            })
        }
        _ => {
            return Err(RealizationError::new(
                "complex signal shape is not scalar or vector3",
            ))
        }
    })
}

fn interpolate_value(
    left: SampledValue,
    right: SampledValue,
    weight: f64,
) -> Result<SampledValue, RealizationError> {
    if weight.is_nan() {
        return Ok(match left {
            SampledValue::Scalar(_) => SampledValue::Scalar(ComplexScalar::real(0.0)),
            SampledValue::Vector3(_) => SampledValue::Vector3(ComplexVector3::real([0.0; 3])),
        });
    }
    let blend = |a: f64, b: f64| a + weight * (b - a);
    Ok(match (left, right) {
        (SampledValue::Scalar(left), SampledValue::Scalar(right)) => {
            SampledValue::Scalar(ComplexScalar {
                real: blend(left.real, right.real),
                imaginary: blend(left.imaginary, right.imaginary),
            })
        }
        (SampledValue::Vector3(left), SampledValue::Vector3(right)) => {
            let mut real = [0.0; 3];
            let mut imaginary = [0.0; 3];
            for index in 0..3 {
                real[index] = blend(left.real[index], right.real[index]);
                imaginary[index] = blend(left.imaginary[index], right.imaginary[index]);
            }
            SampledValue::Vector3(ComplexVector3 { real, imaginary })
        }
        _ => {
            return Err(RealizationError::new(
                "signal shape changes between samples",
            ))
        }
    })
}

fn expect_scalar(value: SampledValue) -> Result<ComplexScalar, RealizationError> {
    match value {
        SampledValue::Scalar(value) => Ok(value),
        SampledValue::Vector3(_) => Err(RealizationError::new("excitation requires scalar signal")),
    }
}

fn expect_vector3(value: SampledValue) -> Result<ComplexVector3, RealizationError> {
    match value {
        SampledValue::Vector3(value) => Ok(value),
        SampledValue::Scalar(_) => Err(RealizationError::new("excitation requires vector3 signal")),
    }
}

fn expect_vector3_or_direction(
    value: SampledValue,
    direction: Option<runmat_analysis_core::Direction>,
) -> Result<ComplexVector3, RealizationError> {
    match value {
        SampledValue::Vector3(value) => Ok(value),
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
    }
}

fn quantity_mismatch() -> RealizationError {
    RealizationError::new("excitation action and physical signal quantity do not match")
}

fn coalesce_wrenches(entries: Vec<RealizedExcitation>) -> Vec<RealizedExcitation> {
    let mut consumed = vec![false; entries.len()];
    let mut output = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        if consumed[index] {
            continue;
        }
        let RealizedExcitationKind::Force(force) = entry.kind else {
            output.push(entry.clone());
            continue;
        };
        let RealizedExcitationTarget::ReferencePoint { position, .. } = &entry.target else {
            output.push(entry.clone());
            continue;
        };
        let matching = entries
            .iter()
            .enumerate()
            .find(|(candidate_index, candidate)| {
                *candidate_index != index
                    && !consumed[*candidate_index]
                    && candidate.case_id == entry.case_id
                    && candidate.target == entry.target
                    && matches!(candidate.kind, RealizedExcitationKind::Moment(_))
            });
        if let Some((moment_index, moment_entry)) = matching {
            let RealizedExcitationKind::Moment(moment) = moment_entry.kind else {
                unreachable!();
            };
            consumed[moment_index] = true;
            output.push(RealizedExcitation {
                excitation_id: entry.excitation_id.clone(),
                case_id: entry.case_id.clone(),
                target: entry.target.clone(),
                kind: RealizedExcitationKind::Wrench {
                    force,
                    moment,
                    point: *position,
                },
            });
        } else {
            output.push(entry.clone());
        }
    }
    output
}
