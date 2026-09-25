use runmat_analysis_core::{
    AnalysisModel, Direction, HeatFlux, PhysicalSignal, PrescribedValue, ProcedureStep, Signal,
    SignalExtrapolation, SignalInterpolation, SignalKnot, SignalKnotValue, SignalOrdinate,
    SignalRepresentation, SignalSampleApproach, SignalValue, TabulatedSignalDomain, Temperature,
};

use super::{
    model::{ComplexScalar, ComplexVector3},
    realize::RealizationError,
    ProjectionSample,
};

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

impl From<ExcitationSamplePoint<'_>> for ProjectionSample {
    fn from(sample: ExcitationSamplePoint<'_>) -> Self {
        match sample {
            ExcitationSamplePoint::Static => Self::Static,
            ExcitationSamplePoint::TimeSeconds { value, approach } => {
                Self::TimeSeconds { value, approach }
            }
            ExcitationSamplePoint::FrequencyHertz { value, approach } => {
                Self::FrequencyHertz { value, approach }
            }
            ExcitationSamplePoint::StudyParameter {
                parameter_id,
                canonical_value,
                approach,
            } => Self::StudyParameter {
                parameter_id: parameter_id.to_string(),
                canonical_value,
                approach,
            },
        }
    }
}

pub(crate) fn time_breakpoints(
    model: &AnalysisModel,
    step: &ProcedureStep,
    start_time_s: f64,
    end_time_s: f64,
) -> Result<Vec<f64>, RealizationError> {
    let active_cases = super::activation::active_case_ids(model, Some(step.step_id.as_str()))?;
    let active_signal_ids = model
        .excitations
        .iter()
        .map(|action| action.excitation())
        .filter(|excitation| {
            model.excitation_cases.iter().any(|case| {
                active_cases.contains(case.case_id.0.as_str())
                    && case.action_ids.contains(&excitation.excitation_id)
            }) && step.object_is_active(&runmat_analysis_core::StepObjectRef::Excitation(
                excitation.excitation_id.0.clone(),
            )) && (excitation.active_step_ids.is_empty()
                || excitation
                    .active_step_ids
                    .iter()
                    .any(|active| active == step.step_id.as_str()))
        })
        .map(|excitation| excitation.signal_id.0.as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut active_signal_ids = active_signal_ids;
    collect_constraint_signal_ids(model, step, &mut active_signal_ids);
    let mut events = Vec::new();
    macro_rules! collect {
        ($signal:expr) => {
            extend_time_jumps(&mut events, $signal, start_time_s, end_time_s)
        };
    }
    for signal in &model.signals {
        if !active_signal_ids.contains(signal.signal_id().0.as_str()) {
            continue;
        }
        match signal {
            PhysicalSignal::Displacement(signal) => collect!(signal),
            PhysicalSignal::Rotation(signal) => collect!(signal),
            PhysicalSignal::Velocity(signal) => collect!(signal),
            PhysicalSignal::Acceleration(signal) => collect!(signal),
            PhysicalSignal::AngularVelocity(signal) => collect!(signal),
            PhysicalSignal::AngularAcceleration(signal) => collect!(signal),
            PhysicalSignal::Force(signal) => collect!(signal),
            PhysicalSignal::Moment(signal) => collect!(signal),
            PhysicalSignal::Traction(signal) => collect!(signal),
            PhysicalSignal::Pressure(signal) => collect!(signal),
            PhysicalSignal::BodyForceDensity(signal) => collect!(signal),
            PhysicalSignal::Temperature(signal) => collect!(signal),
            PhysicalSignal::TemperatureDifference(signal) => collect!(signal),
            PhysicalSignal::HeatFlux(signal) => collect!(signal),
            PhysicalSignal::HeatGenerationDensity(signal) => collect!(signal),
            PhysicalSignal::ElectricPotential(signal) => collect!(signal),
            PhysicalSignal::ElectricCurrent(signal) => collect!(signal),
            PhysicalSignal::ElectricCharge(signal) => collect!(signal),
            PhysicalSignal::ElectricCurrentDensity(signal) => collect!(signal),
            PhysicalSignal::ElectricChargeDensity(signal) => collect!(signal),
            PhysicalSignal::MagneticVectorPotential(signal) => collect!(signal),
            PhysicalSignal::MassFlowRate(signal) => collect!(signal),
        }
    }
    events.sort_by(f64::total_cmp);
    events.dedup_by(|left, right| left.to_bits() == right.to_bits());
    Ok(events)
}

fn collect_constraint_signal_ids<'a>(
    model: &'a AnalysisModel,
    step: &ProcedureStep,
    ids: &mut std::collections::HashSet<&'a str>,
) {
    use runmat_analysis_core::{
        AnalysisConstraint, CoupledPhysicsFamily, PhysicsFamily, StructuralConstraintKind,
        ThermalConstraintKind,
    };
    let structural = matches!(
        step.family,
        PhysicsFamily::Structural | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
    );
    let thermal = matches!(
        step.family,
        PhysicsFamily::Thermal
            | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
            | PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer)
            | PhysicsFamily::Coupled(CoupledPhysicsFamily::Electrothermal)
    );
    for constraint in &model.constraints {
        if !step.object_is_active(&runmat_analysis_core::StepObjectRef::Constraint(
            constraint.constraint_id().to_string(),
        )) {
            continue;
        }
        match constraint {
            AnalysisConstraint::Structural(constraint) if structural => match &constraint.kind {
                StructuralConstraintKind::Prescribed(value) => {
                    for translation in &value.translations {
                        insert_prescribed_id(ids, &translation.value);
                    }
                    for rotation in &value.rotations {
                        insert_prescribed_id(ids, &rotation.value);
                    }
                }
                StructuralConstraintKind::NormalDisplacement { value, .. } => {
                    insert_prescribed_id(ids, value)
                }
                StructuralConstraintKind::PrescribedMotion(motion) => {
                    for translation in &motion.translations {
                        let runmat_analysis_core::StructuralKinematicHistory::AccelerationDriven {
                            acceleration,
                            ..
                        } = &translation.history;
                        ids.insert(acceleration.0 .0.as_str());
                    }
                }
                _ => {}
            },
            AnalysisConstraint::Thermal(constraint) if thermal => match &constraint.kind {
                ThermalConstraintKind::Prescribed { value, .. } => insert_prescribed_id(ids, value),
                ThermalConstraintKind::AffineJump {
                    temperature_jump, ..
                } => insert_prescribed_id(ids, temperature_jump),
                _ => {}
            },
            _ => {}
        }
    }
}

fn insert_prescribed_id<'a, Q>(
    ids: &mut std::collections::HashSet<&'a str>,
    value: &'a runmat_analysis_core::PrescribedValue<Q>,
) {
    if let runmat_analysis_core::PrescribedValue::Signal(signal_id) = value {
        ids.insert(signal_id.0.as_str());
    }
}

fn extend_time_jumps<Q>(events: &mut Vec<f64>, signal: &Signal<Q>, start: f64, end: f64) {
    match &signal.representation {
        SignalRepresentation::Tabulated(table)
            if table.domain == TabulatedSignalDomain::TimeSeconds =>
        {
            events.extend(table.knots.iter().filter_map(|knot| {
                (knot.abscissa > start
                    && knot.abscissa <= end
                    && matches!(knot.value, SignalKnotValue::Jump { .. }))
                .then_some(knot.abscissa)
            }));
        }
        SignalRepresentation::Waveform(waveform) => {
            events.extend(super::waveform::breakpoints(waveform, start, end));
        }
        _ => {}
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SampledValue {
    Scalar(ComplexScalar),
    Vector3(ComplexVector3),
}

pub(crate) fn sample_signal<Q>(
    signal: &Signal<Q>,
    sample: ExcitationSamplePoint<'_>,
    component: impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    match &signal.representation {
        SignalRepresentation::Constant(value) => Ok(real_value(value, &component)),
        SignalRepresentation::Tabulated(table) => {
            if matches!(sample, ExcitationSamplePoint::Static) {
                return sample_static_table(&table.knots, &component);
            }
            let (abscissa, approach) = sample_abscissa(sample, &table.domain)?;
            sample_table(
                &table.knots,
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

/// A static step has no abscissa. A table is unambiguous only when its complete sampled content
/// is constant; choosing an arbitrary knot would silently change a generated history.
fn sample_static_table<Q>(
    knots: &[SignalKnot<Q>],
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
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
    let mut sampled = None;
    for knot in knots {
        let SignalKnotValue::Continuous(value) = &knot.value else {
            return Err(RealizationError::new(
                "static excitation cannot select a jump in a tabulated signal",
            ));
        };
        let value = ordinate_value(value, component)?;
        if sampled.as_ref().is_some_and(|previous| previous != &value) {
            return Err(RealizationError::new(
                "static excitation requires constant tabulated values",
            ));
        }
        sampled = Some(value);
    }
    Ok(sampled.expect("nonempty validated table has a sample"))
}

pub(crate) fn sample_prescribed_temperature(
    value: &PrescribedValue<Temperature>,
    signals: &[PhysicalSignal],
    sample: ExcitationSamplePoint<'_>,
) -> Result<ComplexScalar, RealizationError> {
    sample_prescribed_scalar(
        value,
        signals,
        sample,
        |signal| match signal {
            PhysicalSignal::Temperature(signal) => Some(signal),
            _ => None,
        },
        |temperature| temperature.kelvin,
    )
}

pub(crate) fn sample_prescribed_heat_flux(
    value: &PrescribedValue<HeatFlux>,
    signals: &[PhysicalSignal],
    sample: ExcitationSamplePoint<'_>,
) -> Result<ComplexScalar, RealizationError> {
    sample_prescribed_scalar(
        value,
        signals,
        sample,
        |signal| match signal {
            PhysicalSignal::HeatFlux(signal) => Some(signal),
            _ => None,
        },
        |heat_flux| heat_flux.watts_per_square_meter,
    )
}

fn sample_prescribed_scalar<'a, Q>(
    value: &'a PrescribedValue<Q>,
    signals: &'a [PhysicalSignal],
    sample: ExcitationSamplePoint<'_>,
    typed_signal: impl Fn(&'a PhysicalSignal) -> Option<&'a Signal<Q>>,
    component: impl Fn(&Q) -> f64,
) -> Result<ComplexScalar, RealizationError> {
    match value {
        PrescribedValue::Constant(value) => Ok(ComplexScalar::real(component(value))),
        PrescribedValue::Signal(signal_id) => {
            let signal = signals
                .iter()
                .find(|candidate| candidate.signal_id() == signal_id)
                .and_then(typed_signal)
                .ok_or_else(|| {
                    RealizationError::new(format!(
                        "prescribed value references missing or incompatible signal {}",
                        signal_id.0
                    ))
                })?;
            expect_scalar(sample_signal(signal, sample, component)?)
        }
    }
}

pub(crate) fn expect_scalar(value: SampledValue) -> Result<ComplexScalar, RealizationError> {
    match value {
        SampledValue::Scalar(value) => Ok(value),
        SampledValue::Vector3(_) => Err(RealizationError::new("excitation requires scalar signal")),
    }
}

pub(crate) fn expect_vector3(value: SampledValue) -> Result<ComplexVector3, RealizationError> {
    match value {
        SampledValue::Vector3(value) => Ok(value),
        SampledValue::Scalar(_) => Err(RealizationError::new("excitation requires vector3 signal")),
    }
}

pub(super) fn expect_vector3_or_direction(
    value: SampledValue,
    direction: Option<Direction>,
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

fn sample_abscissa(
    sample: ExcitationSamplePoint<'_>,
    domain: &TabulatedSignalDomain,
) -> Result<(f64, SignalSampleApproach), RealizationError> {
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
            },
        ) if parameter_id == axis_parameter => (canonical_value, approach),
        _ => {
            return Err(RealizationError::new(
                "excitation sample point does not match the signal domain",
            ));
        }
    };
    if value.is_finite() {
        Ok((value, approach))
    } else {
        Err(RealizationError::new(
            "excitation sample point must be finite",
        ))
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
    knots: &[SignalKnot<Q>],
    x: f64,
    approach: SignalSampleApproach,
    interpolation: SignalInterpolation,
    extrapolation: SignalExtrapolation,
    component: impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
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

fn zero_value<Q>(
    ordinate: &SignalOrdinate<Q>,
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    Ok(match ordinate_value(ordinate, component)? {
        SampledValue::Scalar(_) => SampledValue::Scalar(ComplexScalar::real(0.0)),
        SampledValue::Vector3(_) => SampledValue::Vector3(ComplexVector3::real([0.0; 3])),
    })
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
            ));
        }
    })
}

fn interpolate_value(
    left: SampledValue,
    right: SampledValue,
    weight: f64,
) -> Result<SampledValue, RealizationError> {
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
