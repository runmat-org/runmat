use runmat_analysis_core::{
    AnalysisModel, CfdDomain, ExcitationAction, ExcitationTarget, PhysicalSignal, SignalId,
    SignalOrdinate, SignalRepresentation, SignalValue, ThermoMechanicalDomain,
};

pub(crate) fn cfd_velocity_scales(model: &AnalysisModel, _domain: &CfdDomain) -> Vec<f64> {
    normalized_scales(excitation_signal_values(
        model,
        |action| matches!(action, ExcitationAction::InletVelocity(_)),
        |signal| match signal {
            PhysicalSignal::Velocity(signal) => {
                scalar_or_vector_magnitudes(&signal.representation, |value| value.meters_per_second)
            }
            _ => None,
        },
    ))
}

pub(crate) fn nominal_inlet_velocity_m_per_s(model: &AnalysisModel) -> Option<f64> {
    maximum_absolute(excitation_signal_values(
        model,
        |action| matches!(action, ExcitationAction::InletVelocity(_)),
        |signal| match signal {
            PhysicalSignal::Velocity(signal) => {
                scalar_or_vector_magnitudes(&signal.representation, |value| value.meters_per_second)
            }
            _ => None,
        },
    ))
}

pub(crate) fn applied_temperature_delta_k(
    model: &AnalysisModel,
    domain: &ThermoMechanicalDomain,
) -> Option<f64> {
    maximum_absolute(
        temperature_field_values(model)
            .into_iter()
            .map(|value| value - domain.reference_temperature.kelvin)
            .collect(),
    )
}

pub(crate) fn regional_temperature_deltas_k(
    model: &AnalysisModel,
    domain: &ThermoMechanicalDomain,
) -> Vec<(String, f64)> {
    model
        .excitations
        .iter()
        .filter_map(|action| {
            let ExcitationAction::TemperatureField(excitation) = action else {
                return None;
            };
            let ExcitationTarget::Region(target) = &excitation.target else {
                return None;
            };
            let PhysicalSignal::Temperature(signal) = signal(model, &excitation.signal_id)? else {
                return None;
            };
            maximum_absolute(
                scalar_values(&signal.representation, |value| value.kelvin)?
                    .into_iter()
                    .map(|value| value - domain.reference_temperature.kelvin)
                    .collect(),
            )
            .map(|delta| (target.region_id.clone(), delta))
        })
        .collect()
}

pub(crate) fn applied_voltage_v(model: &AnalysisModel) -> Option<f64> {
    maximum_absolute(terminal_voltage_values(model))
}

fn temperature_field_values(model: &AnalysisModel) -> Vec<f64> {
    excitation_signal_values(
        model,
        |action| {
            matches!(action, ExcitationAction::TemperatureField(_))
                && matches!(action.excitation().target, ExcitationTarget::WholeModel)
        },
        |signal| match signal {
            PhysicalSignal::Temperature(signal) => {
                scalar_values(&signal.representation, |value| value.kelvin)
            }
            _ => None,
        },
    )
}

fn terminal_voltage_values(model: &AnalysisModel) -> Vec<f64> {
    excitation_signal_values(
        model,
        |action| matches!(action, ExcitationAction::TerminalVoltage(_)),
        |signal| match signal {
            PhysicalSignal::ElectricPotential(signal) => {
                scalar_values(&signal.representation, |value| value.volts)
            }
            _ => None,
        },
    )
}

fn excitation_signal_values(
    model: &AnalysisModel,
    accepts: impl Fn(&ExcitationAction) -> bool,
    convert: impl Fn(&PhysicalSignal) -> Option<Vec<f64>>,
) -> Vec<f64> {
    model
        .excitations
        .iter()
        .filter(|action| accepts(action))
        .filter_map(|action| signal(model, &action.excitation().signal_id))
        .filter_map(&convert)
        .flatten()
        .collect()
}

fn signal<'a>(model: &'a AnalysisModel, id: &SignalId) -> Option<&'a PhysicalSignal> {
    model.signals.iter().find(|signal| signal.signal_id() == id)
}

fn normalized_scales(values: Vec<f64>) -> Vec<f64> {
    let Some(nominal) = maximum_absolute(values.clone()) else {
        return Vec::new();
    };
    if nominal <= f64::EPSILON {
        return Vec::new();
    }
    values.into_iter().map(|value| value / nominal).collect()
}

fn maximum_absolute(values: Vec<f64>) -> Option<f64> {
    values
        .into_iter()
        .filter(|value| value.is_finite())
        .map(f64::abs)
        .reduce(f64::max)
}

fn scalar_values<Q>(
    representation: &SignalRepresentation<Q>,
    component: impl Fn(&Q) -> f64,
) -> Option<Vec<f64>> {
    values(representation)?
        .map(|value| scalar_value(value, &component))
        .collect()
}

fn scalar_or_vector_magnitudes<Q>(
    representation: &SignalRepresentation<Q>,
    component: impl Fn(&Q) -> f64,
) -> Option<Vec<f64>> {
    values(representation)?
        .map(|value| match value {
            SignalValue::Scalar(value) => Some(component(value)),
            SignalValue::Vector3(values) => Some(
                values
                    .iter()
                    .map(|value| component(value).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            ),
            SignalValue::Tensor { .. } => None,
        })
        .collect()
}

fn values<'a, Q>(
    representation: &'a SignalRepresentation<Q>,
) -> Option<Box<dyn Iterator<Item = &'a SignalValue<Q>> + 'a>> {
    match representation {
        SignalRepresentation::Constant(value) => Some(Box::new(std::iter::once(value))),
        SignalRepresentation::Tabulated(table) => Some(Box::new(table.values.iter().filter_map(
            |value| match value {
                SignalOrdinate::Real(value) => Some(value),
                SignalOrdinate::Complex { .. } => None,
            },
        ))),
        SignalRepresentation::Waveform(waveform) => {
            Some(Box::new(std::iter::once(&waveform.amplitude)))
        }
        SignalRepresentation::Artifact(_) | SignalRepresentation::Generator(_) => None,
    }
}

fn scalar_value<Q>(value: &SignalValue<Q>, component: &impl Fn(&Q) -> f64) -> Option<f64> {
    match value {
        SignalValue::Scalar(value) => Some(component(value)),
        SignalValue::Vector3(_) | SignalValue::Tensor { .. } => None,
    }
}

pub(crate) fn pressure_sources(model: &AnalysisModel) -> Vec<(String, f64)> {
    model
        .excitations
        .iter()
        .filter_map(|action| match action {
            ExcitationAction::Pressure(value) | ExcitationAction::AcousticPressure(value) => {
                let target = target_id(&value.target);
                let signal = signal(model, &value.signal_id)?;
                let PhysicalSignal::Pressure(signal) = signal else {
                    return None;
                };
                scalar_values(&signal.representation, |value| value.pascals)
                    .and_then(|values| values.first().copied())
                    .map(|magnitude| (target, magnitude))
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn has_nonzero_electromagnetic_source(model: &AnalysisModel) -> bool {
    model.excitations.iter().any(|action| match action {
        ExcitationAction::ElectricCurrent(value) => signal(model, &value.signal_id)
            .and_then(|signal| match signal {
                PhysicalSignal::ElectricCurrent(signal) => Some(representation_has_nonzero(
                    &signal.representation,
                    |value| value.amperes,
                )),
                _ => None,
            })
            .unwrap_or(false),
        ExcitationAction::CurrentDensity(value) => signal(model, &value.signal_id)
            .and_then(|signal| match signal {
                PhysicalSignal::ElectricCurrentDensity(signal) => Some(representation_has_nonzero(
                    &signal.representation,
                    |value| value.amperes_per_square_meter,
                )),
                _ => None,
            })
            .unwrap_or(false),
        _ => false,
    })
}

fn target_id(target: &ExcitationTarget) -> String {
    match target {
        ExcitationTarget::Region(target) => target.region_id.clone(),
        ExcitationTarget::ReferencePoint(value) => value.reference_point_id.clone(),
        ExcitationTarget::Port { port_id } => port_id.clone(),
        ExcitationTarget::Interface { interface_id } => interface_id.clone(),
        ExcitationTarget::WholeModel => "whole_model".to_string(),
    }
}

fn representation_has_nonzero<Q>(
    representation: &SignalRepresentation<Q>,
    component: impl Fn(&Q) -> f64,
) -> bool {
    match representation {
        SignalRepresentation::Constant(value) => value_has_nonzero(value, &component),
        SignalRepresentation::Tabulated(table) => {
            table.values.iter().any(|ordinate| match ordinate {
                SignalOrdinate::Real(value) => value_has_nonzero(value, &component),
                SignalOrdinate::Complex { real, imaginary } => {
                    value_has_nonzero(real, &component) || value_has_nonzero(imaginary, &component)
                }
            })
        }
        SignalRepresentation::Waveform(waveform) => {
            value_has_nonzero(&waveform.amplitude, &component)
        }
        SignalRepresentation::Artifact(_) | SignalRepresentation::Generator(_) => false,
    }
}

fn value_has_nonzero<Q>(value: &SignalValue<Q>, component: &impl Fn(&Q) -> f64) -> bool {
    match value {
        SignalValue::Scalar(value) => component(value).abs() > 0.0,
        SignalValue::Vector3(values) => values.iter().any(|value| component(value).abs() > 0.0),
        SignalValue::Tensor { values, .. } => {
            values.iter().any(|value| component(value).abs() > 0.0)
        }
    }
}
