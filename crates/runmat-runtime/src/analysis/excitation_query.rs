use runmat_analysis_core::{
    AnalysisModel, CfdDomain, ElectroThermalDomain, ExcitationAction, ExcitationTarget,
    PhysicalSignal, SignalId, SignalOrdinate, SignalRepresentation, SignalValue,
    ThermoMechanicalDomain,
};

pub(crate) fn cfd_velocity_scales(model: &AnalysisModel, domain: &CfdDomain) -> Vec<f64> {
    if domain.inlet_velocity_m_per_s.abs() <= f64::EPSILON {
        return Vec::new();
    }
    scalar_signal_values(model, domain.inlet_velocity_signal_id.as_ref(), |signal| {
        let PhysicalSignal::Velocity(signal) = signal else {
            return None;
        };
        scalar_values(&signal.representation, |value| value.meters_per_second)
    })
    .map(|values| {
        values
            .into_iter()
            .map(|value| value / domain.inlet_velocity_m_per_s)
            .collect()
    })
    .unwrap_or_default()
}

pub(crate) fn thermo_temperature_scales(
    model: &AnalysisModel,
    domain: &ThermoMechanicalDomain,
) -> Vec<f64> {
    if domain.applied_temperature_delta_k.abs() <= f64::EPSILON {
        return Vec::new();
    }
    scalar_signal_values(model, domain.temperature_signal_id.as_ref(), |signal| {
        let PhysicalSignal::Temperature(signal) = signal else {
            return None;
        };
        scalar_values(&signal.representation, |value| value.kelvin)
    })
    .map(|values| {
        values
            .into_iter()
            .map(|value| {
                (value - domain.reference_temperature_k) / domain.applied_temperature_delta_k
            })
            .collect()
    })
    .unwrap_or_default()
}

pub(crate) fn electric_potential_scales(
    model: &AnalysisModel,
    domain: &ElectroThermalDomain,
) -> Vec<f64> {
    if domain.applied_voltage_v.abs() <= f64::EPSILON {
        return Vec::new();
    }
    scalar_signal_values(
        model,
        domain.electric_potential_signal_id.as_ref(),
        |signal| {
            let PhysicalSignal::ElectricPotential(signal) = signal else {
                return None;
            };
            scalar_values(&signal.representation, |value| value.volts)
        },
    )
    .map(|values| {
        values
            .into_iter()
            .map(|value| value / domain.applied_voltage_v)
            .collect()
    })
    .unwrap_or_default()
}

pub(crate) fn pressure_sources(model: &AnalysisModel) -> Vec<(String, f64)> {
    model
        .excitations
        .iter()
        .filter_map(|action| match action {
            ExcitationAction::Pressure(value) | ExcitationAction::AcousticPressure(value) => {
                let target = target_id(&value.target);
                let signal = model
                    .signals
                    .iter()
                    .find(|signal| signal.signal_id() == &value.signal_id)?;
                let PhysicalSignal::Pressure(signal) = signal else {
                    return None;
                };
                constant_scalar(&signal.representation, |value| value.pascals)
                    .map(|magnitude| (target, magnitude))
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn has_nonzero_electromagnetic_source(model: &AnalysisModel) -> bool {
    model.excitations.iter().any(|action| match action {
        ExcitationAction::ElectricCurrent(value) => model
            .signals
            .iter()
            .find(|signal| signal.signal_id() == &value.signal_id)
            .and_then(|signal| match signal {
                PhysicalSignal::ElectricCurrent(signal) => Some(representation_has_nonzero(
                    &signal.representation,
                    |value| value.amperes,
                )),
                _ => None,
            })
            .unwrap_or(false),
        ExcitationAction::CurrentDensity(value) => model
            .signals
            .iter()
            .find(|signal| signal.signal_id() == &value.signal_id)
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

fn constant_scalar<Q>(
    representation: &SignalRepresentation<Q>,
    component: impl Fn(&Q) -> f64,
) -> Option<f64> {
    let SignalRepresentation::Constant(SignalValue::Scalar(value)) = representation else {
        return None;
    };
    Some(component(value))
}

fn scalar_signal_values(
    model: &AnalysisModel,
    signal_id: Option<&SignalId>,
    convert: impl FnOnce(&PhysicalSignal) -> Option<Vec<f64>>,
) -> Option<Vec<f64>> {
    let signal_id = signal_id?;
    model
        .signals
        .iter()
        .find(|signal| signal.signal_id() == signal_id)
        .and_then(convert)
}

fn scalar_values<Q>(
    representation: &SignalRepresentation<Q>,
    component: impl Fn(&Q) -> f64,
) -> Option<Vec<f64>> {
    match representation {
        SignalRepresentation::Constant(SignalValue::Scalar(value)) => Some(vec![component(value)]),
        SignalRepresentation::Tabulated(table) => table
            .values
            .iter()
            .map(|value| match value {
                SignalOrdinate::Real(SignalValue::Scalar(value)) => Some(component(value)),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn representation_has_nonzero<Q>(
    representation: &SignalRepresentation<Q>,
    component: impl Fn(&Q) -> f64,
) -> bool {
    match representation {
        SignalRepresentation::Constant(value) => value_has_nonzero(value, &component),
        SignalRepresentation::Tabulated(table) => table.values.iter().any(|value| match value {
            SignalOrdinate::Real(value) => value_has_nonzero(value, &component),
            SignalOrdinate::Complex { real, imaginary } => {
                value_has_nonzero(real, &component) || value_has_nonzero(imaginary, &component)
            }
        }),
        SignalRepresentation::Waveform(waveform) => {
            value_has_nonzero(&waveform.amplitude, &component)
                || waveform
                    .offset
                    .as_ref()
                    .is_some_and(|value| value_has_nonzero(value, &component))
        }
        SignalRepresentation::Artifact(_) | SignalRepresentation::Generator(_) => true,
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
