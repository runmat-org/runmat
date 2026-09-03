use crate::{
    AnalysisModel, AnalysisValidationError, CfdDomain, CfdSolveFamily, ElectroThermalDomain,
    PhysicalQuantityKind, PhysicalSignal, SignalDomainKind, SignalId, SignalShape,
    ThermoMechanicalDomain,
};

pub(crate) fn validate_domains(model: &AnalysisModel) -> Result<(), AnalysisValidationError> {
    if let Some(domain) = &model.thermo_mechanical {
        validate_thermo_mechanical(domain, &model.signals)?;
    }
    if let Some(domain) = &model.electro_thermal {
        validate_electro_thermal(domain, &model.signals)?;
    }
    if let Some(domain) = &model.cfd {
        validate_cfd(domain, &model.signals)?;
    }
    Ok(())
}

fn validate_thermo_mechanical(
    domain: &ThermoMechanicalDomain,
    signals: &[PhysicalSignal],
) -> Result<(), AnalysisValidationError> {
    validate_optional_signal(
        "thermo_mechanical",
        domain.temperature_signal_id.as_ref(),
        signals,
        PhysicalQuantityKind::Temperature,
        &[SignalShape::Scalar],
        &[SignalDomainKind::Constant, SignalDomainKind::Time],
    )
}

fn validate_electro_thermal(
    domain: &ElectroThermalDomain,
    signals: &[PhysicalSignal],
) -> Result<(), AnalysisValidationError> {
    validate_optional_signal(
        "electro_thermal",
        domain.electric_potential_signal_id.as_ref(),
        signals,
        PhysicalQuantityKind::ElectricPotential,
        &[SignalShape::Scalar],
        &[SignalDomainKind::Constant, SignalDomainKind::Time],
    )
}

fn validate_cfd(
    domain: &CfdDomain,
    signals: &[PhysicalSignal],
) -> Result<(), AnalysisValidationError> {
    let allowed_domains: &[SignalDomainKind] = match domain.solve_family {
        CfdSolveFamily::SteadyState => &[SignalDomainKind::Constant],
        CfdSolveFamily::Transient => &[SignalDomainKind::Constant, SignalDomainKind::Time],
    };
    validate_optional_signal(
        "cfd",
        domain.inlet_velocity_signal_id.as_ref(),
        signals,
        PhysicalQuantityKind::Velocity,
        &[SignalShape::Scalar],
        allowed_domains,
    )
}

fn validate_optional_signal(
    domain_name: &'static str,
    signal_id: Option<&SignalId>,
    signals: &[PhysicalSignal],
    expected_quantity: PhysicalQuantityKind,
    allowed_shapes: &[SignalShape],
    allowed_domains: &[SignalDomainKind],
) -> Result<(), AnalysisValidationError> {
    let Some(signal_id) = signal_id else {
        return Ok(());
    };
    let Some(signal) = signals
        .iter()
        .find(|signal| signal.signal_id() == signal_id)
    else {
        return Err(domain_error(
            domain_name,
            format!("referenced signal `{}` does not exist", signal_id.0),
        ));
    };
    if signal.quantity() != expected_quantity {
        return Err(domain_error(
            domain_name,
            format!(
                "signal `{}` must provide {:?}, not {:?}",
                signal_id.0,
                expected_quantity,
                signal.quantity()
            ),
        ));
    }
    if !allowed_shapes.contains(signal.shape()) {
        return Err(domain_error(
            domain_name,
            format!(
                "signal `{}` has unsupported shape {:?}",
                signal_id.0,
                signal.shape()
            ),
        ));
    }
    if !allowed_domains.contains(&signal.domain()) {
        return Err(domain_error(
            domain_name,
            format!(
                "signal `{}` has unsupported domain {:?}",
                signal_id.0,
                signal.domain()
            ),
        ));
    }
    Ok(())
}

fn domain_error(domain: &'static str, detail: impl Into<String>) -> AnalysisValidationError {
    AnalysisValidationError::InvalidDomain {
        domain,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ElectricPotential, Pressure, Signal, SignalApproximationEvidence, SignalAxis,
        SignalExtrapolation, SignalInterpolation, SignalOrdinate, SignalPrecision,
        SignalProvenance, SignalRepresentation, SignalValue, TabulatedSignal, Temperature,
        Velocity,
    };

    fn evidence(sample_count: usize) -> SignalApproximationEvidence {
        SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count,
        }
    }

    fn temperature_signal(id: &str, shape: SignalShape) -> PhysicalSignal {
        PhysicalSignal::Temperature(Signal {
            signal_id: SignalId(id.into()),
            shape,
            frame: None,
            representation: SignalRepresentation::Constant(SignalValue::Scalar(Temperature::new(
                300.0,
            ))),
            precision: SignalPrecision::Real64,
            evidence: evidence(1),
            provenance: SignalProvenance::Authored,
        })
    }

    fn potential_signal(id: &str) -> PhysicalSignal {
        PhysicalSignal::ElectricPotential(Signal {
            signal_id: SignalId(id.into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Constant(SignalValue::Scalar(
                ElectricPotential::new(12.0),
            )),
            precision: SignalPrecision::Real64,
            evidence: evidence(1),
            provenance: SignalProvenance::Authored,
        })
    }

    fn time_velocity_signal(id: &str, shape: SignalShape) -> PhysicalSignal {
        let value = match shape {
            SignalShape::Vector3 => {
                SignalValue::Vector3([Velocity::new(1.0), Velocity::new(0.0), Velocity::new(0.0)])
            }
            _ => SignalValue::Scalar(Velocity::new(1.0)),
        };
        PhysicalSignal::Velocity(Signal {
            signal_id: SignalId(id.into()),
            shape,
            frame: None,
            representation: SignalRepresentation::Tabulated(TabulatedSignal {
                axis: SignalAxis::TimeSeconds(vec![0.0, 1.0]),
                values: vec![
                    SignalOrdinate::Real(value.clone()),
                    SignalOrdinate::Real(value),
                ],
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::HoldEndpoint,
                breakpoints: Vec::new(),
            }),
            precision: SignalPrecision::Real64,
            evidence: evidence(2),
            provenance: SignalProvenance::Authored,
        })
    }

    #[test]
    fn thermo_mechanical_accepts_scalar_temperature_signal() {
        let domain = ThermoMechanicalDomain {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 0.0,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            temperature_signal_id: Some(SignalId("temperature".into())),
        };
        assert_eq!(
            validate_thermo_mechanical(
                &domain,
                &[temperature_signal("temperature", SignalShape::Scalar)]
            ),
            Ok(())
        );
    }

    #[test]
    fn domain_signal_must_exist_and_match_quantity_and_shape() {
        let domain = ThermoMechanicalDomain {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_temperature_delta_k: 0.0,
            field_artifact_id: None,
            field_source: None,
            region_temperature_deltas: Vec::new(),
            temperature_signal_id: Some(SignalId("temperature".into())),
        };
        assert!(validate_thermo_mechanical(&domain, &[]).is_err());

        let pressure = PhysicalSignal::Pressure(Signal {
            signal_id: SignalId("temperature".into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Constant(SignalValue::Scalar(Pressure::new(1.0))),
            precision: SignalPrecision::Real64,
            evidence: evidence(1),
            provenance: SignalProvenance::Authored,
        });
        assert!(validate_thermo_mechanical(&domain, &[pressure]).is_err());
        assert!(validate_thermo_mechanical(
            &domain,
            &[temperature_signal("temperature", SignalShape::Vector3)]
        )
        .is_err());
    }

    #[test]
    fn electro_thermal_accepts_scalar_potential_signal() {
        let domain = ElectroThermalDomain {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 12.0,
            region_conductivity_scales: Vec::new(),
            electric_potential_signal_id: Some(SignalId("potential".into())),
        };
        assert_eq!(
            validate_electro_thermal(&domain, &[potential_signal("potential")]),
            Ok(())
        );
    }

    #[test]
    fn cfd_time_signal_requires_transient_scalar_semantics() {
        let mut domain = CfdDomain {
            enabled: true,
            solve_family: CfdSolveFamily::Transient,
            reference_density_kg_per_m3: 1.2,
            dynamic_viscosity_pa_s: 1.8e-5,
            inlet_velocity_m_per_s: 1.0,
            turbulence_intensity: 0.01,
            inlet_velocity_signal_id: Some(SignalId("velocity".into())),
        };
        assert_eq!(
            validate_cfd(
                &domain,
                &[time_velocity_signal("velocity", SignalShape::Scalar)]
            ),
            Ok(())
        );
        domain.solve_family = CfdSolveFamily::SteadyState;
        assert!(validate_cfd(
            &domain,
            &[time_velocity_signal("velocity", SignalShape::Scalar)]
        )
        .is_err());
        domain.solve_family = CfdSolveFamily::Transient;
        assert!(validate_cfd(
            &domain,
            &[time_velocity_signal("velocity", SignalShape::Vector3)]
        )
        .is_err());
    }
}
