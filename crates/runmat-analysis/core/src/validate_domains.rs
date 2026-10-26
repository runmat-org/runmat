use crate::{AnalysisModel, AnalysisValidationError};

pub(crate) fn validate_domains(model: &AnalysisModel) -> Result<(), AnalysisValidationError> {
    validate_maxwell_constraints(model.electromagnetic.as_ref(), &model.constraints)?;
    if let Some(domain) = &model.thermal {
        validate_reference_temperature("thermal", domain.reference_temperature)?;
    }
    if let Some(domain) = &model.thermo_mechanical {
        validate_reference_temperature("thermo_mechanical", domain.reference_temperature)?;
    }
    if let Some(domain) = &model.electro_thermal {
        validate_reference_temperature("electro_thermal", domain.reference_temperature)?;
        for material in &model.materials {
            let Some(electrical) = &material.electrical else {
                return Err(AnalysisValidationError::InvalidMaterial {
                    material_id: material.material_id.clone(),
                    detail: "electro-thermal analysis requires electrical material properties"
                        .to_string(),
                });
            };
            if !electrical
                .conductivity_s_per_m
                .canonical_value()
                .is_finite()
                || electrical.conductivity_s_per_m.canonical_value() <= 0.0
            {
                return Err(AnalysisValidationError::InvalidMaterial {
                    material_id: material.material_id.clone(),
                    detail: "electro-thermal conductivity must be finite and positive".to_string(),
                });
            }
        }
    }
    if let Some(domain) = &model.electromagnetic {
        match domain {
            crate::ElectromagneticDomain::Magnetostatic {
                formulation: crate::MagnetostaticFormulation::MagneticVectorPotential,
            } => {
                for material in &model.materials {
                    let Some(electrical) = &material.electrical else {
                        return Err(AnalysisValidationError::InvalidMaterial {
                            material_id: material.material_id.clone(),
                            detail: "magnetostatics requires magnetic permeability properties"
                                .into(),
                        });
                    };
                    if !electrical
                        .relative_permeability
                        .canonical_value()
                        .is_finite()
                        || electrical.relative_permeability.canonical_value() <= 0.0
                    {
                        return Err(AnalysisValidationError::InvalidMaterial {
                            material_id: material.material_id.clone(),
                            detail:
                                "magnetostatic relative permeability must be finite and positive"
                                    .into(),
                        });
                    }
                }
            }
            crate::ElectromagneticDomain::Harmonic {
                reference_frequency,
            } => {
                if !reference_frequency.is_finite() || reference_frequency.canonical_value() <= 0.0
                {
                    return Err(domain_error(
                        "electromagnetic",
                        "harmonic reference frequency must be finite and positive",
                    ));
                }
            }
            crate::ElectromagneticDomain::Static {
                equation: crate::StaticElectricalEquation::SteadyConduction,
            } => {
                for material in &model.materials {
                    let Some(electrical) = &material.electrical else {
                        return Err(AnalysisValidationError::InvalidMaterial {
                            material_id: material.material_id.clone(),
                            detail: "steady conduction requires electrical material properties"
                                .into(),
                        });
                    };
                    if !electrical
                        .conductivity_s_per_m
                        .canonical_value()
                        .is_finite()
                        || electrical.conductivity_s_per_m.canonical_value() <= 0.0
                    {
                        return Err(AnalysisValidationError::InvalidMaterial {
                            material_id: material.material_id.clone(),
                            detail: "steady conduction conductivity must be finite and positive"
                                .into(),
                        });
                    }
                }
            }
            crate::ElectromagneticDomain::Static {
                equation: crate::StaticElectricalEquation::Electrostatics,
            } => {}
        }
    }
    if let Some(domain) = &model.cfd {
        validate_fluid_domain(domain)?;
    }
    Ok(())
}

fn validate_fluid_domain(domain: &crate::CfdDomain) -> Result<(), AnalysisValidationError> {
    if !domain.density.is_finite() || domain.density.canonical_value() <= 0.0 {
        return Err(domain_error(
            "fluid",
            "mass density must be finite and positive",
        ));
    }
    if !domain.dynamic_viscosity.is_finite() || domain.dynamic_viscosity.canonical_value() <= 0.0 {
        return Err(domain_error(
            "fluid",
            "dynamic viscosity must be finite and positive",
        ));
    }
    if !domain.turbulence_intensity.is_finite()
        || !(0.0..=1.0).contains(&domain.turbulence_intensity.canonical_value())
    {
        return Err(domain_error(
            "fluid",
            "turbulence intensity must be finite and in [0, 1]",
        ));
    }
    Ok(())
}

fn validate_maxwell_constraints(
    domain: Option<&crate::ElectromagneticDomain>,
    constraints: &[crate::AnalysisConstraint],
) -> Result<(), AnalysisValidationError> {
    let harmonic = matches!(domain, Some(crate::ElectromagneticDomain::Harmonic { .. }));
    for constraint in constraints {
        let pec = matches!(constraint, crate::AnalysisConstraint::Electrical(value)
            if matches!(value.kind, crate::ElectricalConstraintKind::ElectricPerfectConductor { .. }));
        if pec
            && (!harmonic
                || constraint
                    .target()
                    .is_none_or(|target| target.entity_dimension != crate::EntityDimension::Face))
        {
            return Err(AnalysisValidationError::InvalidConstraint {
                constraint_id: constraint.constraint_id().into(),
                detail: "electric perfect conductor requires harmonic Maxwell and a face target"
                    .into(),
            });
        }
        // Harmonic fields admit grounded faces (tangential E = 0) and terminal potentials or
        // currents of driven conductors; a Coulomb gauge or prescribed vector potential has
        // no meaning for the electric-field formulation.
        let admitted_in_harmonic = pec
            || matches!(constraint, crate::AnalysisConstraint::Magnetic(value)
                if matches!(value.kind, crate::MagneticConstraintKind::VectorPotentialGround { .. }))
            || matches!(constraint, crate::AnalysisConstraint::Electrical(value)
                if matches!(value.kind,
                    crate::ElectricalConstraintKind::PrescribedPotential { .. }
                    | crate::ElectricalConstraintKind::PotentialReference { .. }
                    | crate::ElectricalConstraintKind::IntegralCurrent { .. }));
        if harmonic
            && matches!(
                constraint,
                crate::AnalysisConstraint::Magnetic(_) | crate::AnalysisConstraint::Electrical(_)
            )
            && !admitted_in_harmonic
        {
            return Err(AnalysisValidationError::InvalidConstraint {
                constraint_id: constraint.constraint_id().into(),
                detail: "harmonic Maxwell admits grounded faces (magnetic.vector_potential_ground, electrical.perfect_conductor) and terminal potentials or currents; a Coulomb gauge, prescribed or periodic vector potential is not admitted".into(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "validate_domains/maxwell_tests.rs"]
mod maxwell_tests;

fn validate_reference_temperature(
    domain: &'static str,
    value: crate::Temperature,
) -> Result<(), AnalysisValidationError> {
    if value.is_finite() && value.canonical_value() >= 0.0 {
        Ok(())
    } else {
        Err(domain_error(
            domain,
            "reference temperature must be finite and at or above zero kelvin",
        ))
    }
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
    use crate::{CfdDomain, DynamicViscosity, MassDensity, Temperature, TurbulenceIntensity};

    #[test]
    fn physical_domain_reference_values_are_typed_and_validated() {
        assert!(matches!(
            validate_reference_temperature("thermal", Temperature::new(-1.0)),
            Err(AnalysisValidationError::InvalidDomain {
                domain: "thermal",
                ..
            })
        ));
        validate_reference_temperature("thermo_mechanical", Temperature::new(293.15)).unwrap();
        let domain = CfdDomain {
            density: MassDensity::new(1.225),
            dynamic_viscosity: DynamicViscosity::new(1.81e-5),
            turbulence_intensity: TurbulenceIntensity::new(0.05),
            rheology: None,
            porous_zones: Vec::new(),
        };
        validate_fluid_domain(&domain).unwrap();

        let invalid = CfdDomain {
            turbulence_intensity: TurbulenceIntensity::new(1.1),
            ..domain
        };
        assert!(matches!(
            validate_fluid_domain(&invalid),
            Err(AnalysisValidationError::InvalidDomain {
                domain: "fluid",
                ..
            })
        ));
    }
}
