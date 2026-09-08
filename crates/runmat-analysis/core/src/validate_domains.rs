use crate::{AnalysisModel, AnalysisValidationError};

pub(crate) fn validate_domains(model: &AnalysisModel) -> Result<(), AnalysisValidationError> {
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
            if !electrical.conductivity_s_per_m.is_finite()
                || electrical.conductivity_s_per_m <= 0.0
            {
                return Err(AnalysisValidationError::InvalidMaterial {
                    material_id: material.material_id.clone(),
                    detail: "electro-thermal conductivity must be finite and positive".to_string(),
                });
            }
        }
    }
    if let Some(domain) = &model.electromagnetic {
        if !domain.reference_frequency.is_finite() || domain.reference_frequency.hertz <= 0.0 {
            return Err(domain_error(
                "electromagnetic",
                "reference frequency must be finite and positive",
            ));
        }
    }
    if let Some(domain) = &model.cfd {
        validate_fluid_domain(domain)?;
    }
    Ok(())
}

fn validate_fluid_domain(domain: &crate::CfdDomain) -> Result<(), AnalysisValidationError> {
    if !domain.density.is_finite() || domain.density.kilograms_per_cubic_meter <= 0.0 {
        return Err(domain_error(
            "fluid",
            "mass density must be finite and positive",
        ));
    }
    if !domain.dynamic_viscosity.is_finite() || domain.dynamic_viscosity.pascal_seconds <= 0.0 {
        return Err(domain_error(
            "fluid",
            "dynamic viscosity must be finite and positive",
        ));
    }
    if !domain.turbulence_intensity.is_finite()
        || !(0.0..=1.0).contains(&domain.turbulence_intensity.ratio)
    {
        return Err(domain_error(
            "fluid",
            "turbulence intensity must be finite and in [0, 1]",
        ));
    }
    Ok(())
}

fn validate_reference_temperature(
    domain: &'static str,
    value: crate::Temperature,
) -> Result<(), AnalysisValidationError> {
    if value.is_finite() && value.kelvin >= 0.0 {
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
