use std::collections::HashSet;

use crate::{
    AnalysisValidationError, ConductivityFrequencyPoint, MaterialAssignment, MaterialModel,
};

pub(crate) fn validate_materials(
    materials: &[MaterialModel],
    assignments: &[MaterialAssignment],
) -> Result<(), AnalysisValidationError> {
    let mut material_ids = HashSet::new();
    for material in materials {
        if material.material_id.trim().is_empty() {
            return Err(AnalysisValidationError::InvalidId { kind: "material" });
        }
        if !material_ids.insert(material.material_id.as_str()) {
            return Err(AnalysisValidationError::DuplicateId {
                kind: "material",
                id: material.material_id.clone(),
            });
        }
        validate_material(material)?;
    }

    let mut assigned_regions = HashSet::new();
    for assignment in assignments {
        let target_label = assignment.target_label();
        if assignment
            .selected_region_id()
            .is_some_and(|region_id| region_id.trim().is_empty())
        {
            return Err(invalid_assignment(
                target_label,
                "region id must be non-empty",
            ));
        }
        if !assigned_regions.insert(&assignment.target) {
            return Err(invalid_assignment(
                target_label,
                "a material domain target may have only one material assignment",
            ));
        }
        if assignment.expected_material_id.trim().is_empty()
            || assignment.assigned_material_id.trim().is_empty()
        {
            return Err(invalid_assignment(
                target_label,
                "expected and assigned material ids must be non-empty",
            ));
        }
        if !material_ids.contains(assignment.expected_material_id.as_str()) {
            return Err(invalid_assignment(
                target_label,
                format!(
                    "expected material `{}` does not exist",
                    assignment.expected_material_id
                ),
            ));
        }
        if !material_ids.contains(assignment.assigned_material_id.as_str()) {
            return Err(invalid_assignment(
                target_label,
                format!(
                    "assigned material `{}` does not exist",
                    assignment.assigned_material_id
                ),
            ));
        }
    }
    Ok(())
}

fn validate_material(material: &MaterialModel) -> Result<(), AnalysisValidationError> {
    if material.name.trim().is_empty() {
        return Err(invalid_material(material, "name must be non-empty"));
    }
    let mechanical = &material.mechanical;
    require_positive(material, "Young's modulus", mechanical.youngs_modulus_pa)?;
    require_positive(material, "mechanical density", mechanical.density_kg_per_m3)?;
    if !mechanical.poisson_ratio.is_finite()
        || mechanical.poisson_ratio <= -1.0
        || mechanical.poisson_ratio >= 0.5
    {
        return Err(invalid_material(
            material,
            "Poisson ratio must be finite and in (-1, 0.5)",
        ));
    }
    if let crate::MechanicalDampingModel::Rayleigh {
        mass_coefficient,
        stiffness_coefficient,
    } = mechanical.damping
    {
        if !mass_coefficient.is_finite()
            || !stiffness_coefficient.is_finite()
            || mass_coefficient.per_second < 0.0
            || stiffness_coefficient.seconds < 0.0
            || (mass_coefficient.per_second == 0.0 && stiffness_coefficient.seconds == 0.0)
        {
            return Err(invalid_material(
                material,
                "Rayleigh damping coefficients must be finite and non-negative, with at least one positive coefficient",
            ));
        }
    }

    let thermal = &material.thermal;
    require_nonnegative(
        material,
        "thermal reference temperature",
        thermal.reference_temperature_k,
    )?;
    require_finite(
        material,
        "temperature coefficient of modulus",
        thermal.modulus_temp_coeff_per_k,
    )?;
    require_positive(
        material,
        "thermal conductivity",
        thermal.conductivity_w_per_mk,
    )?;
    require_positive(material, "specific heat", thermal.specific_heat_j_per_kgk)?;
    require_finite(
        material,
        "thermal expansion coefficient",
        thermal.expansion_coefficient_per_k,
    )?;

    if let Some(acoustic) = &material.acoustic {
        require_positive(material, "acoustic density", acoustic.density_kg_per_m3)?;
        require_positive(material, "speed of sound", acoustic.speed_of_sound_m_per_s)?;
        if !acoustic.damping_ratio.is_finite() || !(0.0..1.0).contains(&acoustic.damping_ratio) {
            return Err(invalid_material(
                material,
                "acoustic damping ratio must be finite and in [0, 1)",
            ));
        }
    }

    if let Some(electrical) = &material.electrical {
        require_nonnegative(
            material,
            "electrical reference temperature",
            electrical.reference_temperature_k,
        )?;
        require_nonnegative(
            material,
            "electrical conductivity",
            electrical.conductivity_s_per_m,
        )?;
        require_finite(
            material,
            "resistive heating coefficient",
            electrical.resistive_heating_coefficient,
        )?;
        require_positive(
            material,
            "relative permittivity",
            electrical.relative_permittivity,
        )?;
        require_positive(
            material,
            "relative permeability",
            electrical.relative_permeability,
        )?;
        validate_frequency_response(material, &electrical.conductivity_frequency_response)?;
    }

    if let Some(plastic) = &material.plastic {
        require_positive(material, "yield strain", plastic.yield_strain)?;
        require_nonnegative(
            material,
            "hardening modulus ratio",
            plastic.hardening_modulus_ratio,
        )?;
        require_positive(
            material,
            "plastic saturation exponent",
            plastic.saturation_exponent,
        )?;
    }
    Ok(())
}

fn validate_frequency_response(
    material: &MaterialModel,
    points: &[ConductivityFrequencyPoint],
) -> Result<(), AnalysisValidationError> {
    let mut previous_frequency = None;
    for point in points {
        require_positive(
            material,
            "conductivity response frequency",
            point.frequency_hz,
        )?;
        if previous_frequency.is_some_and(|previous| point.frequency_hz <= previous) {
            return Err(invalid_material(
                material,
                "conductivity response frequencies must be strictly increasing",
            ));
        }
        previous_frequency = Some(point.frequency_hz);
        require_nonnegative(
            material,
            "conductivity response scale",
            point.conductivity_scale,
        )?;
        if let Some(value) = point.dispersive_loss_scale {
            require_nonnegative(material, "dispersive loss scale", value)?;
        }
        if let Some(value) = point.relative_permittivity_scale {
            require_positive(material, "relative permittivity scale", value)?;
        }
        if let Some(value) = point.relative_permeability_scale {
            require_positive(material, "relative permeability scale", value)?;
        }
    }
    Ok(())
}

fn require_finite(
    material: &MaterialModel,
    property: &str,
    value: f64,
) -> Result<(), AnalysisValidationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_material(
            material,
            format!("{property} must be finite"),
        ))
    }
}

fn require_positive(
    material: &MaterialModel,
    property: &str,
    value: f64,
) -> Result<(), AnalysisValidationError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid_material(
            material,
            format!("{property} must be finite and positive"),
        ))
    }
}

fn require_nonnegative(
    material: &MaterialModel,
    property: &str,
    value: f64,
) -> Result<(), AnalysisValidationError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(invalid_material(
            material,
            format!("{property} must be finite and nonnegative"),
        ))
    }
}

fn invalid_material(
    material: &MaterialModel,
    detail: impl Into<String>,
) -> AnalysisValidationError {
    AnalysisValidationError::InvalidMaterial {
        material_id: material.material_id.clone(),
        detail: detail.into(),
    }
}

fn invalid_assignment(region_id: &str, detail: impl Into<String>) -> AnalysisValidationError {
    AnalysisValidationError::InvalidMaterialAssignment {
        region_id: region_id.to_string(),
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EvidenceConfidence, MaterialElectricalModel, MaterialMechanicalModel, MaterialThermalModel,
        MechanicalDampingModel, RayleighMassCoefficient, RayleighStiffnessCoefficient,
    };

    fn material() -> MaterialModel {
        MaterialModel {
            material_id: "steel".to_string(),
            name: "Steel".to_string(),
            mechanical: MaterialMechanicalModel {
                youngs_modulus_pa: 200.0e9,
                poisson_ratio: 0.3,
                density_kg_per_m3: 7850.0,
                damping: MechanicalDampingModel::Undamped,
            },
            thermal: MaterialThermalModel::default(),
            acoustic: None,
            electrical: Some(MaterialElectricalModel::default()),
            plastic: None,
        }
    }

    #[test]
    fn validates_constitutive_ranges() {
        let mut invalid = material();
        invalid.electrical.as_mut().unwrap().conductivity_s_per_m = -1.0;
        assert!(matches!(
            validate_materials(&[invalid], &[]),
            Err(AnalysisValidationError::InvalidMaterial { .. })
        ));

        let mut invalid = material();
        invalid.mechanical.poisson_ratio = 0.5;
        assert!(matches!(
            validate_materials(&[invalid], &[]),
            Err(AnalysisValidationError::InvalidMaterial { .. })
        ));

        let mut invalid = material();
        invalid.mechanical.poisson_ratio = -1.0;
        assert!(matches!(
            validate_materials(&[invalid], &[]),
            Err(AnalysisValidationError::InvalidMaterial { .. })
        ));

        let mut auxetic = material();
        auxetic.mechanical.poisson_ratio = -0.25;
        validate_materials(&[auxetic], &[]).expect("finite auxetic material should be valid");

        let mut damped = material();
        damped.mechanical.damping = MechanicalDampingModel::Rayleigh {
            mass_coefficient: RayleighMassCoefficient::new(0.0),
            stiffness_coefficient: RayleighStiffnessCoefficient::new(0.0),
        };
        assert!(matches!(
            validate_materials(&[damped], &[]),
            Err(AnalysisValidationError::InvalidMaterial { .. })
        ));

        let mut damped = material();
        damped.mechanical.damping = MechanicalDampingModel::Rayleigh {
            mass_coefficient: RayleighMassCoefficient::new(0.25),
            stiffness_coefficient: RayleighStiffnessCoefficient::new(1.0e-5),
        };
        validate_materials(&[damped], &[]).expect("positive Rayleigh damping should be valid");
    }

    #[test]
    fn validates_material_assignment_references_and_evidence() {
        let assignment = MaterialAssignment {
            target: crate::PropertyRegionTarget::solid("solid"),
            expected_material_id: "steel".to_string(),
            assigned_material_id: "missing".to_string(),
            confidence: EvidenceConfidence::Verified,
        };
        assert!(matches!(
            validate_materials(&[material()], &[assignment]),
            Err(AnalysisValidationError::InvalidMaterialAssignment { .. })
        ));

        let mut alternate = material();
        alternate.material_id = "polymer".to_string();
        alternate.name = "Polymer".to_string();
        let mismatch = MaterialAssignment {
            target: crate::PropertyRegionTarget::solid("solid"),
            expected_material_id: "steel".to_string(),
            assigned_material_id: "polymer".to_string(),
            confidence: EvidenceConfidence::Inferred,
        };
        validate_materials(&[material(), alternate], &[mismatch])
            .expect("a deliberate assignment mismatch may carry quality evidence");
    }
}
