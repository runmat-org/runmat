use std::collections::HashSet;

use crate::{
    AnalysisValidationError, ConductivityFrequencyPoint, ImportedCadMaterialPropertyKind,
    MaterialAssignment, MaterialModel, MaterialProvenance,
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
        assignment
            .target
            .validate_resolution()
            .map_err(|error| invalid_assignment(target_label, error.to_string()))?;
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

pub(crate) fn validate_material_definitions(
    materials: &[MaterialModel],
) -> Result<(), AnalysisValidationError> {
    validate_materials(materials, &[])
}

fn validate_material(material: &MaterialModel) -> Result<(), AnalysisValidationError> {
    if material.name.trim().is_empty() {
        return Err(invalid_material(material, "name must be non-empty"));
    }
    validate_provenance(material)?;
    if material
        .mass_density_kg_per_m3
        .map(|quantity| quantity.canonical_value())
        .is_none()
        && material.elasticity.is_none()
        && material.damping.is_none()
        && material.thermal.is_none()
        && material.acoustic.is_none()
        && material.fluid.is_none()
        && material.electrical.is_none()
        && material.plastic.is_none()
    {
        return Err(invalid_material(
            material,
            "at least one constitutive capability must be declared",
        ));
    }
    if let Some(density) = material
        .mass_density_kg_per_m3
        .map(|quantity| quantity.canonical_value())
    {
        require_positive(material, "mass density", density)?;
    }
    match material.fluid {
        Some(crate::MaterialFluidModel::PorousMedium { law }) => {
            if !law.is_valid() {
                return Err(invalid_material(
                    material,
                    "porous medium needs a positive permeability, a nonnegative inertial \
                     coefficient and a porosity in (0, 1]",
                ));
            }
        }
        Some(_) => {
            material
                .require_incompressible_fluid()
                .map_err(|error| invalid_material(material, error.to_string()))?;
        }
        None => {}
    }
    if let Some(crate::MaterialElasticityModel::IsotropicLinear(mechanical)) = &material.elasticity
    {
        require_positive(
            material,
            "Young's modulus",
            mechanical.youngs_modulus_pa.canonical_value(),
        )?;
        if !mechanical.poisson_ratio.canonical_value().is_finite()
            || mechanical.poisson_ratio.canonical_value() <= -1.0
            || mechanical.poisson_ratio.canonical_value() >= 0.5
        {
            return Err(invalid_material(
                material,
                "Poisson ratio must be finite and in (-1, 0.5)",
            ));
        }
    }
    if let Some(crate::MaterialElasticityModel::OrthotropicLamina(lamina)) = &material.elasticity {
        require_positive(
            material,
            "lamina longitudinal modulus",
            lamina.longitudinal_modulus_pa.canonical_value(),
        )?;
        require_positive(
            material,
            "lamina transverse modulus",
            lamina.transverse_modulus_pa.canonical_value(),
        )?;
        require_positive(
            material,
            "lamina in-plane shear modulus",
            lamina.in_plane_shear_modulus_pa.canonical_value(),
        )?;
        require_positive(
            material,
            "lamina transverse shear modulus 13",
            lamina.transverse_shear_modulus_13_pa.canonical_value(),
        )?;
        require_positive(
            material,
            "lamina transverse shear modulus 23",
            lamina.transverse_shear_modulus_23_pa.canonical_value(),
        )?;
        if !lamina.major_poisson_ratio.canonical_value().is_finite() {
            return Err(invalid_material(
                material,
                "lamina major Poisson ratio must be finite",
            ));
        }
        let reciprocal = lamina.major_poisson_ratio.canonical_value()
            * lamina.transverse_modulus_pa.canonical_value()
            / lamina.longitudinal_modulus_pa.canonical_value();
        if 1.0 - lamina.major_poisson_ratio.canonical_value() * reciprocal <= 0.0 {
            return Err(invalid_material(
                material,
                "lamina plane-stress compliance must be positive definite",
            ));
        }
    }
    if let Some(crate::MechanicalDampingModel::Rayleigh {
        mass_coefficient,
        stiffness_coefficient,
    }) = material.damping
    {
        if !mass_coefficient.is_finite()
            || !stiffness_coefficient.is_finite()
            || mass_coefficient.canonical_value() < 0.0
            || stiffness_coefficient.canonical_value() < 0.0
            || (mass_coefficient.canonical_value() == 0.0
                && stiffness_coefficient.canonical_value() == 0.0)
        {
            return Err(invalid_material(
                material,
                "Rayleigh damping coefficients must be finite and non-negative, with at least one positive coefficient",
            ));
        }
    }

    if let Some(thermal) = &material.thermal {
        require_nonnegative(
            material,
            "thermal reference temperature",
            thermal.reference_temperature_k.canonical_value(),
        )?;
        require_finite(
            material,
            "temperature coefficient of modulus",
            thermal.modulus_temp_coeff_per_k.canonical_value(),
        )?;
        require_positive(
            material,
            "thermal conductivity",
            thermal.conductivity_w_per_mk.canonical_value(),
        )?;
        require_positive(
            material,
            "specific heat",
            thermal.specific_heat_j_per_kgk.canonical_value(),
        )?;
        require_finite(
            material,
            "thermal expansion coefficient",
            thermal.expansion_coefficient_per_k.canonical_value(),
        )?;
    }

    if let Some(acoustic) = &material.acoustic {
        require_positive(
            material,
            "speed of sound",
            acoustic.speed_of_sound_m_per_s.canonical_value(),
        )?;
        if !acoustic.damping_ratio.canonical_value().is_finite()
            || !(0.0..1.0).contains(&acoustic.damping_ratio.canonical_value())
        {
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
            electrical.reference_temperature_k.canonical_value(),
        )?;
        require_nonnegative(
            material,
            "electrical conductivity",
            electrical.conductivity_s_per_m.canonical_value(),
        )?;
        require_finite(
            material,
            "resistive heating coefficient",
            electrical.resistive_heating_coefficient.canonical_value(),
        )?;
        require_positive(
            material,
            "relative permittivity",
            electrical.relative_permittivity.canonical_value(),
        )?;
        require_positive(
            material,
            "relative permeability",
            electrical.relative_permeability.canonical_value(),
        )?;
        validate_frequency_response(material, &electrical.conductivity_frequency_response)?;
    }

    if let Some(plastic) = &material.plastic {
        let crate::MaterialPlasticModel::J2IsotropicLinearHardening {
            yield_stress_pa,
            isotropic_hardening_modulus_pa,
        } = plastic;
        material.require_mechanical().map_err(|_| {
            invalid_material(
                material,
                "J2 plasticity requires isotropic linear elasticity",
            )
        })?;
        require_positive(material, "yield stress", yield_stress_pa.canonical_value())?;
        require_nonnegative(
            material,
            "isotropic hardening modulus",
            isotropic_hardening_modulus_pa.canonical_value(),
        )?;
    }
    Ok(())
}

fn validate_provenance(material: &MaterialModel) -> Result<(), AnalysisValidationError> {
    let MaterialProvenance::ImportedCad(source) = &material.provenance else {
        return Ok(());
    };
    if source.geometry_module_path.trim().is_empty()
        || source.geometry_id.trim().is_empty()
        || source.source_material_id.trim().is_empty()
        || source.source_material_name.trim().is_empty()
        || source.source_entity_id == 0
    {
        return Err(invalid_material(
            material,
            "imported CAD material provenance requires complete nonzero source identity",
        ));
    }
    if source
        .persistent_id
        .as_ref()
        .is_some_and(|value| value.trim().is_empty())
        || source
            .specification
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
    {
        return Err(invalid_material(
            material,
            "imported CAD material identifiers and specifications must be non-empty",
        ));
    }
    let mut associations = HashSet::new();
    for association in &source.associations {
        if association.association_id.trim().is_empty()
            || association.source_entity_id == 0
            || !associations.insert(association.association_id.as_str())
        {
            return Err(invalid_material(
                material,
                "imported CAD material associations require unique nonzero source identity",
            ));
        }
    }
    if source.adopted_properties.is_empty() {
        return Err(invalid_material(
            material,
            "imported CAD material provenance requires adopted properties",
        ));
    }
    let mut properties = HashSet::new();
    for property in &source.adopted_properties {
        if property.source_property_id.trim().is_empty()
            || property.source_name.trim().is_empty()
            || property.value.source_symbol().trim().is_empty()
            || property.source_entity_id == 0
            || property.representation_entity_id == 0
            || property.value.validate().is_err()
            || !property.value.source_value().is_finite()
            || !properties.insert(property.property)
        {
            return Err(invalid_material(
                material,
                "adopted CAD properties require unique kinds and complete finite source evidence",
            ));
        }
        for condition in &property.environment {
            if condition.name.trim().is_empty()
                || condition.value.source_symbol().trim().is_empty()
                || condition.value.validate().is_err()
                || !condition.value.source_value().is_finite()
            {
                return Err(invalid_material(
                    material,
                    "adopted CAD property environments require complete finite evidence",
                ));
            }
        }
        let canonical = match property.property {
            ImportedCadMaterialPropertyKind::MassDensity => material
                .mass_density_kg_per_m3
                .map(|quantity| quantity.canonical_value()),
            ImportedCadMaterialPropertyKind::YoungsModulus => material
                .require_mechanical()
                .ok()
                .map(|mechanical| mechanical.youngs_modulus_pa.canonical_value()),
            ImportedCadMaterialPropertyKind::PoissonRatio => material
                .require_mechanical()
                .ok()
                .map(|mechanical| mechanical.poisson_ratio.canonical_value()),
            _ => None,
        };
        if canonical
            .is_some_and(|value| value.to_bits() != property.value.canonical_value().to_bits())
        {
            return Err(invalid_material(
                material,
                "canonical material values must equal their adopted CAD conversion evidence",
            ));
        }
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
            point.frequency_hz.canonical_value(),
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
        EvidenceConfidence, MaterialElasticityModel, MaterialElectricalModel,
        MaterialMechanicalModel, MaterialThermalModel, MechanicalDampingModel,
        RayleighMassCoefficient, RayleighStiffnessCoefficient,
    };

    fn material() -> MaterialModel {
        MaterialModel {
            fluid: None,
            material_id: "steel".to_string(),
            name: "Steel".to_string(),
            provenance: crate::MaterialProvenance::Authored,
            mass_density_kg_per_m3: Some(runmat_types::physical::MassDensity::new(7850.0)),
            elasticity: Some(MaterialElasticityModel::IsotropicLinear(
                MaterialMechanicalModel {
                    youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(200.0e9),
                    poisson_ratio: runmat_types::physical::PoissonRatio::new(0.3),
                },
            )),
            damping: None,
            thermal: Some(MaterialThermalModel {
                reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
                modulus_temp_coeff_per_k: runmat_types::physical::InverseTemperature::new(-2.5e-4),
                conductivity_w_per_mk: runmat_types::physical::ThermalConductivity::new(45.0),
                specific_heat_j_per_kgk: runmat_types::physical::SpecificHeat::new(500.0),
                expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion::new(1.2e-5),
            }),
            acoustic: None,
            electrical: Some(MaterialElectricalModel {
                reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
                conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(1.0),
                resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
                relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
                relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
                conductivity_frequency_response: Vec::new(),
            }),
            plastic: None,
        }
    }

    #[test]
    fn validates_constitutive_ranges() {
        let mut invalid = material();
        invalid.electrical.as_mut().unwrap().conductivity_s_per_m =
            runmat_types::physical::ElectricalConductivity::new(-1.0);
        assert!(matches!(
            validate_materials(&[invalid], &[]),
            Err(AnalysisValidationError::InvalidMaterial { .. })
        ));

        let mut invalid = material();
        invalid.require_mechanical_mut().unwrap().poisson_ratio =
            runmat_types::physical::PoissonRatio::new(0.5);
        assert!(matches!(
            validate_materials(&[invalid], &[]),
            Err(AnalysisValidationError::InvalidMaterial { .. })
        ));

        let mut invalid = material();
        invalid.require_mechanical_mut().unwrap().poisson_ratio =
            runmat_types::physical::PoissonRatio::new(-1.0);
        assert!(matches!(
            validate_materials(&[invalid], &[]),
            Err(AnalysisValidationError::InvalidMaterial { .. })
        ));

        let mut auxetic = material();
        auxetic.require_mechanical_mut().unwrap().poisson_ratio =
            runmat_types::physical::PoissonRatio::new(-0.25);
        validate_materials(&[auxetic], &[]).expect("finite auxetic material should be valid");

        let mut damped = material();
        damped.damping = Some(MechanicalDampingModel::Rayleigh {
            mass_coefficient: RayleighMassCoefficient::new(0.0),
            stiffness_coefficient: RayleighStiffnessCoefficient::new(0.0),
        });
        assert!(matches!(
            validate_materials(&[damped], &[]),
            Err(AnalysisValidationError::InvalidMaterial { .. })
        ));

        let mut damped = material();
        damped.damping = Some(MechanicalDampingModel::Rayleigh {
            mass_coefficient: RayleighMassCoefficient::new(0.25),
            stiffness_coefficient: RayleighStiffnessCoefficient::new(1.0e-5),
        });
        validate_materials(&[damped], &[]).expect("positive Rayleigh damping should be valid");
    }

    #[test]
    fn accepts_partial_capabilities_and_rejects_an_empty_material() {
        let mut density_only = material();
        density_only.elasticity = None;
        density_only.damping = None;
        density_only.thermal = None;
        density_only.acoustic = None;
        density_only.electrical = None;
        density_only.plastic = None;
        validate_materials(&[density_only.clone()], &[])
            .expect("a density-only material is meaningful for compatible formulations");

        density_only.mass_density_kg_per_m3 = (None).map(runmat_types::physical::MassDensity::new);
        assert!(matches!(
            validate_materials(&[density_only], &[]),
            Err(AnalysisValidationError::InvalidMaterial { detail, .. })
                if detail.contains("at least one constitutive capability")
        ));
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
