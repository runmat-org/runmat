use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::quantity::{RayleighMassCoefficient, RayleighStiffnessCoefficient};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConductivityFrequencyPoint {
    pub frequency_hz: f64,
    pub conductivity_scale: f64,
    #[serde(default)]
    pub dispersive_loss_scale: Option<f64>,
    #[serde(default)]
    pub relative_permittivity_scale: Option<f64>,
    #[serde(default)]
    pub relative_permeability_scale: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialMechanicalModel {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MechanicalDampingModel {
    Undamped,
    Rayleigh {
        mass_coefficient: RayleighMassCoefficient,
        stiffness_coefficient: RayleighStiffnessCoefficient,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialThermalModel {
    pub reference_temperature_k: f64,
    pub modulus_temp_coeff_per_k: f64,
    pub conductivity_w_per_mk: f64,
    pub specific_heat_j_per_kgk: f64,
    pub expansion_coefficient_per_k: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialAcousticModel {
    pub speed_of_sound_m_per_s: f64,
    pub damping_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialElectricalModel {
    pub reference_temperature_k: f64,
    pub conductivity_s_per_m: f64,
    pub resistive_heating_coefficient: f64,
    pub relative_permittivity: f64,
    pub relative_permeability: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conductivity_frequency_response: Vec<ConductivityFrequencyPoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialPlasticModel {
    pub yield_strain: f64,
    pub hardening_modulus_ratio: f64,
    pub saturation_exponent: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialModel {
    pub material_id: String,
    pub name: String,
    #[serde(default)]
    pub mass_density_kg_per_m3: Option<f64>,
    #[serde(default)]
    pub mechanical: Option<MaterialMechanicalModel>,
    #[serde(default)]
    pub damping: Option<MechanicalDampingModel>,
    #[serde(default)]
    pub thermal: Option<MaterialThermalModel>,
    #[serde(default)]
    pub acoustic: Option<MaterialAcousticModel>,
    #[serde(default)]
    pub electrical: Option<MaterialElectricalModel>,
    #[serde(default)]
    pub plastic: Option<MaterialPlasticModel>,
}

/// Closed constitutive facts that a formulation may require from an assigned material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialCapability {
    MassDensity,
    LinearElasticity,
    MechanicalDamping,
    Thermal,
    Acoustic,
    Electrical,
    Plasticity,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("material `{material_id}` does not declare required `{capability:?}` capability")]
pub struct MissingMaterialCapability {
    pub material_id: String,
    pub capability: MaterialCapability,
}

impl MaterialModel {
    pub fn has_capability(&self, capability: MaterialCapability) -> bool {
        match capability {
            MaterialCapability::MassDensity => self.mass_density_kg_per_m3.is_some(),
            MaterialCapability::LinearElasticity => self.mechanical.is_some(),
            MaterialCapability::MechanicalDamping => self.damping.is_some(),
            MaterialCapability::Thermal => self.thermal.is_some(),
            MaterialCapability::Acoustic => self.acoustic.is_some(),
            MaterialCapability::Electrical => self.electrical.is_some(),
            MaterialCapability::Plasticity => self.plastic.is_some(),
        }
    }

    pub fn require_mass_density(&self) -> Result<f64, MissingMaterialCapability> {
        self.mass_density_kg_per_m3
            .ok_or_else(|| self.missing(MaterialCapability::MassDensity))
    }

    pub fn require_mechanical(
        &self,
    ) -> Result<&MaterialMechanicalModel, MissingMaterialCapability> {
        self.mechanical
            .as_ref()
            .ok_or_else(|| self.missing(MaterialCapability::LinearElasticity))
    }

    pub fn damping_or_undamped(&self) -> MechanicalDampingModel {
        self.damping.unwrap_or(MechanicalDampingModel::Undamped)
    }

    pub fn require_thermal(&self) -> Result<&MaterialThermalModel, MissingMaterialCapability> {
        self.thermal
            .as_ref()
            .ok_or_else(|| self.missing(MaterialCapability::Thermal))
    }

    pub fn require_acoustic(&self) -> Result<&MaterialAcousticModel, MissingMaterialCapability> {
        self.acoustic
            .as_ref()
            .ok_or_else(|| self.missing(MaterialCapability::Acoustic))
    }

    pub fn require_electrical(
        &self,
    ) -> Result<&MaterialElectricalModel, MissingMaterialCapability> {
        self.electrical
            .as_ref()
            .ok_or_else(|| self.missing(MaterialCapability::Electrical))
    }

    pub fn require_plastic(&self) -> Result<&MaterialPlasticModel, MissingMaterialCapability> {
        self.plastic
            .as_ref()
            .ok_or_else(|| self.missing(MaterialCapability::Plasticity))
    }

    fn missing(&self, capability: MaterialCapability) -> MissingMaterialCapability {
        MissingMaterialCapability {
            material_id: self.material_id.clone(),
            capability,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn density_only_material() -> MaterialModel {
        MaterialModel {
            material_id: "fluid".into(),
            name: "Fluid".into(),
            mass_density_kg_per_m3: Some(997.0),
            mechanical: None,
            damping: None,
            thermal: None,
            acoustic: None,
            electrical: None,
            plastic: None,
        }
    }

    #[test]
    fn capability_queries_do_not_invent_constitutive_data() {
        let material = density_only_material();
        assert!(material.has_capability(MaterialCapability::MassDensity));
        assert!(!material.has_capability(MaterialCapability::LinearElasticity));
        assert_eq!(material.require_mass_density(), Ok(997.0));
        assert_eq!(
            material.require_mechanical(),
            Err(MissingMaterialCapability {
                material_id: "fluid".into(),
                capability: MaterialCapability::LinearElasticity,
            })
        );
    }

    #[test]
    fn absent_damping_has_explicit_undamped_semantics() {
        let material = density_only_material();
        assert_eq!(
            material.damping_or_undamped(),
            MechanicalDampingModel::Undamped
        );
        assert!(!material.has_capability(MaterialCapability::MechanicalDamping));
    }
}
