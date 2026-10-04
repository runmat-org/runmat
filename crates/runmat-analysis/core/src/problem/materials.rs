use runmat_types::PhysicalDimension;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::quantity::{RayleighMassCoefficient, RayleighStiffnessCoefficient};

/// A closed, constant-coefficient fluid law. Density is owned by the shared
/// material, not repeated inside the law or inferred from an acoustic capability.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MaterialFluidModel {
    IncompressibleNewtonian {
        dynamic_viscosity: crate::DynamicViscosity,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IncompressibleNewtonianMaterial {
    pub density: crate::MassDensity,
    pub dynamic_viscosity: crate::DynamicViscosity,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum FluidMaterialError {
    #[error(transparent)]
    Missing(#[from] MissingMaterialCapability),
    #[error("material `{material_id}` requires finite positive Newtonian {property}")]
    Invalid {
        material_id: String,
        property: &'static str,
    },
}

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

/// Orthotropic plane-stress and transverse-shear law for one shell laminate ply.
///
/// Axes 1 and 2 are the ply material axes. The 13 and 23 shear moduli govern transverse shear in
/// first-order shear-deformation shells; this is deliberately not a generic three-dimensional
/// orthotropic solid law.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialOrthotropicLaminaModel {
    pub longitudinal_modulus_pa: f64,
    pub transverse_modulus_pa: f64,
    pub major_poisson_ratio: f64,
    pub in_plane_shear_modulus_pa: f64,
    pub transverse_shear_modulus_13_pa: f64,
    pub transverse_shear_modulus_23_pa: f64,
}

/// Closed mechanical elasticity law owned by a material.
///
/// A material has at most one elasticity law. Isotropic linear elasticity is also a valid
/// degenerate lamina law; the consuming formulation decides which variants it admits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "properties")]
pub enum MaterialElasticityModel {
    IsotropicLinear(MaterialMechanicalModel),
    OrthotropicLamina(MaterialOrthotropicLaminaModel),
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
#[serde(tag = "law", rename_all = "snake_case", deny_unknown_fields)]
pub enum MaterialPlasticModel {
    J2IsotropicLinearHardening {
        yield_stress_pa: f64,
        /// Yield-stress increase per unit equivalent plastic strain, not a uniaxial tangent.
        isotropic_hardening_modulus_pa: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialModel {
    pub material_id: String,
    pub name: String,
    pub provenance: MaterialProvenance,
    #[serde(default)]
    pub mass_density_kg_per_m3: Option<f64>,
    #[serde(default)]
    pub elasticity: Option<MaterialElasticityModel>,
    #[serde(default)]
    pub damping: Option<MechanicalDampingModel>,
    #[serde(default)]
    pub thermal: Option<MaterialThermalModel>,
    #[serde(default)]
    pub acoustic: Option<MaterialAcousticModel>,
    #[serde(default)]
    pub fluid: Option<MaterialFluidModel>,
    #[serde(default)]
    pub electrical: Option<MaterialElectricalModel>,
    #[serde(default)]
    pub plastic: Option<MaterialPlasticModel>,
}

/// Numerical material semantics without display or importer provenance.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SolverMaterialModel<'a> {
    pub material_id: &'a str,
    pub mass_density_kg_per_m3: Option<f64>,
    pub elasticity: &'a Option<MaterialElasticityModel>,
    pub damping: Option<MechanicalDampingModel>,
    pub thermal: &'a Option<MaterialThermalModel>,
    pub acoustic: &'a Option<MaterialAcousticModel>,
    pub fluid: &'a Option<MaterialFluidModel>,
    pub electrical: &'a Option<MaterialElectricalModel>,
    pub plastic: &'a Option<MaterialPlasticModel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum MaterialProvenance {
    #[default]
    Authored,
    ImportedCad(ImportedCadMaterialProvenance),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedCadMaterialProvenance {
    pub geometry_module_path: String,
    pub geometry_id: String,
    pub source_material_id: String,
    pub source_material_name: String,
    pub source_entity_id: u64,
    pub persistent_id: Option<String>,
    pub specification: Option<String>,
    pub associations: Vec<ImportedCadMaterialAssociation>,
    pub adopted_properties: Vec<ImportedCadMaterialProperty>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedCadMaterialAssociation {
    pub association_id: String,
    pub source_entity_id: u64,
    pub target: ImportedCadMaterialAssociationTarget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum ImportedCadMaterialAssociationTarget {
    ProductEntity(u64),
    ProductDefinitionEntity(u64),
    OccurrenceEntity(u64),
    ShapeAspectEntity(u64),
    PersistentId(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedCadMaterialProperty {
    pub property: ImportedCadMaterialPropertyKind,
    pub source_property_id: String,
    pub source_name: String,
    pub source_entity_id: u64,
    pub representation_entity_id: u64,
    pub dimension: PhysicalDimension,
    pub value_si: f64,
    pub source_value: f64,
    pub source_unit: String,
    pub specification: Option<String>,
    pub environment: Vec<ImportedCadEnvironmentCondition>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportedCadMaterialPropertyKind {
    MassDensity,
    YoungsModulus,
    PoissonRatio,
    ThermalConductivity,
    SpecificHeat,
    ThermalExpansionCoefficient,
    ElectricalConductivity,
    RelativePermittivity,
    RelativePermeability,
    AcousticImpedance,
    DynamicViscosity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedCadEnvironmentCondition {
    pub name: String,
    pub dimension: PhysicalDimension,
    pub value_si: f64,
    pub source_value: f64,
    pub source_unit: String,
}

/// Closed constitutive facts that a formulation may require from an assigned material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialCapability {
    MassDensity,
    IsotropicLinearElasticity,
    LaminaElasticity,
    MechanicalDamping,
    Thermal,
    Acoustic,
    IncompressibleNewtonianFluid,
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
    pub fn solver_identity_contract(&self) -> SolverMaterialModel<'_> {
        SolverMaterialModel {
            material_id: &self.material_id,
            mass_density_kg_per_m3: self.mass_density_kg_per_m3,
            elasticity: &self.elasticity,
            damping: self.damping,
            thermal: &self.thermal,
            acoustic: &self.acoustic,
            fluid: &self.fluid,
            electrical: &self.electrical,
            plastic: &self.plastic,
        }
    }

    pub fn has_capability(&self, capability: MaterialCapability) -> bool {
        match capability {
            MaterialCapability::MassDensity => self.mass_density_kg_per_m3.is_some(),
            MaterialCapability::IsotropicLinearElasticity => matches!(
                self.elasticity,
                Some(MaterialElasticityModel::IsotropicLinear(_))
            ),
            MaterialCapability::LaminaElasticity => self.elasticity.is_some(),
            MaterialCapability::MechanicalDamping => self.damping.is_some(),
            MaterialCapability::Thermal => self.thermal.is_some(),
            MaterialCapability::Acoustic => self.acoustic.is_some(),
            MaterialCapability::IncompressibleNewtonianFluid => {
                self.require_incompressible_newtonian().is_ok()
            }
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
        match self.elasticity.as_ref() {
            Some(MaterialElasticityModel::IsotropicLinear(value)) => Ok(value),
            _ => Err(self.missing(MaterialCapability::IsotropicLinearElasticity)),
        }
    }

    pub fn require_mechanical_mut(
        &mut self,
    ) -> Result<&mut MaterialMechanicalModel, MissingMaterialCapability> {
        if !matches!(
            self.elasticity,
            Some(MaterialElasticityModel::IsotropicLinear(_))
        ) {
            return Err(self.missing(MaterialCapability::IsotropicLinearElasticity));
        }
        match self.elasticity.as_mut() {
            Some(MaterialElasticityModel::IsotropicLinear(value)) => Ok(value),
            _ => unreachable!("isotropic elasticity was checked above"),
        }
    }

    pub fn require_lamina_elasticity(
        &self,
    ) -> Result<&MaterialElasticityModel, MissingMaterialCapability> {
        self.elasticity
            .as_ref()
            .ok_or_else(|| self.missing(MaterialCapability::LaminaElasticity))
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

    /// Typed physical coefficients from their sole shared material authorities.
    /// This checks positivity even when called before whole-model validation.
    pub fn require_incompressible_newtonian(
        &self,
    ) -> Result<IncompressibleNewtonianMaterial, FluidMaterialError> {
        let MaterialFluidModel::IncompressibleNewtonian { dynamic_viscosity } = self
            .fluid
            .ok_or_else(|| self.missing(MaterialCapability::IncompressibleNewtonianFluid))?;
        let density = self.require_mass_density()?;
        for (property, value) in [
            ("density", density),
            ("dynamic viscosity", dynamic_viscosity.pascal_seconds),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(FluidMaterialError::Invalid {
                    material_id: self.material_id.clone(),
                    property,
                });
            }
        }
        Ok(IncompressibleNewtonianMaterial {
            density: crate::MassDensity::new(density),
            dynamic_viscosity,
        })
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
            fluid: None,
            material_id: "fluid".into(),
            name: "Fluid".into(),
            provenance: MaterialProvenance::Authored,
            mass_density_kg_per_m3: Some(997.0),
            elasticity: None,
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
        assert!(!material.has_capability(MaterialCapability::IsotropicLinearElasticity));
        assert_eq!(material.require_mass_density(), Ok(997.0));
        assert_eq!(
            material.require_mechanical(),
            Err(MissingMaterialCapability {
                material_id: "fluid".into(),
                capability: MaterialCapability::IsotropicLinearElasticity,
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

    #[test]
    fn newtonian_fluid_uses_single_density_and_typed_positive_viscosity() {
        let mut material = density_only_material();
        assert!(material.require_incompressible_newtonian().is_err());
        material.fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
            dynamic_viscosity: crate::DynamicViscosity::new(0.001),
        });
        let coefficients = material.require_incompressible_newtonian().unwrap();
        assert_eq!(coefficients.density, crate::MassDensity::new(997.0));
        assert_eq!(
            coefficients.dynamic_viscosity,
            crate::DynamicViscosity::new(0.001)
        );
        assert!(material.has_capability(MaterialCapability::IncompressibleNewtonianFluid));
        let wire = serde_json::to_value(&material).unwrap();
        assert_eq!(wire["fluid"]["kind"], "incompressible_newtonian");
        assert!(wire["fluid"].get("density").is_none());
        assert_eq!(
            serde_json::from_value::<MaterialModel>(wire).unwrap(),
            material
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut invalid = material.clone();
            invalid.mass_density_kg_per_m3 = Some(bad);
            assert!(invalid.require_incompressible_newtonian().is_err());
            assert!(crate::validate_material_definitions(&[invalid]).is_err());
            let mut invalid = material.clone();
            invalid.fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
                dynamic_viscosity: crate::DynamicViscosity::new(bad),
            });
            assert!(invalid.require_incompressible_newtonian().is_err());
            assert!(crate::validate_material_definitions(&[invalid]).is_err());
        }
        material.mass_density_kg_per_m3 = None;
        assert!(material.require_incompressible_newtonian().is_err());
        assert!(crate::validate_material_definitions(&[material]).is_err());
    }

    #[test]
    fn newtonian_law_is_closed_and_identity_significant() {
        let mut material = density_only_material();
        let missing = serde_json::to_vec(&material.solver_identity_contract()).unwrap();
        material.fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
            dynamic_viscosity: crate::DynamicViscosity::new(0.001),
        });
        let first = serde_json::to_vec(&material.solver_identity_contract()).unwrap();
        assert_ne!(first, missing);
        material.fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
            dynamic_viscosity: crate::DynamicViscosity::new(0.002),
        });
        assert_ne!(
            first,
            serde_json::to_vec(&material.solver_identity_contract()).unwrap()
        );
        for wire in [
            r#"{"kind":"incompressible_newtonian"}"#,
            r#"{"kind":"turbulent","dynamic_viscosity":{"pascal_seconds":0.001}}"#,
            r#"{"kind":"incompressible_newtonian","dynamic_viscosity":{"pascal_seconds":0.001},"density":1000}"#,
        ] {
            assert!(serde_json::from_str::<MaterialFluidModel>(wire).is_err());
        }
    }
}
