mod storage;

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
    /// Inelastic non-Newtonian fluid: viscosity a function of the local shear rate.
    GeneralizedNewtonian {
        law: super::rheology::GeneralizedNewtonianLaw,
    },
    /// A porous zone the surrounding fluid flows through (filter media, packed bed): its
    /// resistance, not a second fluid.
    PorousMedium { law: PorousMediumLaw },
}

/// Darcy–Forchheimer resistance of a porous zone, superficial-velocity form: the momentum
/// sink −(μ/K) u − ½ ρ C₂ |u| u (Darcy when C₂ = 0; with the Navier–Stokes viscous term this is
/// the Brinkman–Forchheimer extended model).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PorousMediumLaw {
    pub permeability_m2: f64,
    pub inertial_coefficient_per_m: f64,
    pub porosity: f64,
}

impl PorousMediumLaw {
    pub fn is_valid(&self) -> bool {
        self.permeability_m2.is_finite()
            && self.permeability_m2 > 0.0
            && self.inertial_coefficient_per_m.is_finite()
            && self.inertial_coefficient_per_m >= 0.0
            && self.porosity.is_finite()
            && self.porosity > 0.0
            && self.porosity <= 1.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IncompressibleNewtonianMaterial {
    pub density: crate::MassDensity,
    pub dynamic_viscosity: crate::DynamicViscosity,
}

/// An incompressible fluid's coefficients: density, reference (low-shear) viscosity and, for a
/// generalized-Newtonian fluid, its viscosity law.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IncompressibleFluidMaterial {
    pub density: crate::MassDensity,
    pub dynamic_viscosity: crate::DynamicViscosity,
    pub rheology: Option<super::rheology::GeneralizedNewtonianLaw>,
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
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub frequency_hz:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Frequency, f64>,
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
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub youngs_modulus_pa: runmat_types::physical::YoungsModulus,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub poisson_ratio: runmat_types::physical::PoissonRatio,
}

/// Orthotropic plane-stress and transverse-shear law for one shell laminate ply.
///
/// Axes 1 and 2 are the ply material axes. The 13 and 23 shear moduli govern transverse shear in
/// first-order shear-deformation shells; this is deliberately not a generic three-dimensional
/// orthotropic solid law.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialOrthotropicLaminaModel {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub longitudinal_modulus_pa: runmat_types::physical::YoungsModulus,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub transverse_modulus_pa: runmat_types::physical::YoungsModulus,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub major_poisson_ratio: runmat_types::physical::PoissonRatio,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub in_plane_shear_modulus_pa: runmat_types::physical::ShearModulus,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub transverse_shear_modulus_13_pa: runmat_types::physical::ShearModulus,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub transverse_shear_modulus_23_pa: runmat_types::physical::ShearModulus,
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
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub modulus_temp_coeff_per_k: runmat_types::physical::InverseTemperature,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub conductivity_w_per_mk: runmat_types::physical::ThermalConductivity,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub specific_heat_j_per_kgk: runmat_types::physical::SpecificHeat,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialAcousticModel {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub damping_ratio: runmat_types::physical::DampingRatio,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialElectricalModel {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub conductivity_s_per_m: runmat_types::physical::ElectricalConductivity,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub resistive_heating_coefficient: runmat_types::physical::InverseTemperature,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub relative_permittivity: runmat_types::physical::RelativePermittivity,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub relative_permeability: runmat_types::physical::RelativePermeability,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conductivity_frequency_response: Vec<ConductivityFrequencyPoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "law", rename_all = "snake_case", deny_unknown_fields)]
pub enum MaterialPlasticModel {
    J2IsotropicLinearHardening {
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        yield_stress_pa: runmat_types::physical::YieldStress,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        /// Yield-stress increase per unit equivalent plastic strain, not a uniaxial tangent.
        isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialModel {
    pub material_id: String,
    pub name: String,
    pub provenance: MaterialProvenance,
    #[serde(with = "runmat_types::physical::canonical_scalar::optional")]
    #[serde(default)]
    pub mass_density_kg_per_m3: Option<runmat_types::physical::MassDensity>,
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
    #[serde(with = "runmat_types::physical::canonical_scalar::optional")]
    pub mass_density_kg_per_m3: Option<runmat_types::physical::MassDensity>,
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
    #[serde(with = "runmat_types::wire_integer")]
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
    #[serde(with = "runmat_types::wire_integer")]
    pub source_entity_id: u64,
    pub target: ImportedCadMaterialAssociationTarget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum ImportedCadMaterialAssociationTarget {
    ProductEntity(#[serde(with = "runmat_types::wire_integer")] u64),
    ProductDefinitionEntity(#[serde(with = "runmat_types::wire_integer")] u64),
    OccurrenceEntity(#[serde(with = "runmat_types::wire_integer")] u64),
    ShapeAspectEntity(#[serde(with = "runmat_types::wire_integer")] u64),
    PersistentId(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedCadMaterialProperty {
    pub property: ImportedCadMaterialPropertyKind,
    pub source_property_id: String,
    pub source_name: String,
    #[serde(with = "runmat_types::wire_integer")]
    pub source_entity_id: u64,
    #[serde(with = "runmat_types::wire_integer")]
    pub representation_entity_id: u64,
    pub value: runmat_types::physical::SourceQuantity,
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
    pub value: runmat_types::physical::SourceQuantity,
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
            MaterialCapability::MassDensity => self
                .mass_density_kg_per_m3
                .map(|quantity| quantity.canonical_value())
                .is_some(),
            MaterialCapability::IsotropicLinearElasticity => matches!(
                self.elasticity,
                Some(MaterialElasticityModel::IsotropicLinear(_))
            ),
            MaterialCapability::LaminaElasticity => self.elasticity.is_some(),
            MaterialCapability::MechanicalDamping => self.damping.is_some(),
            MaterialCapability::Thermal => self.thermal.is_some(),
            MaterialCapability::Acoustic => self.acoustic.is_some(),
            // The incompressible-fluid capability: Newtonian or generalized-Newtonian.
            MaterialCapability::IncompressibleNewtonianFluid => {
                self.require_incompressible_fluid().is_ok()
            }
            MaterialCapability::Electrical => self.electrical.is_some(),
            MaterialCapability::Plasticity => self.plastic.is_some(),
        }
    }

    pub fn require_mass_density(
        &self,
    ) -> Result<runmat_types::physical::MassDensity, MissingMaterialCapability> {
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
    /// Any incompressible fluid law: Newtonian, or generalized-Newtonian with its law and
    /// low-shear reference viscosity.
    pub fn require_incompressible_fluid(
        &self,
    ) -> Result<IncompressibleFluidMaterial, FluidMaterialError> {
        let fluid = self
            .fluid
            .ok_or_else(|| self.missing(MaterialCapability::IncompressibleNewtonianFluid))?;
        if let MaterialFluidModel::PorousMedium { .. } = fluid {
            return Err(FluidMaterialError::Invalid {
                material_id: self.material_id.clone(),
                property: "fluid law (a porous zone is not a flowing fluid)",
            });
        }
        if let MaterialFluidModel::GeneralizedNewtonian { law } = fluid {
            let density = self.require_mass_density()?;
            let reference = law.low_shear_viscosity();
            if !law.is_valid()
                || !density.is_finite()
                || density.canonical_value() <= 0.0
                || !reference.is_finite()
                || reference <= 0.0
            {
                return Err(FluidMaterialError::Invalid {
                    material_id: self.material_id.clone(),
                    property: "viscosity law",
                });
            }
            return Ok(IncompressibleFluidMaterial {
                density,
                dynamic_viscosity: crate::DynamicViscosity::new(reference),
                rheology: Some(law),
            });
        }
        let newtonian = self.require_incompressible_newtonian()?;
        Ok(IncompressibleFluidMaterial {
            density: newtonian.density,
            dynamic_viscosity: newtonian.dynamic_viscosity,
            rheology: None,
        })
    }

    pub fn require_incompressible_newtonian(
        &self,
    ) -> Result<IncompressibleNewtonianMaterial, FluidMaterialError> {
        let Some(MaterialFluidModel::IncompressibleNewtonian { dynamic_viscosity }) = self.fluid
        else {
            return Err(self
                .missing(MaterialCapability::IncompressibleNewtonianFluid)
                .into());
        };
        let density = self.require_mass_density()?;
        for (property, value) in [
            ("density", density.canonical_value()),
            ("dynamic viscosity", dynamic_viscosity.canonical_value()),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(FluidMaterialError::Invalid {
                    material_id: self.material_id.clone(),
                    property,
                });
            }
        }
        Ok(IncompressibleNewtonianMaterial {
            density,
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

    pub(super) fn density_only_material() -> MaterialModel {
        MaterialModel {
            fluid: None,
            material_id: "fluid".into(),
            name: "Fluid".into(),
            provenance: MaterialProvenance::Authored,
            mass_density_kg_per_m3: Some(runmat_types::physical::MassDensity::new(997.0)),
            elasticity: None,
            damping: None,
            thermal: None,
            acoustic: None,
            electrical: None,
            plastic: None,
        }
    }

    #[test]
    fn retained_material_buffers_follow_capacity_without_changing_semantics() {
        let mut material = density_only_material();
        let expected = material.material_id.capacity() + material.name.capacity();
        assert_eq!(material.owned_buffer_bytes(), Some(expected as u64));
        let original = material.clone();
        material.name.reserve(8192);
        assert_eq!(material, original);
        assert_eq!(
            material.owned_buffer_bytes(),
            Some((material.material_id.capacity() + material.name.capacity()) as u64)
        );
        assert!(material.owned_buffer_bytes().unwrap() > original.owned_buffer_bytes().unwrap());
    }

    #[test]
    fn capability_queries_do_not_invent_constitutive_data() {
        let material = density_only_material();
        assert!(material.has_capability(MaterialCapability::MassDensity));
        assert!(!material.has_capability(MaterialCapability::IsotropicLinearElasticity));
        assert_eq!(
            material.require_mass_density(),
            Ok(crate::MassDensity::new(997.0))
        );
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
            invalid.mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(bad));
            assert!(invalid.require_incompressible_newtonian().is_err());
            assert!(crate::validate_material_definitions(&[invalid]).is_err());
            let mut invalid = material.clone();
            invalid.fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
                dynamic_viscosity: crate::DynamicViscosity::new(bad),
            });
            assert!(invalid.require_incompressible_newtonian().is_err());
            assert!(crate::validate_material_definitions(&[invalid]).is_err());
        }
        material.mass_density_kg_per_m3 = (None).map(runmat_types::physical::MassDensity::new);
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
