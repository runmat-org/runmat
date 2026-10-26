use serde::{Deserialize, Serialize};

use super::quantity::{DynamicViscosity, Frequency, MassDensity, Temperature, TurbulenceIntensity};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermoMechanicalDomain {
    pub reference_temperature: Temperature,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalDomain {
    pub reference_temperature: Temperature,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectroThermalDomain {
    pub reference_temperature: Temperature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StaticElectricalEquation {
    Electrostatics,
    SteadyConduction,
}

impl StaticElectricalEquation {
    pub const fn constitutive_coefficient_kind(self) -> runmat_types::physical::QuantityKind {
        match self {
            Self::Electrostatics => runmat_types::physical::QuantityKind::ElectricPermittivity,
            Self::SteadyConduction => runmat_types::physical::QuantityKind::ElectricalConductivity,
        }
    }
    pub const fn conjugate_flux_kind(self) -> runmat_types::physical::QuantityKind {
        match self {
            Self::Electrostatics => runmat_types::physical::QuantityKind::ElectricCharge,
            Self::SteadyConduction => runmat_types::physical::QuantityKind::ElectricCurrent,
        }
    }
    pub const fn work_kind(self) -> runmat_types::physical::QuantityKind {
        match self {
            Self::Electrostatics => runmat_types::physical::QuantityKind::Energy,
            Self::SteadyConduction => runmat_types::physical::QuantityKind::Power,
        }
    }
}

/// Magnetostatic unknowns are distinct from scalar electrical potential and from
/// harmonic electromagnetic fields. No frequency is implied by this formulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MagnetostaticFormulation {
    MagneticVectorPotential,
}

/// Static electrical, magnetostatic, and harmonic intent are distinct contracts.
/// Neither static formulation carries a fictitious frequency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ElectromagneticDomain {
    Static {
        equation: StaticElectricalEquation,
    },
    Magnetostatic {
        formulation: MagnetostaticFormulation,
    },
    Harmonic {
        reference_frequency: Frequency,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn electrical_equations_have_distinct_closed_domain_contracts() {
        let domains = [
            ElectromagneticDomain::Static {
                equation: StaticElectricalEquation::Electrostatics,
            },
            ElectromagneticDomain::Static {
                equation: StaticElectricalEquation::SteadyConduction,
            },
            ElectromagneticDomain::Harmonic {
                reference_frequency: Frequency::new(60.0),
            },
            ElectromagneticDomain::Magnetostatic {
                formulation: MagnetostaticFormulation::MagneticVectorPotential,
            },
        ];
        let mut encodings = std::collections::BTreeSet::new();
        for domain in domains {
            let bytes = serde_json::to_vec(&domain).unwrap();
            assert!(encodings.insert(bytes.clone()));
            assert_eq!(
                serde_json::from_slice::<ElectromagneticDomain>(&bytes).unwrap(),
                domain
            );
        }
        assert!(serde_json::from_str::<ElectromagneticDomain>(
            r#"{"static":{"equation":"steady_conduction","reference_frequency":{"hertz":60.0}}}"#
        )
        .is_err());
        assert!(serde_json::from_str::<ElectromagneticDomain>(
            r#"{"static":{"equation":"unknown"}}"#
        )
        .is_err());
        assert!(serde_json::from_str::<ElectromagneticDomain>(
            r#"{"reference_frequency":{"hertz":60.0}}"#
        )
        .is_err());
        for invalid in [
            r#"{"magnetostatic":{"formulation":"magnetic_vector_potential","reference_frequency":{"hertz":0.0}}}"#,
            r#"{"magnetostatic":{"formulation":"unknown"}}"#,
            r#"{"magnetostatic":{}}"#,
        ] {
            assert!(serde_json::from_str::<ElectromagneticDomain>(invalid).is_err());
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CfdDomain {
    pub density: MassDensity,
    /// Newtonian viscosity, or the low-shear reference viscosity of a generalized law.
    pub dynamic_viscosity: DynamicViscosity,
    pub turbulence_intensity: TurbulenceIntensity,
    /// Shear-rate dependent viscosity; absent for a Newtonian fluid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rheology: Option<super::rheology::GeneralizedNewtonianLaw>,
    /// Porous zones of the flow: the cell regions each resistance law applies to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub porous_zones: Vec<PorousZone>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PorousZone {
    /// Sorted, unique cell regions.
    pub regions: Vec<runmat_geometry_core::PersistentEntityId>,
    pub law: super::materials::PorousMediumLaw,
}
