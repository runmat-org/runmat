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

/// Static scalar-potential intent and harmonic electromagnetic intent are distinct
/// physical contracts. Static analysis does not carry a fictitious frequency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ElectromagneticDomain {
    Static { equation: StaticElectricalEquation },
    Harmonic { reference_frequency: Frequency },
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
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CfdDomain {
    pub density: MassDensity,
    pub dynamic_viscosity: DynamicViscosity,
    pub turbulence_intensity: TurbulenceIntensity,
}
