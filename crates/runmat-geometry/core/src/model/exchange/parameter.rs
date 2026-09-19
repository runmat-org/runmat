use serde::{Deserialize, Serialize};

use super::{finite, invalid, token};
use crate::GeometryContractError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ap242PhysicalDimension {
    Dimensionless,
    Length,
    PlaneAngle,
    Area,
    Volume,
    Time,
    Mass,
    MassDensity,
    Temperature,
    Pressure,
    Force,
    Energy,
    Power,
    ThermalConductivity,
    SpecificHeat,
    InverseTemperature,
    ElectricalConductivity,
    AcousticImpedance,
    DynamicViscosity,
    ElectricPotential,
    ElectricCurrent,
    MagneticFluxDensity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ap242Measure {
    pub dimension: Ap242PhysicalDimension,
    pub value_si: f64,
    pub source_value: f64,
    pub source_unit: String,
}

impl Ap242Measure {
    pub(super) fn validate(&self) -> Result<(), GeometryContractError> {
        finite("AP242 SI value", self.value_si)?;
        finite("AP242 source value", self.source_value)?;
        token("AP242 source unit", &self.source_unit)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ap242ParameterAccess {
    ReadOnly,
    Regenerable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ap242Parameter {
    pub parameter_id: String,
    pub name: String,
    pub source_entity_id: u64,
    pub representation_entity_id: u64,
    pub owner_entity_ids: Vec<u64>,
    pub persistent_id: Option<String>,
    pub value: Ap242Measure,
    pub minimum: Option<Ap242Measure>,
    pub maximum: Option<Ap242Measure>,
    pub access: Ap242ParameterAccess,
}

impl Ap242Parameter {
    pub(crate) fn validate(&self) -> Result<(), GeometryContractError> {
        token("AP242 parameter id", &self.parameter_id)?;
        token("AP242 parameter name", &self.name)?;
        if self.source_entity_id == 0
            || self.representation_entity_id == 0
            || self.owner_entity_ids.contains(&0)
        {
            return Err(invalid(
                "AP242 parameter source",
                "source identities must be nonzero",
            ));
        }
        if let Some(persistent_id) = &self.persistent_id {
            token("AP242 parameter persistent id", persistent_id)?;
        }
        self.value.validate()?;
        for bound in [&self.minimum, &self.maximum].into_iter().flatten() {
            bound.validate()?;
            if bound.dimension != self.value.dimension {
                return Err(invalid(
                    "AP242 parameter bounds",
                    "bounds must have the parameter dimension",
                ));
            }
        }
        if let (Some(minimum), Some(maximum)) = (&self.minimum, &self.maximum) {
            if minimum.value_si > maximum.value_si {
                return Err(invalid(
                    "AP242 parameter bounds",
                    "minimum must not exceed maximum",
                ));
            }
        }
        Ok(())
    }
}
