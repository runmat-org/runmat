use serde::{Deserialize, Serialize};

use super::{invalid, token, Ap242ExchangeInventory, Ap242Measure};
use crate::GeometryContractError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ap242EngineeringPropertyKind {
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
pub struct Ap242EngineeringProperty {
    pub property_id: String,
    pub kind: Ap242EngineeringPropertyKind,
    pub name: String,
    #[serde(with = "runmat_types::wire_integer")]
    pub source_entity_id: u64,
    #[serde(with = "runmat_types::wire_integer")]
    pub representation_entity_id: u64,
    pub value: Ap242Measure,
    pub specification: Option<String>,
    pub environment: Vec<Ap242EnvironmentCondition>,
}

impl Ap242EngineeringProperty {
    pub(super) fn validate(&self) -> Result<(), GeometryContractError> {
        token("AP242 property id", &self.property_id)?;
        token("AP242 property name", &self.name)?;
        if self.source_entity_id == 0 || self.representation_entity_id == 0 {
            return Err(invalid(
                "AP242 property source",
                "source identities must be nonzero",
            ));
        }
        self.value.validate()?;
        let expected = match self.kind {
            Ap242EngineeringPropertyKind::MassDensity => super::QuantityKind::MassDensity,
            Ap242EngineeringPropertyKind::YoungsModulus => super::QuantityKind::Pressure,
            Ap242EngineeringPropertyKind::PoissonRatio
            | Ap242EngineeringPropertyKind::RelativePermittivity
            | Ap242EngineeringPropertyKind::RelativePermeability => {
                super::QuantityKind::Dimensionless
            }
            Ap242EngineeringPropertyKind::ThermalConductivity => {
                super::QuantityKind::ThermalConductivity
            }
            Ap242EngineeringPropertyKind::SpecificHeat => super::QuantityKind::SpecificHeat,
            Ap242EngineeringPropertyKind::ThermalExpansionCoefficient => {
                super::QuantityKind::InverseTemperature
            }
            Ap242EngineeringPropertyKind::ElectricalConductivity => {
                super::QuantityKind::ElectricalConductivity
            }
            Ap242EngineeringPropertyKind::AcousticImpedance => {
                super::QuantityKind::SpecificAcousticImpedance
            }
            Ap242EngineeringPropertyKind::DynamicViscosity => super::QuantityKind::DynamicViscosity,
        };
        if self.value.kind() != expected {
            return Err(invalid(
                "AP242 property dimension",
                "measure dimension is incompatible with the engineering-property kind",
            ));
        }
        if let Some(specification) = &self.specification {
            token("AP242 property specification", specification)?;
        }
        for condition in &self.environment {
            condition.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ap242EnvironmentCondition {
    pub name: String,
    pub value: Ap242Measure,
}

impl Ap242EnvironmentCondition {
    fn validate(&self) -> Result<(), GeometryContractError> {
        token("AP242 environment condition", &self.name)?;
        self.value.validate().map_err(Into::into)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ap242MaterialClass {
    Physical,
    Visual,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ap242MaterialCandidate {
    pub material_id: String,
    pub name: String,
    #[serde(with = "runmat_types::wire_integer")]
    pub source_entity_id: u64,
    pub persistent_id: Option<String>,
    pub class: Ap242MaterialClass,
    pub property_ids: Vec<String>,
    pub specification: Option<String>,
}

impl Ap242MaterialCandidate {
    pub(super) fn validate(
        &self,
        inventory: &Ap242ExchangeInventory,
    ) -> Result<(), GeometryContractError> {
        token("AP242 material id", &self.material_id)?;
        token("AP242 material name", &self.name)?;
        if self.source_entity_id == 0 {
            return Err(invalid(
                "AP242 material source",
                "source identity must be nonzero",
            ));
        }
        for property_id in &self.property_ids {
            if !inventory
                .engineering_properties
                .iter()
                .any(|property| property.property_id == *property_id)
            {
                return Err(invalid(
                    "AP242 material property",
                    "referenced property does not exist",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum Ap242AssociationTarget {
    ProductEntity(#[serde(with = "runmat_types::wire_integer")] u64),
    ProductDefinitionEntity(#[serde(with = "runmat_types::wire_integer")] u64),
    OccurrenceEntity(#[serde(with = "runmat_types::wire_integer")] u64),
    ShapeAspectEntity(#[serde(with = "runmat_types::wire_integer")] u64),
    PersistentId(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ap242Association {
    pub association_id: String,
    #[serde(with = "runmat_types::wire_integer")]
    pub source_entity_id: u64,
    pub material_id: String,
    pub target: Ap242AssociationTarget,
}

impl Ap242Association {
    pub(super) fn validate(
        &self,
        inventory: &Ap242ExchangeInventory,
    ) -> Result<(), GeometryContractError> {
        token("AP242 association id", &self.association_id)?;
        if self.source_entity_id == 0 {
            return Err(invalid(
                "AP242 association source",
                "source identity must be nonzero",
            ));
        }
        if !inventory
            .materials
            .iter()
            .any(|material| material.material_id == self.material_id)
        {
            return Err(invalid(
                "AP242 association material",
                "referenced material does not exist",
            ));
        }
        if let Ap242AssociationTarget::PersistentId(value) = &self.target {
            token("AP242 association persistent id", value)?;
        }
        Ok(())
    }
}
