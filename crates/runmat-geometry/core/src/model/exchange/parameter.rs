use serde::{Deserialize, Serialize};

use super::{invalid, token};
use crate::GeometryContractError;

pub use runmat_types::physical::QuantityKind;
pub type Ap242Measure = runmat_types::physical::SourceQuantity;

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
    #[serde(with = "runmat_types::wire_integer")]
    pub source_entity_id: u64,
    #[serde(with = "runmat_types::wire_integer")]
    pub representation_entity_id: u64,
    #[serde(with = "runmat_types::wire_integer::vec")]
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
            if bound.kind() != self.value.kind() {
                return Err(invalid(
                    "AP242 parameter bounds",
                    "bounds must have the parameter dimension",
                ));
            }
        }
        if let (Some(minimum), Some(maximum)) = (&self.minimum, &self.maximum) {
            if minimum.canonical_value() > maximum.canonical_value() {
                return Err(invalid(
                    "AP242 parameter bounds",
                    "minimum must not exceed maximum",
                ));
            }
        }
        Ok(())
    }
}
