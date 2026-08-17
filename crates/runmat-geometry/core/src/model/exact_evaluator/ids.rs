use serde::{Deserialize, Serialize};

use super::super::GeometryContractError;

macro_rules! evaluator_id {
    ($name:ident, $field:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, GeometryContractError> {
                let id = Self(value.into());
                id.validate()?;
                Ok(id)
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn validate(&self) -> Result<(), GeometryContractError> {
                validate_evaluator_id($field, &self.0)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }
    };
}

evaluator_id!(CurveEvaluatorId, "curve evaluator id");
evaluator_id!(PcurveEvaluatorId, "pcurve evaluator id");
evaluator_id!(SurfaceEvaluatorId, "surface evaluator id");
evaluator_id!(TrimClassifierId, "trim classifier id");
evaluator_id!(MassPropertiesEvaluatorId, "mass-properties evaluator id");

fn validate_evaluator_id(field: &str, value: &str) -> Result<(), GeometryContractError> {
    if value.is_empty()
        || value.len() > 512
        || !value.is_ascii()
        || value.chars().any(char::is_control)
        || value.trim() != value
    {
        return Err(GeometryContractError::invalid(
            field,
            "must be 1..=512 printable ASCII bytes without surrounding space",
        ));
    }
    Ok(())
}
