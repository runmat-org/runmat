use serde::{Deserialize, Serialize};

use super::{validate_finite, MeshingContractError};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryTolerancePolicy {
    pub source_tolerance_m: f64,
    pub absolute_floor_m: f64,
    pub model_relative_term: f64,
    pub requested_deviation_m: f64,
    pub maximum_healing_displacement_m: f64,
}

impl GeometryTolerancePolicy {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        for (field, value) in [
            ("source_tolerance_m", self.source_tolerance_m),
            ("absolute_floor_m", self.absolute_floor_m),
            ("model_relative_term", self.model_relative_term),
            ("requested_deviation_m", self.requested_deviation_m),
            (
                "maximum_healing_displacement_m",
                self.maximum_healing_displacement_m,
            ),
        ] {
            validate_finite(field, value)?;
            if value < 0.0 {
                return Err(MeshingContractError::invalid(field, "must be non-negative"));
            }
        }
        if self.requested_deviation_m == 0.0 {
            return Err(MeshingContractError::invalid(
                "requested_deviation_m",
                "must be greater than zero",
            ));
        }
        Ok(())
    }

    pub fn equivalence_tolerance_m(&self, model_scale_m: f64) -> Result<f64, MeshingContractError> {
        self.validate()?;
        validate_finite("model_scale_m", model_scale_m)?;
        if model_scale_m < 0.0 {
            return Err(MeshingContractError::invalid(
                "model_scale_m",
                "must be non-negative",
            ));
        }
        Ok(self
            .source_tolerance_m
            .max(self.absolute_floor_m)
            .max(self.model_relative_term * model_scale_m))
    }
}
