use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometryContractError {
    pub field: String,
    pub reason: String,
}

impl GeometryContractError {
    pub fn invalid(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for GeometryContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for GeometryContractError {}

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
    pub fn validate(&self) -> Result<(), GeometryContractError> {
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
                return Err(GeometryContractError::invalid(
                    field,
                    "must be non-negative",
                ));
            }
        }
        if self.requested_deviation_m == 0.0 {
            return Err(GeometryContractError::invalid(
                "requested_deviation_m",
                "must be greater than zero",
            ));
        }
        Ok(())
    }

    pub fn equivalence_tolerance_m(
        &self,
        model_scale_m: f64,
    ) -> Result<f64, GeometryContractError> {
        self.validate()?;
        validate_finite("model_scale_m", model_scale_m)?;
        if model_scale_m < 0.0 {
            return Err(GeometryContractError::invalid(
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PersistentEntityKind {
    Assembly,
    Instance,
    Body,
    Lump,
    Solid,
    Shell,
    Face,
    Wire,
    Coedge,
    Edge,
    Vertex,
    Region,
    Contact,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistentEntityId {
    pub kind: PersistentEntityKind,
    pub source_topology_id: String,
    #[serde(default)]
    pub assembly_path: Vec<String>,
}

impl PersistentEntityId {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        validate_token(
            "persistent entity source topology id",
            &self.source_topology_id,
            512,
        )?;
        if self.assembly_path.len() > 256 {
            return Err(GeometryContractError::invalid(
                "persistent entity assembly path",
                "must contain at most 256 segments",
            ));
        }
        for segment in &self.assembly_path {
            validate_token("persistent entity assembly path segment", segment, 256)?;
        }
        Ok(())
    }
}

fn validate_token(
    field: &str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), GeometryContractError> {
    if value.is_empty()
        || value.len() > maximum_bytes
        || !value.is_ascii()
        || value.chars().any(char::is_control)
        || value.trim() != value
    {
        return Err(GeometryContractError::invalid(
            field,
            format!("must be 1..={maximum_bytes} printable ASCII bytes without surrounding space"),
        ));
    }
    Ok(())
}

fn validate_finite(field: &str, value: f64) -> Result<(), GeometryContractError> {
    if !value.is_finite() {
        return Err(GeometryContractError::invalid(field, "must be finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_is_geometry_owned_and_scale_aware() {
        let policy = GeometryTolerancePolicy {
            source_tolerance_m: 1.0e-8,
            absolute_floor_m: 1.0e-10,
            model_relative_term: 1.0e-6,
            requested_deviation_m: 1.0e-4,
            maximum_healing_displacement_m: 1.0e-5,
        };
        assert_eq!(policy.equivalence_tolerance_m(2.0).unwrap(), 2.0e-6);
        assert!(policy.equivalence_tolerance_m(f64::NAN).is_err());
    }

    #[test]
    fn persistent_identity_rejects_ambiguous_tokens() {
        let valid = PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: "step-label:42".into(),
            assembly_path: vec!["root".into(), "instance-7".into()],
        };
        valid.validate().unwrap();
        assert!(PersistentEntityId {
            source_topology_id: " step-label:42".into(),
            ..valid
        }
        .validate()
        .is_err());
    }
}
