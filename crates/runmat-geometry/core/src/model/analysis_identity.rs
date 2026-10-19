use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const PERSISTENT_SELECTION_PREFIX: &str = "persistent-entity:sha256:";

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

    /// Returns the canonical analysis selector for this exact geometry occurrence.
    ///
    /// Unlike `source_topology_id`, this identity includes the entity kind and assembly path, so
    /// repeated occurrences of the same source part cannot alias one another in solver boundary
    /// selection.
    pub fn analysis_selection_id(&self) -> Result<String, GeometryContractError> {
        self.validate()?;
        let mut digest = Sha256::new();
        digest.update(b"runmat.geometry.persistent-analysis-selection/v1\0");
        digest.update([persistent_kind_tag(self.kind)]);
        update_digest_token(&mut digest, &self.source_topology_id);
        digest.update((self.assembly_path.len() as u64).to_be_bytes());
        for segment in &self.assembly_path {
            update_digest_token(&mut digest, segment);
        }
        let digest = digest.finalize();
        let mut selection = String::with_capacity(PERSISTENT_SELECTION_PREFIX.len() + 64);
        selection.push_str(PERSISTENT_SELECTION_PREFIX);
        for byte in digest {
            use std::fmt::Write as _;
            write!(&mut selection, "{byte:02x}").expect("writing to a String cannot fail");
        }
        Ok(selection)
    }

    /// Tests an analysis region selector against this exact entity.
    ///
    /// Canonical persistent selectors are occurrence-qualified. Plain authored selectors retain
    /// their established source-topology matching semantics.
    pub fn matches_analysis_selection(&self, selection_id: &str) -> bool {
        if selection_id.starts_with(PERSISTENT_SELECTION_PREFIX) {
            self.analysis_selection_id().as_deref() == Ok(selection_id)
        } else {
            self.source_topology_id == selection_id
        }
    }
}

fn update_digest_token(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value.as_bytes());
}

const fn persistent_kind_tag(kind: PersistentEntityKind) -> u8 {
    match kind {
        PersistentEntityKind::Assembly => 0,
        PersistentEntityKind::Instance => 1,
        PersistentEntityKind::Body => 2,
        PersistentEntityKind::Lump => 3,
        PersistentEntityKind::Solid => 4,
        PersistentEntityKind::Shell => 5,
        PersistentEntityKind::Face => 6,
        PersistentEntityKind::Wire => 7,
        PersistentEntityKind::Coedge => 8,
        PersistentEntityKind::Edge => 9,
        PersistentEntityKind::Vertex => 10,
        PersistentEntityKind::Region => 11,
        PersistentEntityKind::Contact => 12,
    }
}

pub(crate) fn validate_token(
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

    #[test]
    fn analysis_selection_identity_qualifies_repeated_occurrences() {
        let first = PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: "shared-part-face".into(),
            assembly_path: vec!["root".into(), "instance-a".into()],
        };
        let second = PersistentEntityId {
            assembly_path: vec!["root".into(), "instance-b".into()],
            ..first.clone()
        };

        let first_selection = first.analysis_selection_id().unwrap();
        let second_selection = second.analysis_selection_id().unwrap();
        assert_ne!(first_selection, second_selection);
        assert!(first.matches_analysis_selection(&first_selection));
        assert!(!second.matches_analysis_selection(&first_selection));
        assert!(first.matches_analysis_selection("shared-part-face"));
        assert!(second.matches_analysis_selection("shared-part-face"));
    }
}
