//! Typed immutable artifact roots shared by analysis families.
//!
//! These references locate transport objects. They never establish numerical origin:
//! admission must compare their kind and identities with authenticated publications.
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_analysis_fea::results::{
    field::RecoveredFieldArtifactManifest, observation::ObservationArtifactManifest,
};
use runmat_execution::{value::ValueRef, Digest};

pub(super) mod closure;
mod frequency;
pub use frequency::{
    AdmittedFrequencySamplePublication, FrequencySampleArtifactRootInventory,
    FrequencySamplePublicationAccess,
};

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisArtifactRoot {
    pub(super) reference: ValueRef,
    expected_kind: AnalysisNumericalObjectKind,
    expected_logical_identity: Digest,
    expected_payload_digest: Digest,
}

impl AnalysisArtifactRoot {
    pub fn new(
        reference: ValueRef,
        expected_kind: AnalysisNumericalObjectKind,
        expected_logical_identity: Digest,
        expected_payload_digest: Digest,
    ) -> Result<Self, String> {
        let root = Self {
            reference,
            expected_kind,
            expected_logical_identity,
            expected_payload_digest,
        };
        root.validate_role(expected_kind, expected_logical_identity, "artifact")?;
        Ok(root)
    }

    pub fn reference(&self) -> &ValueRef {
        &self.reference
    }

    pub const fn expected_kind(&self) -> AnalysisNumericalObjectKind {
        self.expected_kind
    }

    pub const fn expected_logical_identity(&self) -> Digest {
        self.expected_logical_identity
    }

    pub const fn expected_payload_digest(&self) -> Digest {
        self.expected_payload_digest
    }

    pub(super) fn validate_role(
        &self,
        expected_kind: AnalysisNumericalObjectKind,
        expected_logical_identity: Digest,
        role: &str,
    ) -> Result<(), String> {
        if self.expected_kind != expected_kind
            || self.expected_logical_identity != expected_logical_identity
            || !nonempty(self.expected_logical_identity)
            || !nonempty(self.expected_payload_digest)
        {
            return Err(format!(
                "analysis {role} root differs from its semantic publication"
            ));
        }
        Ok(())
    }
}

fn nonempty(digest: Digest) -> bool {
    digest != Digest::sha256([]) && digest != Digest::from_bytes([0; 32])
}

pub(super) fn validate_complete_root_inventory(
    mut declared: Vec<ValueRef>,
    catalog: &runmat_execution::value::PublishedObjectCatalog,
) -> Result<(), String> {
    catalog.validate().map_err(|error| error.to_string())?;
    declared.sort_by_key(|reference| reference.id);
    if declared != catalog.roots {
        return Err(
            "analysis semantic roots differ from the complete published root inventory".into(),
        );
    }
    Ok(())
}

/// Field and optional observation roots in exact numerical sample order.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrequencySampleArtifactRoots {
    pub(super) fields: AnalysisArtifactRoot,
    pub(super) observations: Option<AnalysisArtifactRoot>,
}

impl FrequencySampleArtifactRoots {
    pub fn new(fields: AnalysisArtifactRoot, observations: Option<AnalysisArtifactRoot>) -> Self {
        Self {
            fields,
            observations,
        }
    }

    pub const fn fields(&self) -> &AnalysisArtifactRoot {
        &self.fields
    }

    pub const fn observations(&self) -> Option<&AnalysisArtifactRoot> {
        self.observations.as_ref()
    }

    pub(super) fn validate_catalogs(
        &self,
        fields: &RecoveredFieldArtifactManifest,
        observations: Option<&ObservationArtifactManifest>,
    ) -> Result<(), String> {
        self.fields.validate_role(
            AnalysisNumericalObjectKind::Field,
            fields.identity.digest(),
            "field",
        )?;
        match (&self.observations, observations) {
            (Some(root), Some(observations)) => root.validate_role(
                AnalysisNumericalObjectKind::Observation,
                observations.identity.digest(),
                "observation",
            ),
            (None, None) => Ok(()),
            _ => Err("analysis observation roots differ from the published sample order".into()),
        }
    }
}

/// An explicitly assigned original observation-history checkpoint root.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationHistoryArtifactRoot {
    pub(super) checkpoint: AnalysisArtifactRoot,
}

impl ObservationHistoryArtifactRoot {
    pub const fn new(checkpoint: AnalysisArtifactRoot) -> Self {
        Self { checkpoint }
    }

    pub const fn checkpoint(&self) -> &AnalysisArtifactRoot {
        &self.checkpoint
    }
}

#[cfg(test)]
mod tests;
