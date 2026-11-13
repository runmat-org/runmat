//! Construct a source-bound semantic catalog without gathering numerical fields.
use super::*;
impl RecoveredFieldArtifactManifest {
    pub fn from_entries(
        recovery_request: FieldRecoveryRequestIdentity,
        source: FieldRecoverySource,
        topology_digest: Digest,
        entity_counts: RecoveredFieldEntityCounts,
        fields: Vec<RecoveredFieldArtifactEntry>,
    ) -> Result<Self, RecoveredFieldError> {
        let mut manifest = Self {
            schema_version: 1,
            identity: RecoveredFieldArtifactIdentity(Digest::sha256([])),
            recovery_request,
            source,
            topology_digest,
            entity_counts,
            fields,
        };
        manifest.identity = derive_artifact_identity(&manifest)?;
        manifest.validate()?;
        Ok(manifest)
    }
}

pub(super) fn derive_artifact_identity(
    manifest: &RecoveredFieldArtifactManifest,
) -> Result<RecoveredFieldArtifactIdentity, RecoveredFieldError> {
    encode_contract(
        FIELD_PREFIX,
        "recovered-field-artifact-identity",
        &(
            manifest.schema_version,
            manifest.recovery_request,
            manifest.source,
            manifest.topology_digest,
            manifest.entity_counts,
            &manifest.fields,
        ),
        FIELD_LIMITS,
    )
    .map(Digest::sha256)
    .map(RecoveredFieldArtifactIdentity)
    .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
}
