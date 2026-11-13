//! One wire authority for manifests and shape projection without hypothetical inventory vectors.
use super::*;

#[derive(Serialize)]
#[serde(rename = "AnalysisObjectManifest")]
pub(super) struct ManifestSerialization<C, P> {
    pub schema_version: u16,
    pub kind: AnalysisNumericalObjectKind,
    pub logical_identity: Digest,
    pub payload_digest: Digest,
    pub encoded_length: u64,
    pub chunks: C,
    pub partitions: P,
}

impl Serialize for AnalysisObjectManifest {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let Self {
            schema_version,
            kind,
            logical_identity,
            payload_digest,
            encoded_length,
            chunks,
            partitions,
        } = self;
        ManifestSerialization {
            schema_version: *schema_version,
            kind: *kind,
            logical_identity: *logical_identity,
            payload_digest: *payload_digest,
            encoded_length: *encoded_length,
            chunks,
            partitions,
        }
        .serialize(serializer)
    }
}
