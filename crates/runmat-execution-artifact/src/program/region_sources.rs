//! Artifact-owned source-bearing interpreter region envelope.

use runmat_execution::{ExecutableSourceMapV2, ProgramRevision};
use serde::{Deserialize, Serialize};

use crate::{ArtifactError, ArtifactResult};

const PREFIX: &[u8] = b"runmat-region-sources-v1\n";
const SCHEMA: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterpreterRegionSourcesV1 {
    schema_version: u16,
    program_revision: ProgramRevision,
    source_map: ExecutableSourceMapV2,
    script: Vec<u8>,
}

impl InterpreterRegionSourcesV1 {
    pub fn new(
        program_revision: ProgramRevision,
        source_map: ExecutableSourceMapV2,
        script: Vec<u8>,
    ) -> ArtifactResult<Self> {
        let payload = Self {
            schema_version: SCHEMA,
            program_revision,
            source_map,
            script,
        };
        payload.source_map.validate().map_err(|error| {
            ArtifactError::Invalid(format!("invalid region source map: {error}"))
        })?;
        if payload.script.is_empty() {
            return Err(ArtifactError::Invalid("region script is empty".into()));
        }
        Ok(payload)
    }

    pub fn encode(&self) -> ArtifactResult<Vec<u8>> {
        if self.schema_version != SCHEMA || self.script.is_empty() {
            return Err(ArtifactError::Invalid(
                "region source schema or script is invalid".into(),
            ));
        }
        self.source_map.validate().map_err(|error| {
            ArtifactError::Invalid(format!("invalid region source map: {error}"))
        })?;
        let mut bytes = PREFIX.to_vec();
        bytes.extend(
            serde_json::to_vec(self).map_err(|error| {
                ArtifactError::Invalid(format!("invalid region payload: {error}"))
            })?,
        );
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8], expected_revision: &ProgramRevision) -> ArtifactResult<Self> {
        let payload = bytes.strip_prefix(PREFIX).ok_or_else(|| {
            ArtifactError::Invalid("region source payload has an unsupported preamble".into())
        })?;
        let payload: Self = serde_json::from_slice(payload)
            .map_err(|error| ArtifactError::Invalid(format!("invalid region payload: {error}")))?;
        if payload.schema_version != SCHEMA || &payload.program_revision != expected_revision {
            return Err(ArtifactError::Invalid(
                "region source schema or revision differs from its recipe".into(),
            ));
        }
        payload.source_map.validate().map_err(|error| {
            ArtifactError::Invalid(format!("invalid region source map: {error}"))
        })?;
        if payload.script.is_empty() {
            return Err(ArtifactError::Invalid("region script is empty".into()));
        }
        Ok(payload)
    }

    pub fn source_map(&self) -> &ExecutableSourceMapV2 {
        &self.source_map
    }

    pub fn script(&self) -> &[u8] {
        &self.script
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{Digest, LanguageCompatibilityMode, ProgramEnvironment};
    use runmat_types::ProgramSourceId;

    fn revision(label: &[u8]) -> ProgramRevision {
        ProgramRevision::new(
            Digest::sha256(label),
            Digest::sha256(b"source"),
            ProgramEnvironment::new(
                1,
                1,
                Digest::sha256(b"runtime"),
                Digest::sha256(b"catalog"),
                LanguageCompatibilityMode::Matlab,
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn exact_region_map_and_revision_are_admitted_before_script_use() {
        let text = "source = 1;";
        let map = ExecutableSourceMapV2::new(vec![runmat_execution::ExecutableSourceRecordV2 {
            source_id: ProgramSourceId(0),
            owner_identity: "fixture".into(),
            relative_path: "source.m".into(),
            display_name: "source.m".into(),
            content_revision: Digest::sha256(text.as_bytes()),
            text: text.into(),
        }])
        .unwrap();
        let admitted =
            InterpreterRegionSourcesV1::new(revision(b"original"), map, b"opaque-script".to_vec())
                .unwrap();
        let bytes = admitted.encode().unwrap();
        assert_eq!(
            InterpreterRegionSourcesV1::decode(&bytes, &revision(b"original"))
                .unwrap()
                .source_map(),
            admitted.source_map()
        );
        assert!(InterpreterRegionSourcesV1::decode(&bytes, &revision(b"changed")).is_err());
        let mut forged: serde_json::Value =
            serde_json::from_slice(bytes.strip_prefix(PREFIX).unwrap()).unwrap();
        forged["source_map"]["entries"][0]["text"] = serde_json::json!("source = 2;");
        let mut forged_bytes = PREFIX.to_vec();
        forged_bytes.extend(serde_json::to_vec(&forged).unwrap());
        assert!(InterpreterRegionSourcesV1::decode(&forged_bytes, &revision(b"original")).is_err());
        forged["schema_version"] = serde_json::json!(SCHEMA + 1);
        let mut stale_bytes = PREFIX.to_vec();
        stale_bytes.extend(serde_json::to_vec(&forged).unwrap());
        assert!(InterpreterRegionSourcesV1::decode(&stale_bytes, &revision(b"original")).is_err());
    }
}
