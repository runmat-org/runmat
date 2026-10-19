//! Artifact-owned source-bearing interpreter region envelope.

use runmat_execution::{ExecutableSourceMapV2, ProgramRevision};
use serde::{Deserialize, Serialize};

use crate::{archive::ArchiveLimits, error::contract_error_in, ArtifactError, ArtifactResult};

const PREFIX: &[u8] = b"runmat-region-sources-v1\n";
const SCHEMA: u16 = 1;

fn admit_encoded_length(length: usize) -> ArtifactResult<()> {
    if u64::try_from(length)
        .ok()
        .is_none_or(|length| length > ArchiveLimits::default().max_object_bytes)
    {
        return Err(ArtifactError::Limit(
            "interpreter region source object exceeds the archive object byte limit".into(),
        ));
    }
    Ok(())
}

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
        payload
            .source_map
            .validate()
            .map_err(|error| contract_error_in("invalid region source map", error))?;
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
        self.source_map
            .validate()
            .map_err(|error| contract_error_in("invalid region source map", error))?;
        let mut bytes = PREFIX.to_vec();
        bytes.extend(
            serde_json::to_vec(self).map_err(|error| {
                ArtifactError::Invalid(format!("invalid region payload: {error}"))
            })?,
        );
        admit_encoded_length(bytes.len())?;
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8], expected_revision: &ProgramRevision) -> ArtifactResult<Self> {
        admit_encoded_length(bytes.len())?;
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
        payload
            .source_map
            .validate()
            .map_err(|error| contract_error_in("invalid region source map", error))?;
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

    fn admitted() -> InterpreterRegionSourcesV1 {
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
        InterpreterRegionSourcesV1::new(revision(b"original"), map, b"opaque-script".to_vec())
            .unwrap()
    }

    fn forged_bytes(mutator: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        let original = admitted().encode().unwrap();
        let mut value: serde_json::Value =
            serde_json::from_slice(original.strip_prefix(PREFIX).unwrap()).unwrap();
        mutator(&mut value);
        let mut bytes = PREFIX.to_vec();
        bytes.extend(serde_json::to_vec(&value).unwrap());
        bytes
    }

    #[test]
    fn exact_region_map_and_revision_are_admitted_before_script_use() {
        let admitted = admitted();
        let bytes = admitted.encode().unwrap();
        assert_eq!(
            InterpreterRegionSourcesV1::decode(&bytes, &revision(b"original"))
                .unwrap()
                .source_map(),
            admitted.source_map()
        );
        assert!(InterpreterRegionSourcesV1::decode(&bytes, &revision(b"changed")).is_err());
    }

    #[test]
    fn preamble_schema_and_unknown_fields_are_independently_rejected() {
        let original = admitted().encode().unwrap();
        assert!(matches!(
            InterpreterRegionSourcesV1::decode(b"unknown-preamble\n{}", &revision(b"original")),
            Err(ArtifactError::Invalid(_))
        ));
        let stale_schema =
            forged_bytes(|value| value["schema_version"] = serde_json::json!(SCHEMA + 1));
        assert!(matches!(
            InterpreterRegionSourcesV1::decode(&stale_schema, &revision(b"original")),
            Err(ArtifactError::Invalid(_))
        ));
        let unknown = forged_bytes(|value| value["unknown_field"] = serde_json::json!(true));
        assert!(matches!(
            InterpreterRegionSourcesV1::decode(&unknown, &revision(b"original")),
            Err(ArtifactError::Invalid(_))
        ));
        assert!(InterpreterRegionSourcesV1::decode(&original, &revision(b"original")).is_ok());
    }

    #[test]
    fn empty_script_changed_text_and_nondense_ordinals_fail_separately() {
        for forged in [
            forged_bytes(|value| value["script"] = serde_json::json!([])),
            forged_bytes(|value| {
                value["source_map"]["entries"][0]["text"] = serde_json::json!("source = 2;")
            }),
            forged_bytes(|value| {
                value["source_map"]["entries"][0]["source_id"] = serde_json::json!(1)
            }),
        ] {
            assert!(matches!(
                InterpreterRegionSourcesV1::decode(&forged, &revision(b"original")),
                Err(ArtifactError::Invalid(_))
            ));
        }
        let mut empty = admitted();
        empty.script.clear();
        assert!(matches!(empty.encode(), Err(ArtifactError::Invalid(_))));
    }

    #[test]
    fn existing_archive_object_limit_is_checked_before_json_decode() {
        let length = usize::try_from(ArchiveLimits::default().max_object_bytes).unwrap() + 1;
        assert!(matches!(
            admit_encoded_length(length),
            Err(ArtifactError::Limit(_))
        ));
        assert!(matches!(
            InterpreterRegionSourcesV1::decode(&[], &revision(b"original")),
            Err(ArtifactError::Invalid(_))
        ));
    }
}
