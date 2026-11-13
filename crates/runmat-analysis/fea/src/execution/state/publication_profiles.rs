use super::*;
use runmat_canonical_codec::{
    profile_serialized, profile_serialized_shape,
    projected_canonical_materialization_resident_bytes_estimate, SerializedCanonicalProfile,
};
use serde::ser::{SerializeMap, SerializeSeq};

pub const STATE_PUBLICATION_PROFILES_MAX_ENTRIES: usize = 256;
const PREFIX: &[u8] = b"runmat-analysis-state-publication-profiles/v1\0";
const DOMAIN: &str = "state-publication-profiles";
const LIMITS: CanonicalLimits = CanonicalLimits::new(
    STATE_LIMITS.maximum_encoded_bytes,
    STATE_PUBLICATION_PROFILES_MAX_ENTRIES,
    STATE_LIMITS.maximum_string_bytes,
    STATE_LIMITS.maximum_nesting_depth,
);

/// A metadata-only size certificate for exactly one publication request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatePublicationProfileEntry {
    pub request: SemanticStatePublicationRequest,
    pub encoded_bytes_upper_bound: u64,
    pub materialization_resident_bytes_estimate: u64,
}

impl StatePublicationProfileEntry {
    pub fn new(
        request: SemanticStatePublicationRequest,
        profile: SerializedCanonicalProfile,
    ) -> AnalysisExecutionResult<Self> {
        let entry = Self {
            request,
            encoded_bytes_upper_bound: profile.encoded_bytes_upper_bound,
            materialization_resident_bytes_estimate: profile
                .materialization_resident_bytes_estimate,
        };
        entry.validate()?;
        Ok(entry)
    }

    pub fn profile(&self) -> SerializedCanonicalProfile {
        SerializedCanonicalProfile {
            encoded_bytes_upper_bound: self.encoded_bytes_upper_bound,
            materialization_resident_bytes_estimate: self.materialization_resident_bytes_estimate,
        }
    }

    fn validate(&self) -> AnalysisExecutionResult<()> {
        self.request.validate()?;
        let maximum_resident = projected_canonical_materialization_resident_bytes_estimate(
            STATE_LIMITS.maximum_encoded_bytes as u64,
            1,
        )
        .ok_or_else(|| invalid("semantic-state resident limit overflow"))?;
        if self.encoded_bytes_upper_bound == 0
            || self.encoded_bytes_upper_bound > STATE_LIMITS.maximum_encoded_bytes as u64
            || self.materialization_resident_bytes_estimate == 0
            || self.materialization_resident_bytes_estimate > maximum_resident
        {
            return Err(invalid(
                "state publication profile exceeds semantic-state limits",
            ));
        }
        profile_serialized(&self.request, LIMITS)?;
        Ok(())
    }
}

/// Immutable, content-addressed catalog bound to the exact solver payload bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StatePublicationProfilesWire")]
pub struct StatePublicationProfiles {
    schema_version: u16,
    identity: Digest,
    producer: StepIdentity,
    source_result: SolverResultIdentity,
    topology_digest: Digest,
    source_payload_digest: Digest,
    entries: Vec<StatePublicationProfileEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StatePublicationProfilesWire {
    schema_version: u16,
    identity: Digest,
    producer: StepIdentity,
    source_result: SolverResultIdentity,
    topology_digest: Digest,
    source_payload_digest: Digest,
    entries: Vec<StatePublicationProfileEntry>,
}

impl TryFrom<StatePublicationProfilesWire> for StatePublicationProfiles {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StatePublicationProfilesWire) -> Result<Self, Self::Error> {
        let value = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            source_result: wire.source_result,
            topology_digest: wire.topology_digest,
            source_payload_digest: wire.source_payload_digest,
            entries: wire.entries,
        };
        value.validate()?;
        Ok(value)
    }
}

impl StatePublicationProfiles {
    pub fn new(
        producer: StepIdentity,
        source_result: SolverResultIdentity,
        topology_digest: Digest,
        source_payload_digest: Digest,
        mut entries: Vec<StatePublicationProfileEntry>,
    ) -> AnalysisExecutionResult<Self> {
        if entries.len() > STATE_PUBLICATION_PROFILES_MAX_ENTRIES {
            return Err(invalid("too many state publication profiles"));
        }
        entries.sort_by(|a, b| {
            a.request
                .declaration
                .state_id
                .0
                .cmp(&b.request.declaration.state_id.0)
        });
        let mut value = Self {
            schema_version: 1,
            identity: Digest::sha256([]),
            producer,
            source_result,
            topology_digest,
            source_payload_digest,
            entries,
        };
        value.validate_metadata()?;
        value.identity = value.derive_identity()?;
        Ok(value)
    }

    pub fn identity(&self) -> Digest {
        self.identity
    }

    pub fn entries(&self) -> &[StatePublicationProfileEntry] {
        &self.entries
    }

    pub fn validate_source(
        &self,
        producer: StepIdentity,
        source_result: SolverResultIdentity,
        topology_digest: Digest,
        source_payload_digest: Digest,
    ) -> AnalysisExecutionResult<()> {
        if self.producer != producer
            || self.source_result != source_result
            || self.topology_digest != topology_digest
            || self.source_payload_digest != source_payload_digest
        {
            return Err(invalid(
                "state publication profiles name another solver payload",
            ));
        }
        Ok(())
    }

    pub fn entry(
        &self,
        request: &SemanticStatePublicationRequest,
    ) -> AnalysisExecutionResult<&StatePublicationProfileEntry> {
        request.validate()?;
        self.entries
            .binary_search_by(|entry| {
                entry
                    .request
                    .declaration
                    .state_id
                    .0
                    .cmp(&request.declaration.state_id.0)
            })
            .ok()
            .map(|index| &self.entries[index])
            .filter(|entry| entry.request == *request)
            .ok_or_else(|| invalid("state publication request has no exact profile entry"))
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        Ok(encode_contract(PREFIX, DOMAIN, self, LIMITS)?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(PREFIX, DOMAIN, bytes, LIMITS)?)
    }

    /// Profiles only declarations and fixed-width metadata, before numerical results exist.
    pub fn encoding_profile_for_requests(
        requests: &[SemanticStatePublicationRequest],
    ) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
        if requests.len() > STATE_PUBLICATION_PROFILES_MAX_ENTRIES {
            return Err(invalid("too many state publication profiles"));
        }
        let mut ids = std::collections::BTreeSet::new();
        for request in requests {
            request.validate()?;
            if !ids.insert(&request.declaration.state_id.0) {
                return Err(invalid("duplicate state publication profile id"));
            }
            if let Some(first) = requests.first() {
                if request.producer != first.producer
                    || request.source_result != first.source_result
                    || request.topology_digest != first.topology_digest
                {
                    return Err(invalid(
                        "state publication profile requests have different sources",
                    ));
                }
            }
        }
        // Stand-in digests and sizes bound every future catalog, so integers reserve full range.
        Ok(profile_serialized_shape(
            &CatalogProjection(requests),
            LIMITS,
        )?)
    }

    fn validate_metadata(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.entries.len() > STATE_PUBLICATION_PROFILES_MAX_ENTRIES {
            return Err(invalid(
                "invalid state publication profiles schema or count",
            ));
        }
        for (index, entry) in self.entries.iter().enumerate() {
            entry.validate()?;
            if entry.request.producer != self.producer
                || entry.request.source_result != self.source_result
                || entry.request.topology_digest != self.topology_digest
            {
                return Err(invalid(
                    "state publication profile entry names another source",
                ));
            }
            if index > 0
                && self.entries[index - 1].request.declaration.state_id.0
                    >= entry.request.declaration.state_id.0
            {
                return Err(invalid(
                    "state publication profiles must have unique sorted ids",
                ));
            }
        }
        profile_serialized(self, LIMITS)?;
        Ok(())
    }

    fn validate(&self) -> AnalysisExecutionResult<()> {
        self.validate_metadata()?;
        if self.identity != self.derive_identity()? {
            return Err(invalid(
                "state publication profiles identity does not converge",
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<Digest> {
        Ok(Digest::sha256(encode_contract(
            PREFIX,
            "state-publication-profiles-content",
            &(
                self.schema_version,
                self.producer,
                self.source_result,
                self.topology_digest,
                self.source_payload_digest,
                &self.entries,
            ),
            LIMITS,
        )?))
    }
}

struct CatalogProjection<'a>(&'a [SemanticStatePublicationRequest]);
impl Serialize for CatalogProjection<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(7))?;
        map.serialize_entry("schema_version", &1_u16)?;
        for name in [
            "identity",
            "producer",
            "source_result",
            "topology_digest",
            "source_payload_digest",
        ] {
            map.serialize_entry(name, &Digest::sha256([]))?;
        }
        map.serialize_entry("entries", &EntriesProjection(self.0))?;
        map.end()
    }
}

struct EntriesProjection<'a>(&'a [SemanticStatePublicationRequest]);
impl Serialize for EntriesProjection<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for request in self.0 {
            sequence.serialize_element(&EntryProjection(request))?;
        }
        sequence.end()
    }
}

struct EntryProjection<'a>(&'a SemanticStatePublicationRequest);
impl Serialize for EntryProjection<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("request", self.0)?;
        // The shared profiler reserves the maximum integer width for all u64 values.
        map.serialize_entry("encoded_bytes_upper_bound", &1_u64)?;
        map.serialize_entry("materialization_resident_bytes_estimate", &1_u64)?;
        map.end()
    }
}

fn invalid(reason: &str) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(reason.into())
}

#[cfg(test)]
mod tests;
