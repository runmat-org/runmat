//! Publication phase estimates derived from original borrowed inventories before codec work.
use super::*;
use crate::analysis::AnalysisStudyRunData;
use runmat_canonical_codec::{encode_json_bounded, profile_json_bytes};

pub(super) struct RecordView<'a>(pub &'a PersistedAnalysisArtifact);
impl Serialize for RecordView<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            PersistedAnalysisArtifact::Standard(r) => r.serialize(serializer),
            PersistedAnalysisArtifact::StructuralFrequency(r) => r.serialize(serializer),
            PersistedAnalysisArtifact::AcousticFrequency(r) => r.serialize(serializer),
        }
    }
}

struct DigestKeys<'a, T>(&'a HashMap<String, T>);
impl<T> Serialize for DigestKeys<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        let placeholder = Digest::sha256(b"publication-size-placeholder");
        for id in self.0.keys() {
            map.serialize_entry(id, &placeholder)?;
        }
        map.end()
    }
}
#[derive(Serialize)]
struct DescriptorShape<'a> {
    schema_version: u16,
    definition: Digest,
    manifest_digest: Digest,
    records: DigestKeys<'a, PersistedAnalysisArtifact>,
    observations: DigestKeys<'a, observations::ObservationAttachment>,
}

fn add(a: u64, b: u64) -> Result<u64, AnalysisPublicationError> {
    a.checked_add(b)
        .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
            message: "publication resource count overflow".into(),
        })
}

pub(super) fn retained(
    runs: &HashMap<String, PersistedAnalysisArtifact>,
    chunks: &HashMap<Digest, Arc<[u8]>>,
    observations: &HashMap<String, observations::ObservationAttachment>,
    maximum: u64,
) -> Result<u64, AnalysisPublicationError> {
    let mut total = 0;
    for record in runs.values() {
        total = add(total, resident(&RecordView(record), maximum)?)?;
    }
    for bytes in chunks.values() {
        total = add(total, bytes.len() as u64)?;
    }
    for attachment in observations.values() {
        total = add(total, resident(attachment, maximum)?)?;
    }
    Ok(total)
}

/// Staged input plus manifest, descriptor, canonical projection and JSON commit buffers.
/// This is a requested phase estimate, not measured heap/RSS or the full execution task plan.
pub(super) fn commit_work(
    definition: Digest,
    manifest: &serde_json::Value,
    runs: &HashMap<String, PersistedAnalysisArtifact>,
    observations: &HashMap<String, observations::ObservationAttachment>,
    maximum: u64,
) -> Result<u64, AnalysisPublicationError> {
    let descriptor = DescriptorShape {
        schema_version: 1,
        definition,
        manifest_digest: Digest::sha256(b"publication-size-placeholder"),
        records: DigestKeys(runs),
        observations: DigestKeys(observations),
    };
    let mut record_work = 0;
    for record in runs.values() {
        let view = RecordView(record);
        // Record projection for stable identity and its bounded JSON transport coexist.
        record_work = record_work.max(add(
            resident(&view, maximum)?,
            profile_json_bytes(&view, maximum).map_err(AnalysisPublicationError::from)?,
        )?);
    }
    for attachment in observations.values() {
        record_work = record_work.max(add(
            resident(attachment, maximum)?,
            profile_json_bytes(attachment, maximum).map_err(AnalysisPublicationError::from)?,
        )?);
    }
    let manifest_work = resident(manifest, maximum)?;
    let descriptor_work = resident(&descriptor, maximum)?;
    let memory_transport = profile_json_bytes(&(&descriptor, manifest), maximum)
        .map_err(AnalysisPublicationError::from)?;
    add(
        add(add(manifest_work, descriptor_work)?, record_work)?,
        memory_transport,
    )
}

pub(in crate::analysis::storage) fn admission_bytes(
    prepared: &PreparedAnalysisPublication,
) -> Result<u64, AnalysisPublicationError> {
    add(
        prepared.preparation_resident_bytes,
        add(
            retained(
                &prepared.runs,
                &prepared.chunks,
                &prepared.observations,
                prepared.maximum_bytes,
            )?,
            commit_work(
                prepared.descriptor.definition,
                &prepared.manifest,
                &prepared.runs,
                &prepared.observations,
                prepared.maximum_bytes,
            )?,
        )?,
    )
}

pub(super) fn validate_admission(
    prepared: &PreparedAnalysisPublication,
) -> Result<(), AnalysisPublicationError> {
    if admission_bytes(prepared)? > prepared.maximum_bytes {
        return Err(AnalysisPublicationError::ResourceLimit {
            message: "publication commit exceeds its staging byte budget".into(),
        });
    }
    Ok(())
}

pub(super) fn json(
    value: &(impl Serialize + ?Sized),
    maximum: u64,
) -> Result<Vec<u8>, AnalysisPublicationError> {
    encode_json_bounded(value, maximum).map_err(AnalysisPublicationError::from)
}

impl AnalysisPublicationBuilder {
    /// Borrowed original metadata plus its copy/verification workspace remain charged
    /// while new staged output and commit/recovery work are admitted.
    pub(in crate::analysis) fn admit_preparation(
        &mut self,
        input: &impl Serialize,
    ) -> Result<PreparationReservation, AnalysisPublicationError> {
        let bytes = resident(input, self.maximum_bytes)?;
        let required =
            bytes
                .checked_mul(2)
                .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                    message: "publication input profile overflow".into(),
                })?;
        self.admit(required)?;
        self.preparation_resident_bytes = add(self.preparation_resident_bytes, required)?;
        Ok(PreparationReservation {
            scope: self.preparation_scope.clone(),
            resident_bytes: bytes,
        })
    }

    /// Source owners stream catalog objects. Their largest declared transport and shared
    /// hostile-payload decode envelope coexist with the original completion and its replay.
    /// This counts requested workspace; the source must enforce its existing verified-read
    /// contract and inventory limits. It does not establish measured allocator/RSS usage.
    pub(in crate::analysis) fn admit_source_objects(
        &mut self,
        objects: &[runmat_execution::value::ValueRef],
    ) -> Result<(), AnalysisPublicationError> {
        let required = source_workspace(objects)?;
        self.admit(required)?;
        self.preparation_resident_bytes = add(self.preparation_resident_bytes, required)?;
        self.source_workspace_bytes = add(self.source_workspace_bytes, required)?;
        Ok(())
    }

    pub(in crate::analysis) fn require_source_objects(
        &self,
        objects: &[runmat_execution::value::ValueRef],
    ) -> Result<(), AnalysisPublicationError> {
        if source_workspace(objects)? > self.source_workspace_bytes {
            return Err(AnalysisPublicationError::ResourceLimit {
                message: "source objects lack their admitted publication workspace".into(),
            });
        }
        Ok(())
    }

    pub(in crate::analysis) fn source_workspace_limit(
        &self,
    ) -> Result<u64, AnalysisPublicationError> {
        let retained = retained(
            &*self
                .store
                .runs
                .read()
                .map_err(|_| "staging lock poisoned")?,
            &*self
                .store
                .chunks
                .read()
                .map_err(|_| "staging lock poisoned")?,
            &*self
                .store
                .observations
                .read()
                .map_err(|_| "staging lock poisoned")?,
            self.maximum_bytes,
        )?;
        // Source workspace was charged before authentication. Reuse that reservation
        // during this phase; original inputs and staged outputs remain charged.
        let fixed = add(retained, self.preparation_resident_bytes)?
            .checked_sub(self.source_workspace_bytes)
            .ok_or("source workspace exceeds its preparation reservation")?;
        let available = self
            .maximum_bytes
            .checked_sub(fixed)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "publication evidence phase exceeds its host allowance".into(),
            })?;
        Ok(available)
    }

    pub(in crate::analysis) fn persist_evidence(
        &self,
        fingerprint: &str,
        stage: &str,
        payload: serde_json::Value,
    ) -> Result<String, AnalysisPublicationError> {
        let available = self.source_workspace_limit()?;
        crate::analysis::structural_study::publication::evidence::persist_admitted(
            &self.evidence_access,
            fingerprint,
            stage,
            payload,
            available,
        )
    }

    pub(in crate::analysis) fn persist_state(
        &self,
        fingerprint: &str,
        state: &runmat_analysis_fea::execution::CanonicalSemanticState,
    ) -> Result<String, AnalysisPublicationError> {
        let digest = fingerprint
            .strip_prefix("sha256:")
            .ok_or("study fingerprint is not a SHA-256 identity")?;
        let maximum = self.source_workspace_limit()?;
        let resident = resident(state, maximum)?;
        let available = maximum
            .checked_sub(resident)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "semantic state encoding exceeds its phase allowance".into(),
            })?;
        let bytes = state.canonical_bytes().map_err(|error| error.to_string())?;
        let relative = format!(
            "sha256_{digest}/states/sha256_{}.state",
            state
                .identity
                .digest()
                .to_string()
                .trim_start_matches("sha256:")
        );
        self.evidence_access
            .persist_bytes(&relative, &bytes, available)
    }

    /// Admit source and projected copies before family-manifest cloning/materialization.
    pub(in crate::analysis) fn admit_manifest_copy(
        &self,
        value: &impl Serialize,
    ) -> Result<(), AnalysisPublicationError> {
        let mut guard = self
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")?;
        self.admit(
            resident(value, self.maximum_bytes)?
                .checked_mul(2)
                .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                    message: "staging byte overflow".into(),
                })?,
        )?;
        *guard = true;
        Ok(())
    }
    pub(in crate::analysis) fn serialize_manifest(
        &self,
        value: &impl Serialize,
    ) -> Result<serde_json::Value, AnalysisPublicationError> {
        let mut guard = self
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")?;
        self.admit(
            resident(value, self.maximum_bytes)?
                .checked_mul(2)
                .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                    message: "staging byte overflow".into(),
                })?,
        )?;
        *guard = true;
        serde_json::to_value(value).map_err(AnalysisPublicationError::from)
    }
    pub(in crate::analysis) fn update_manifest_public_run(
        &self,
        manifest: &mut serde_json::Value,
        run: &AnalysisStudyRunData,
    ) -> Result<(), AnalysisPublicationError> {
        if manifest.get("public_run").is_none() {
            return Err("publication manifest has no public run".into());
        }
        let mut guard = self
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")?;
        self.admit(
            resident(&(manifest as &serde_json::Value, run), self.maximum_bytes)?
                .checked_mul(2)
                .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                    message: "staging byte overflow".into(),
                })?,
        )?;
        *guard = true;
        manifest["public_run"] =
            serde_json::to_value(run).map_err(AnalysisPublicationError::from)?;
        Ok(())
    }
}

fn source_workspace(
    objects: &[runmat_execution::value::ValueRef],
) -> Result<u64, AnalysisPublicationError> {
    let Some(encoded) = objects
        .iter()
        .map(|reference| reference.encoded_length)
        .max()
    else {
        return Ok(0);
    };
    let decoded =
        runmat_canonical_codec::projected_canonical_materialization_resident_bytes_estimate(
            encoded, 1,
        )
        .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
            message: "publication source workspace overflow".into(),
        })?;
    add(encoded, decoded)
}
