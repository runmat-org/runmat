//! Publication phase estimates derived from original borrowed inventories before codec work.
use super::*;
use crate::analysis::structural_study::publication::evidence::immutable_batch_workspace;
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
    // Staged records and attachments are owned typed values; they are priced as such, not as
    // a materialized value tree (only the commit projection builds one, see `commit_work`).
    let _ = maximum;
    let mut total = 0;
    for record in runs.values() {
        total = add(
            total,
            runmat_canonical_codec::typed_resident_bytes_estimate(&RecordView(record))?,
        )?;
    }
    for bytes in chunks.values() {
        total = add(total, bytes.len() as u64)?;
    }
    for attachment in observations.values() {
        total = add(
            total,
            runmat_canonical_codec::typed_resident_bytes_estimate(attachment)?,
        )?;
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
) -> Result<CommitWork, AnalysisPublicationError> {
    let descriptor = DescriptorShape {
        schema_version: 1,
        definition,
        manifest_digest: Digest::sha256(b"publication-size-placeholder"),
        records: DigestKeys(runs),
        observations: DigestKeys(observations),
    };
    let mut record_work = 0;
    let mut largest_record_parts = String::new();
    for record in runs.values() {
        let view = RecordView(record);
        // The record's identity is its direct canonical encoding and its transport is JSON
        // written from the typed record: neither builds a value tree. Both buffers coexist.
        let encoding = runmat_canonical_codec::profile_canonical_encoding(
            &view,
            runmat_canonical_codec::CanonicalLimits::new(
                usize::try_from(maximum).unwrap_or(usize::MAX),
                usize::MAX,
                65_536,
                64,
            ),
        )
        .map_err(AnalysisPublicationError::from)?;
        let work = add(
            add(
                encoding.encoded_bytes_upper_bound,
                encoding.resident_workspace_bytes_estimate,
            )?,
            profile_json_bytes(&view, maximum).map_err(AnalysisPublicationError::from)?,
        )?;
        if work > record_work {
            record_work = work;
            // Which parts of the record carry its size, for the rejection facts.
            if let PersistedAnalysisArtifact::Standard(r) = record {
                let json = |value: &dyn erased_json::Json| value.bytes(maximum);
                largest_record_parts = format!(
                    "run={} metadata={} field_inventory={} fields={} render_tiles={} histories={} original_fields={}",
                    json(&r.run),
                    json(&r.metadata),
                    json(&r.field_inventory),
                    json(&r.fields),
                    json(&r.render_tiles),
                    json(&r.observation_histories),
                    json(&r.original_fields),
                );
            }
        }
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
    let total = add(
        add(add(manifest_work, descriptor_work)?, record_work)?,
        memory_transport,
    )?;
    let mut largest_manifest_entries = Vec::new();
    if let Some(entries) = manifest.as_object() {
        for (key, value) in entries {
            largest_manifest_entries.push((resident(value, maximum)?, key.as_str()));
        }
    }
    largest_manifest_entries.sort_unstable_by(|left, right| right.cmp(left));
    largest_manifest_entries.truncate(3);
    Ok(CommitWork {
        total,
        facts: format!(
            "commit work: manifest={manifest_work} (largest entries {largest_manifest_entries:?}), descriptor={descriptor_work}, largest_record={record_work} ({largest_record_parts}), transport={memory_transport}"
        ),
    })
}

pub(super) struct CommitWork {
    pub total: u64,
    pub facts: String,
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
            )?
            .total,
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
    /// In-process source-owner demand, bound to the complete original reference inventory.
    /// This is resource admission, never a numerical proof decoded from a digest.
    pub(in crate::analysis) fn admit_profiled_source_objects(
        &mut self,
        objects: &[runmat_execution::value::ValueRef],
        required: u64,
    ) -> Result<(), AnalysisPublicationError> {
        if self.source_phase_active()? {
            return Err("cannot change source inventory during its active phase".into());
        }
        let encoded = objects.iter().map(|r| r.encoded_length).max().unwrap_or(0);
        if required < encoded || (encoded > 0 && required == 0) {
            return Err(AnalysisPublicationError::ResourceLimit {
                message: "source demand omits original transport".into(),
            });
        }
        // The source's own demand and the catalog persistence it is followed by share the lease.
        let required = immutable_batch_workspace(encoded)
            .map(|persist| persist.max(required))
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "publication source workspace overflow".into(),
            })?;
        let planned = self.source_workspace_bytes.max(required);
        self.admit(planned)?;
        let identity = canonical_digest("publication-source-inventory", &objects)?;
        self.source_profiles.insert(identity, required);
        self.source_workspace_bytes = planned;
        Ok(())
    }

    pub(super) fn planned_source_workspace(
        &self,
        objects: &[runmat_execution::value::ValueRef],
    ) -> Result<u64, AnalysisPublicationError> {
        let identity = canonical_digest("publication-source-inventory", &objects)?;
        self.source_profiles
            .get(&identity)
            .copied()
            .map_or_else(|| source_workspace(objects), Ok)
    }

    pub(in crate::analysis) fn source_replay_allowance(
        &self,
        objects: &[runmat_execution::value::ValueRef],
    ) -> Result<u64, AnalysisPublicationError> {
        self.require_source_objects(objects)?;
        self.active_source_workspace()
    }
    /// Borrowed original metadata plus its copy/verification workspace remain charged
    /// while new staged output and commit/recovery work are admitted.
    pub(in crate::analysis) fn admit_preparation(
        &mut self,
        input: &impl Serialize,
    ) -> Result<PreparationReservation, AnalysisPublicationError> {
        // The original and its copy are owned typed values, priced as such (no value tree).
        let bytes = typed_resident(input, self.maximum_bytes)?;
        let required =
            bytes
                .checked_mul(2)
                .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                    message: "publication input profile overflow".into(),
                })?;
        self.admit(required)
            .map_err(|error| error.context("publication preparation input and copy"))?;
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
        if self.source_phase_active()? {
            return Err("cannot change source inventory during its active phase".into());
        }
        let required = source_workspace(objects)?;
        let planned = self.source_workspace_bytes.max(required);
        self.admit(planned)?;
        let identity = canonical_digest("publication-source-inventory", &objects)?;
        self.source_profiles.insert(identity, required);
        // Catalogs are consumed sequentially. Input and output storage is retained;
        // only one guarded replay workspace can be active at a time.
        self.source_workspace_bytes = planned;
        Ok(())
    }

    pub(in crate::analysis) fn require_source_objects(
        &self,
        objects: &[runmat_execution::value::ValueRef],
    ) -> Result<(), AnalysisPublicationError> {
        let identity = canonical_digest("publication-source-inventory", &objects)?;
        if !self.source_replay_active()?
            || self
                .source_profiles
                .get(&identity)
                .is_none_or(|required| *required > self.source_workspace_bytes)
        {
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
        // Original inputs and staged outputs remain charged. The source phase uses
        // this remaining allowance; its temporary workspace is not retained at commit.
        let fixed = add(retained, self.preparation_resident_bytes)?;
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
        // The state is encoded directly from its typed value; only the encoder's buffers are new.
        let profile = runmat_canonical_codec::profile_canonical_encoding(
            state,
            CanonicalLimits::new(
                usize::try_from(maximum).unwrap_or(usize::MAX),
                usize::MAX,
                65_536,
                64,
            ),
        )
        .map_err(AnalysisPublicationError::from)?;
        let required = state_persistence_workspace(&profile)
            .filter(|required| *required <= maximum)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: format!(
                    "semantic state encoding exceeds its phase allowance: state {} bytes, encoding workspace {} bytes, allowance {maximum} bytes",
                    profile.encoded_bytes_upper_bound, profile.resident_workspace_bytes_estimate,
                ),
            })?;
        let bytes = state.canonical_bytes().map_err(|error| error.to_string())?;
        if bytes.len() as u64 > profile.encoded_bytes_upper_bound {
            return Err("semantic state exceeds its profiled encoding".into());
        }
        let relative = format!(
            "sha256_{digest}/states/sha256_{}.state",
            state
                .identity
                .digest()
                .to_string()
                .trim_start_matches("sha256:")
        );
        self.evidence_access
            .persist_bytes(&relative, &bytes, required)
    }

    /// Admit the typed clone of the public run placed into the family manifest.
    /// Its JSON projection is admitted separately by `serialize_retained_manifest`.
    pub(in crate::analysis) fn admit_manifest_copy(
        &self,
        value: &impl Serialize,
    ) -> Result<(), AnalysisPublicationError> {
        let mut guard = self
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")?;
        self.admit(runmat_canonical_codec::typed_resident_bytes_estimate(
            value,
        )?)
        .map_err(|error| error.context("family manifest typed public-run copy"))?;
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
        // One JSON tree is built from the borrowed value.
        self.admit(json_projection(value, self.maximum_bytes)?)
            .map_err(|error| error.context("manifest JSON projection"))?;
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
        // The retained manifest is already charged; only the run's JSON tree is new.
        self.admit(json_projection(run, self.maximum_bytes)?)
            .map_err(|error| error.context("manifest public-run JSON projection"))?;
        *guard = true;
        manifest["public_run"] =
            serde_json::to_value(run).map_err(AnalysisPublicationError::from)?;
        Ok(())
    }
}

/// Requested workspace for projecting a borrowed value into one JSON tree.
fn json_projection(value: &impl Serialize, maximum: u64) -> Result<u64, AnalysisPublicationError> {
    runmat_canonical_codec::profile_serialized(
        value,
        CanonicalLimits::content_bounded(
            usize::try_from(maximum).map_err(|_| AnalysisPublicationError::ResourceLimit {
                message: "publication budget exceeds address space".into(),
            })?,
            65_536,
            64,
        ),
    )?
    .json_projection_resident_bytes_estimate()
    .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
        message: "invalid JSON projection workspace".into(),
    })
}

/// Encoding and persistence are sequential: the encoder's workspace, which includes its
/// output, is released except for the output, which then joins one comparison buffer.
pub(super) fn state_persistence_workspace(
    profile: &runmat_canonical_codec::CanonicalEncodingProfile,
) -> Option<u64> {
    immutable_batch_workspace(profile.encoded_bytes_upper_bound)
        .map(|persist| persist.max(profile.resident_workspace_bytes_estimate))
}

pub(super) fn source_workspace(
    objects: &[runmat_execution::value::ValueRef],
) -> Result<u64, AnalysisPublicationError> {
    let Some(encoded) = objects
        .iter()
        .map(|reference| reference.encoded_length)
        .max()
    else {
        return Ok(0);
    };
    let overflow = || AnalysisPublicationError::ResourceLimit {
        message: "publication source workspace overflow".into(),
    };
    let decoded =
        runmat_canonical_codec::projected_canonical_materialization_resident_bytes_estimate(
            encoded, 1,
        )
        .ok_or_else(overflow)?;
    // Replay (transport plus decode) and catalog persistence run in sequence under one lease.
    let persist = immutable_batch_workspace(encoded).ok_or_else(overflow)?;
    Ok(add(encoded, decoded)?.max(persist))
}

mod erased_json {
    /// JSON byte bound of one record part, for diagnostics.
    pub(super) trait Json {
        fn bytes(&self, maximum: u64) -> u64;
    }
    impl<T: serde::Serialize> Json for T {
        fn bytes(&self, maximum: u64) -> u64 {
            runmat_canonical_codec::profile_json_bytes(self, maximum).unwrap_or(u64::MAX)
        }
    }
}
