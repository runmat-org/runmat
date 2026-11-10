//! Privately prepared closed publications. Storage integrity is not numerical qualification.
use super::*;
use runmat_canonical_codec::{encode_contract, profile_serialized, CanonicalLimits};
use std::collections::BTreeSet;

mod assessment;
mod binding;
mod catalog;
mod owned_storage;
pub(in crate::analysis) use owned_storage::OwnedCopyReservation;
pub(super) mod filesystem;
mod initial_fields;
pub(super) mod memory;
mod preparation;
mod run_admission;
mod source_fields;
mod source_phase;
pub(in crate::analysis::storage) mod spill;
mod typed_preparation;
pub(in crate::analysis) use preparation::PreparationReservation;
pub(in crate::analysis) use typed_preparation::retained_mesh_storage;
pub(in crate::analysis::storage) mod resources;
pub(in crate::analysis::storage) mod retention;
#[cfg(test)]
mod tests;

pub struct AnalysisPublicationBuilder {
    definition: Digest,
    maximum_bytes: u64,
    preparation_resident_bytes: u64,
    source_workspace_bytes: u64,
    source_profiles: BTreeMap<Digest, u64>,
    source_phase: Arc<std::sync::Mutex<Option<source_phase::SourceWorkspaceCharge>>>,
    preparation_scope: Arc<()>,
    evidence_access: crate::analysis::structural_study::publication::evidence::StudyEvidenceAccess,
    admission: std::sync::Mutex<bool>,
    store: InMemoryAnalysisArtifactStore,
    /// Field chunks streamed out of staging; the budget bounds only the bytes in flight.
    spill: std::sync::Mutex<spill::SpilledChunks>,
}

pub struct PreparedAnalysisPublication {
    maximum_bytes: u64,
    preparation_resident_bytes: u64,
    descriptor: PublicationDescriptor,
    manifest: serde_json::Value,
    runs: HashMap<String, PersistedAnalysisArtifact>,
    chunks: HashMap<Digest, Arc<[u8]>>,
    observations: HashMap<String, observations::ObservationAttachment>,
    /// Field chunks written to the private spill area during staging, read back at commit.
    spilled: spill::SpilledChunks,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisPublicationReceipt {
    pub identity: Digest,
    pub definition: Digest,
    pub manifest_digest: Digest,
    pub run_ids: Vec<String>,
    /// In-memory stores do not claim a durable filesystem artifact.
    pub manifest_artifact_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::analysis::storage) struct PublicationDescriptor {
    schema_version: u16,
    definition: Digest,
    manifest_digest: Digest,
    records: BTreeMap<String, Digest>,
    observations: BTreeMap<String, Digest>,
}

impl PublicationDescriptor {
    fn identity(&self) -> Result<Digest, String> {
        canonical_digest("analysis-publication-descriptor", self)
    }
    fn receipt(&self, path: Option<PathBuf>) -> Result<AnalysisPublicationReceipt, String> {
        Ok(AnalysisPublicationReceipt {
            identity: self.identity()?,
            definition: self.definition,
            manifest_digest: self.manifest_digest,
            run_ids: self.records.keys().map(ToOwned::to_owned).collect(),
            manifest_artifact_path: path,
        })
    }
}

fn canonical_digest(domain: &'static str, value: &impl Serialize) -> Result<Digest, String> {
    encode_contract(
        b"runmat-analysis-publication/v1\0",
        domain,
        value,
        CanonicalLimits::platform_content(65_536, 64),
    )
    .map(Digest::sha256)
    .map_err(|e| e.to_string())
}

impl PreparedAnalysisPublication {
    pub(in crate::analysis::storage) fn maximum_bytes(&self) -> u64 {
        self.maximum_bytes
    }

    pub(in crate::analysis::storage) fn validate(&self) -> Result<(), AnalysisPublicationError> {
        resources::validate_admission(self)?;
        validate_manifest(
            self.descriptor.identity()?,
            &self.descriptor,
            &self.manifest,
        )?;
        let actual = self
            .runs
            .iter()
            .map(|(id, record)| {
                validate_run_id_token(id)?;
                Ok((id.clone(), record_digest(record)?))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        if actual != self.descriptor.records {
            return Err("prepared records differ from publication inventory".into());
        }
        for (digest, bytes) in &self.chunks {
            if Digest::sha256(bytes.as_ref()) != *digest {
                return Err("prepared chunk differs from publication inventory".into());
            }
        }
        let observations = self
            .observations
            .iter()
            .map(|(id, attachment)| {
                let Some(PersistedAnalysisArtifact::Standard(run)) = self.runs.get(id) else {
                    return Err("observation attachment lacks its original standard run".into());
                };
                attachment.validate(&run.metadata, run.run_digest)?;
                Ok((
                    id.clone(),
                    canonical_digest("analysis-publication-observation", attachment)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        if observations != self.descriptor.observations
            || self.observations.keys().any(|id| !actual.contains_key(id))
        {
            return Err("observation is outside publication inventory".into());
        }
        Ok(())
    }
}

#[cfg(test)]
fn record_value(record: &PersistedAnalysisArtifact) -> Result<serde_json::Value, String> {
    match record {
        PersistedAnalysisArtifact::Standard(r) => serde_json::to_value(r),
        PersistedAnalysisArtifact::StructuralFrequency(r) => serde_json::to_value(r),
        PersistedAnalysisArtifact::AcousticFrequency(r) => serde_json::to_value(r),
    }
    .map_err(|e| e.to_string())
}

/// The record without its creation time, borrowed field by field. The canonical encoder orders
/// keys, so encoding this view directly yields the same bytes as encoding the record's JSON tree
/// with `created_at` removed, without materializing that tree.
#[derive(Serialize)]
struct StandardDigestView<'a> {
    schema_version: &'a String,
    op_version: &'a String,
    run_digest: &'a Digest,
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    run: &'a serde_json::Value,
    metadata: &'a AnalysisRunResult,
    field_inventory: &'a Vec<AnalysisField>,
    materialized_field_ids: &'a Vec<String>,
    fields: &'a Vec<PersistedFieldArtifact>,
    render_tiles: &'a Option<RenderTileIndex>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    observation_histories: &'a Vec<PersistedObservationHistory>,
    #[serde(skip_serializing_if = "Option::is_none")]
    original_fields:
        &'a Option<runmat_analysis_fea::results::field::RecoveredFieldArtifactManifest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dynamic_balance: &'a Option<PersistedDynamicBalance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observation_payload: &'a Option<AnalysisObjectManifest>,
}

#[derive(Serialize)]
struct FrequencyDigestView<'a, S, C, R> {
    schema_version: &'a String,
    op_version: &'a String,
    step: &'a S,
    object_catalog: &'a C,
    artifact_roots: &'a R,
}

pub(super) fn record_digest(record: &PersistedAnalysisArtifact) -> Result<Digest, String> {
    const DOMAIN: &str = "analysis-publication-record";
    match record {
        PersistedAnalysisArtifact::Standard(r) => canonical_digest(
            DOMAIN,
            &StandardDigestView {
                schema_version: &r.schema_version,
                op_version: &r.op_version,
                run_digest: &r.run_digest,
                run: &r.run,
                metadata: &r.metadata,
                field_inventory: &r.field_inventory,
                materialized_field_ids: &r.materialized_field_ids,
                fields: &r.fields,
                render_tiles: &r.render_tiles,
                observation_histories: &r.observation_histories,
                original_fields: &r.original_fields,
                dynamic_balance: &r.dynamic_balance,
                observation_payload: &r.observation_payload,
            },
        ),
        PersistedAnalysisArtifact::StructuralFrequency(r) => canonical_digest(
            DOMAIN,
            &FrequencyDigestView {
                schema_version: &r.schema_version,
                op_version: &r.op_version,
                step: &r.step,
                object_catalog: &r.object_catalog,
                artifact_roots: &r.artifact_roots,
            },
        ),
        PersistedAnalysisArtifact::AcousticFrequency(r) => canonical_digest(
            DOMAIN,
            &FrequencyDigestView {
                schema_version: &r.schema_version,
                op_version: &r.op_version,
                step: &r.step,
                object_catalog: &r.object_catalog,
                artifact_roots: &r.artifact_roots,
            },
        ),
    }
}

#[cfg(test)]
fn tree_record_digest(record: &PersistedAnalysisArtifact) -> Result<Digest, String> {
    let mut value = record_value(record)?;
    value
        .as_object_mut()
        .ok_or("publication record is not an object")?
        .remove("created_at");
    canonical_digest("analysis-publication-record", &value)
}

impl AnalysisPublicationBuilder {
    pub fn new(
        definition: Digest,
        maximum_staging_bytes: u64,
    ) -> Result<Self, AnalysisPublicationError> {
        if maximum_staging_bytes == 0 {
            return Err(AnalysisPublicationError::ResourceLimit {
                message: "publication staging budget must be positive".into(),
            });
        }
        Ok(Self {
            definition,
            maximum_bytes: maximum_staging_bytes,
            preparation_resident_bytes: 0,
            source_workspace_bytes: 0,
            source_profiles: BTreeMap::new(),
            source_phase: Arc::new(std::sync::Mutex::new(None)),
            preparation_scope: Arc::new(()),
            evidence_access: crate::analysis::structural_study::publication::evidence::StudyEvidenceAccess::capture(),
            admission: std::sync::Mutex::new(false),
            store: InMemoryAnalysisArtifactStore::staging(),
            spill: std::sync::Mutex::new(spill::SpilledChunks::default()),
        })
    }
    fn admit(&self, additional: u64) -> Result<(), AnalysisPublicationError> {
        let runs = self
            .store
            .runs
            .read()
            .map_err(|_| "staging lock poisoned")?;
        let chunks = self
            .store
            .chunks
            .read()
            .map_err(|_| "staging lock poisoned")?;
        let observations = self
            .store
            .observations
            .read()
            .map_err(|_| "staging lock poisoned")?;
        let retained = resources::retained(&runs, &chunks, &observations, self.maximum_bytes)?;
        let source_workspace = self.active_source_workspace()?;
        let total = retained
            .checked_add(self.preparation_resident_bytes)
            .and_then(|bytes| bytes.checked_add(source_workspace))
            .and_then(|bytes| bytes.checked_add(additional))
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "staging byte overflow".into(),
            })?;
        if total > self.maximum_bytes {
            return Err(AnalysisPublicationError::ResourceLimit {
                message: format!(
                    "publication staging exceeds its byte budget: retained={retained}, preparation={}, source_workspace={source_workspace}, additional={additional}, required={total}, maximum={} bytes",
                    self.preparation_resident_bytes, self.maximum_bytes,
                ),
            });
        }
        Ok(())
    }
    pub fn stage_run(
        &self,
        run: &AnalysisRunResult,
    ) -> Result<AnalysisArtifactRecord, AnalysisPublicationError> {
        let guard = self
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")?;
        if *guard {
            return Err("publication inventory is frozen for manifest preparation".into());
        }
        validate_run_identity(run)?;
        let mut requested = 0u64;
        let mut denied = None;
        let mut spill = self
            .spill
            .lock()
            .map_err(|_| "publication spill lock poisoned")?;
        // Field payloads stream to the spill area: each is charged only while in flight, on
        // top of what staging already retains, and released once its chunks are written.
        struct Streamed<'a> {
            builder: &'a AnalysisPublicationBuilder,
            run: &'a AnalysisRunResult,
            spill: &'a mut spill::SpilledChunks,
            denied: Option<AnalysisPublicationError>,
        }
        impl super::preparation::FieldChunkSink for Streamed<'_> {
            fn admit_in_flight(&mut self, bytes: u64) -> Result<(), String> {
                self.builder.admit(bytes).map_err(|error| {
                    let error = error.context(&run_admission::run_failure_facts(
                        self.run,
                        self.builder.maximum_bytes,
                    ));
                    let message = error.to_string();
                    self.denied = Some(error);
                    message
                })
            }
            fn put(&mut self, digest: Digest, bytes: &[u8]) -> Result<(), String> {
                self.spill.put(digest, bytes)?;
                self.builder
                    .store
                    .streamed
                    .write()
                    .map_err(|_| "streamed chunk lock poisoned")?
                    .insert(digest);
                Ok(())
            }
        }
        let mut streamed = Streamed {
            builder: self,
            run,
            spill: &mut spill,
            denied: None,
        };
        let prepared = prepare_run_artifact_with_admission(
            run,
            Utc::now().to_rfc3339(),
            &mut |bytes| {
                requested = requested
                    .checked_add(bytes)
                    .ok_or("staging byte overflow")?;
                self.admit(requested).map_err(|error| {
                    let error =
                        error.context(&run_admission::run_failure_facts(run, self.maximum_bytes));
                    let message = error.to_string();
                    denied = Some(error);
                    message
                })
            },
            Some(&mut streamed),
        );
        if let Some(error) = streamed.denied.take().or(denied) {
            return Err(error);
        }
        drop(spill);
        let prepared = prepared?;
        self.store
            .persist_prepared_run(run, prepared)
            .map_err(Into::into)
    }
    pub fn stage_structural_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        catalog: &runmat_execution::value::PublishedObjectCatalog,
        roots: &crate::analysis::StructuralFrequencyArtifactRootIndex,
        render_topology: Option<&crate::analysis::AnalysisRenderTopology>,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, AnalysisPublicationError> {
        let _source_phase = self.begin_catalog_phase(&catalog.objects)?;
        let guard = self
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")?;
        if *guard {
            return Err("publication inventory is frozen for manifest preparation".into());
        }
        self.admit(frequency_bytes(step, catalog, roots)?)
            .map_err(|error| error.context("frequency step typed copies and published objects"))?;
        self.store
            .persist_structural_frequency(step, catalog, roots, render_topology, source)
            .map_err(Into::into)
    }
    pub fn stage_acoustic_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        catalog: &runmat_execution::value::PublishedObjectCatalog,
        roots: &crate::analysis::FrequencySampleArtifactRootInventory,
        render_topology: Option<&crate::analysis::AnalysisRenderTopology>,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, AnalysisPublicationError> {
        let _source_phase = self.begin_catalog_phase(&catalog.objects)?;
        let guard = self
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")?;
        if *guard {
            return Err("publication inventory is frozen for manifest preparation".into());
        }
        self.admit(frequency_bytes(step, catalog, roots)?)
            .map_err(|error| error.context("frequency step typed copies and published objects"))?;
        self.store
            .persist_acoustic_frequency(step, catalog, roots, render_topology, source)
            .map_err(Into::into)
    }
    /// Reads staged chunks where staging wrote them: records, tiles and histories stay resident
    /// in the staging store, while field payloads stream to the spill area. Every reader of a
    /// staged record goes through here.
    pub(in crate::analysis::storage) fn read_staged_chunks<R>(
        &self,
        read: impl FnOnce(
            &mut dyn FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
        ) -> Result<R, String>,
    ) -> Result<R, String> {
        let chunks = self
            .store
            .chunks
            .read()
            .map_err(|_| "publication chunks poisoned")?;
        let spilled = self
            .spill
            .lock()
            .map_err(|_| "publication spill lock poisoned")?;
        read(&mut |digest| match chunks.get(&digest) {
            Some(bytes) => Ok(Some(bytes.to_vec())),
            None => spilled.read(&digest),
        })
    }
    /// The staged standard run `run_id`, materialized with its field values.
    pub(in crate::analysis::storage) fn load_staged_run(
        &self,
        run_id: &str,
    ) -> Result<AnalysisRunResult, String> {
        let runs = self
            .store
            .runs
            .read()
            .map_err(|_| "publication staging store poisoned")?;
        let Some(PersistedAnalysisArtifact::Standard(persisted)) = runs.get(run_id) else {
            return Err("staged standard run is absent".into());
        };
        validate_persisted_run(persisted, run_id)?;
        self.read_staged_chunks(|read| materialize_run(persisted, read))
    }
    pub fn stage_thermal_observations(
        &self,
        run_id: &str,
        completed: &crate::analysis::ThermalStudyCompletion,
        source: &crate::analysis::ThermalStudyObjectSource,
    ) -> Result<(), AnalysisPublicationError> {
        let phase = self.begin_catalog_phase(&completed.catalog.objects)?;
        {
            let guard = self
                .admission
                .lock()
                .map_err(|_| "staging admission lock poisoned")?;
            if *guard {
                return Err("publication inventory is frozen for manifest preparation".into());
            }
        }
        let metadata = resident(
            &(
                &completed.manifest,
                &completed.catalog,
                &completed.observations,
            ),
            self.maximum_bytes,
        )?;
        completed.catalog.validate().map_err(|e| e.to_string())?;
        let bytes = completed
            .catalog
            .objects
            .iter()
            .try_fold(metadata, |total, object| {
                total
                    .checked_add(object.encoded_length.checked_mul(3).ok_or_else(|| {
                        AnalysisPublicationError::ResourceLimit {
                            message: "staging byte overflow".into(),
                        }
                    })?)
                    .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                        message: "staging byte overflow".into(),
                    })
            })?;
        let run = self.load_staged_run(run_id)?;
        let attachment = self
            .store
            .prepare_observations(run_id, &run, completed, source)?;
        phase.with_transport(self, bytes, || {
            self.store
                .install_observations(run_id, attachment, source)
                .map_err(Into::into)
        })
    }
    pub fn load_assessment_metadata(
        &self,
        run_id: &str,
        maximum_bytes: u64,
    ) -> Result<Option<AssessmentRunEvidence>, AssessmentStoreError> {
        self.store.load_assessment_metadata(run_id, maximum_bytes)
    }
    pub fn finish(
        self,
        manifest: serde_json::Value,
    ) -> Result<PreparedAnalysisPublication, AnalysisPublicationError> {
        if self.source_phase_active()? {
            return Err("publication source phase is still active".into());
        }
        {
            let runs = self
                .store
                .runs
                .read()
                .map_err(|_| "staging lock poisoned")?;
            let observations = self
                .store
                .observations
                .read()
                .map_err(|_| "staging lock poisoned")?;
            let additional = resources::commit_work(
                self.definition,
                &manifest,
                &runs,
                &observations,
                self.maximum_bytes,
            )?;
            drop(observations);
            drop(runs);
            self.admit(additional.total).map_err(|error| {
                error.context(&format!("publication commit phase ({})", additional.facts))
            })?;
        }
        if manifest.get("study_definition_digest")
            != Some(&serde_json::to_value(self.definition).map_err(|e| e.to_string())?)
        {
            return Err("publication manifest differs from original definition".into());
        }
        let runs = self
            .store
            .runs
            .into_inner()
            .map_err(|_| "staging lock poisoned")?;
        let chunks = self
            .store
            .chunks
            .into_inner()
            .map_err(|_| "staging lock poisoned")?;
        let observations = self
            .store
            .observations
            .into_inner()
            .map_err(|_| "staging lock poisoned")?;
        if runs.is_empty() {
            return Err("publication has no admitted records".into());
        }
        let mut declared = BTreeSet::new();
        for step in manifest
            .pointer("/public_run/steps")
            .and_then(serde_json::Value::as_array)
            .ok_or("publication step inventory absent")?
        {
            let id = step
                .get("run_id")
                .and_then(serde_json::Value::as_str)
                .ok_or("publication run identity absent")?;
            if !declared.insert(id.to_owned()) {
                return Err("publication repeats a run identity".into());
            }
        }
        if declared != runs.keys().map(ToOwned::to_owned).collect() {
            return Err("publication does not exactly cover staged records".into());
        }
        let records = runs
            .iter()
            .map(|(id, record)| Ok((id.clone(), record_digest(record)?)))
            .collect::<Result<_, AnalysisPublicationError>>()?;
        let observation_digests = observations
            .iter()
            .map(|(id, attachment)| {
                Ok((
                    id.clone(),
                    canonical_digest("analysis-publication-observation", attachment)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, AnalysisPublicationError>>()?;
        Ok(PreparedAnalysisPublication {
            maximum_bytes: self.maximum_bytes,
            preparation_resident_bytes: self.preparation_resident_bytes,
            descriptor: PublicationDescriptor {
                schema_version: 1,
                definition: self.definition,
                manifest_digest: canonical_digest("analysis-publication-manifest", &manifest)?,
                records,
                observations: observation_digests,
            },
            manifest,
            runs,
            chunks,
            observations,
            spilled: self
                .spill
                .into_inner()
                .map_err(|_| "publication spill lock poisoned")?,
        })
    }
}

/// Typed storage of owned in-memory values (or their clones), priced from the serialization
/// shape without a value tree.
pub(in crate::analysis) fn typed_resident(
    value: &impl Serialize,
    maximum: u64,
) -> Result<u64, AnalysisPublicationError> {
    runmat_canonical_codec::profile_typed_resident_bytes(
        value,
        CanonicalLimits::content_bounded(
            usize::try_from(maximum).map_err(|_| AnalysisPublicationError::ResourceLimit {
                message: "publication budget exceeds address space".into(),
            })?,
            65_536,
            64,
        ),
    )
    .map_err(Into::into)
}
/// Resident bytes of owned typed publication inputs and their clones. The direct codec builds
/// no value tree, so these owners are priced as typed storage, not as a materialized tree.
pub(in crate::analysis) fn resident(
    value: &impl Serialize,
    maximum: u64,
) -> Result<u64, AnalysisPublicationError> {
    typed_resident(value, maximum)
}
fn frequency_bytes(
    step: &AnalysisStudyStepRunData,
    catalog: &runmat_execution::value::PublishedObjectCatalog,
    roots: &impl Serialize,
) -> Result<u64, AnalysisPublicationError> {
    // Frequency preparation clones these typed values and copies each published
    // object; it builds no JSON tree.
    let mut bytes = runmat_canonical_codec::typed_resident_bytes_estimate(&(step, catalog, roots))?;
    catalog.validate().map_err(|e| e.to_string())?;
    for object in &catalog.objects {
        bytes = bytes.checked_add(object.encoded_length).ok_or_else(|| {
            AnalysisPublicationError::ResourceLimit {
                message: "staging byte overflow".into(),
            }
        })?;
    }
    Ok(bytes)
}

fn validate_manifest(
    identity: Digest,
    descriptor: &PublicationDescriptor,
    manifest: &serde_json::Value,
) -> Result<(), String> {
    if descriptor
        .observations
        .keys()
        .any(|id| !descriptor.records.contains_key(id))
        || descriptor.schema_version != 1
        || descriptor.identity()? != identity
        || canonical_digest("analysis-publication-manifest", manifest)?
            != descriptor.manifest_digest
    {
        return Err("publication manifest differs from its committed identity".into());
    }
    if manifest.get("study_definition_digest")
        != Some(&serde_json::to_value(descriptor.definition).map_err(|e| e.to_string())?)
    {
        return Err("publication definition differs from its manifest".into());
    }
    let mut declared = BTreeSet::new();
    for step in manifest
        .pointer("/public_run/steps")
        .and_then(serde_json::Value::as_array)
        .ok_or("publication step inventory absent")?
    {
        let id = step
            .get("run_id")
            .and_then(serde_json::Value::as_str)
            .ok_or("publication run identity absent")?;
        if !declared.insert(id.to_owned()) {
            return Err("publication repeats a run identity".into());
        }
    }
    if declared.is_empty() || declared != descriptor.records.keys().map(ToOwned::to_owned).collect()
    {
        return Err("manifest does not cover committed records".into());
    }
    Ok(())
}

pub(in crate::analysis::storage) mod reclamation;

mod outcome;
pub use outcome::AnalysisPublicationError;

/// Stages `run` in a fresh publication and reads it back through the staged-record reader,
/// returning the readback and how many chunks staging streamed to the spill area.
#[cfg(test)]
pub(in crate::analysis) fn stage_and_read_back_for_tests(
    run: &AnalysisRunResult,
) -> Result<(AnalysisRunResult, usize), String> {
    let builder =
        AnalysisPublicationBuilder::new(Digest::sha256(b"staged-run-readback"), 512 * 1024 * 1024)
            .map_err(|error| error.to_string())?;
    builder.stage_run(run).map_err(|error| error.to_string())?;
    let spilled = builder
        .spill
        .lock()
        .map_err(|_| "publication spill lock poisoned")?
        .digests()
        .count();
    Ok((builder.load_staged_run(&run.run_id)?, spilled))
}
