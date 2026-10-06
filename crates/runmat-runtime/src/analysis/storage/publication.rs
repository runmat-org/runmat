//! Privately prepared closed publications. Storage integrity is not numerical qualification.
use super::*;
use runmat_canonical_codec::{encode_contract, profile_serialized, CanonicalLimits};
use std::collections::BTreeSet;

mod assessment;
mod binding;
mod catalog;
pub(super) mod filesystem;
mod initial_fields;
pub(super) mod memory;
mod preparation;
mod source_fields;
mod source_phase;
pub(in crate::analysis) use preparation::PreparationReservation;
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
}

pub struct PreparedAnalysisPublication {
    maximum_bytes: u64,
    preparation_resident_bytes: u64,
    descriptor: PublicationDescriptor,
    manifest: serde_json::Value,
    runs: HashMap<String, PersistedAnalysisArtifact>,
    chunks: HashMap<Digest, Arc<[u8]>>,
    observations: HashMap<String, observations::ObservationAttachment>,
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
            run_ids: self.records.keys().cloned().collect(),
            manifest_artifact_path: path,
        })
    }
}

fn canonical_digest(domain: &'static str, value: &impl Serialize) -> Result<Digest, String> {
    encode_contract(
        b"runmat-analysis-publication/v1\0",
        domain,
        value,
        CanonicalLimits::new(1024 * 1024 * 1024, 100_000_000, 65_536, 64),
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

fn record_value(record: &PersistedAnalysisArtifact) -> Result<serde_json::Value, String> {
    match record {
        PersistedAnalysisArtifact::Standard(r) => serde_json::to_value(r),
        PersistedAnalysisArtifact::StructuralFrequency(r) => serde_json::to_value(r),
        PersistedAnalysisArtifact::AcousticFrequency(r) => serde_json::to_value(r),
    }
    .map_err(|e| e.to_string())
}

pub(super) fn record_digest(record: &PersistedAnalysisArtifact) -> Result<Digest, String> {
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
        self.admit(
            resident(run, self.maximum_bytes)?
                .checked_mul(2)
                .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                    message: "staging byte overflow".into(),
                })?,
        )?;
        self.store.persist_run(run).map_err(Into::into)
    }
    pub fn stage_structural_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        catalog: &runmat_execution::value::PublishedObjectCatalog,
        roots: &crate::analysis::StructuralFrequencyArtifactRootIndex,
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
        self.admit(frequency_bytes(step, catalog, roots, self.maximum_bytes)?)?;
        self.store
            .persist_structural_frequency(step, catalog, roots, source)
            .map_err(Into::into)
    }
    pub fn stage_acoustic_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        catalog: &runmat_execution::value::PublishedObjectCatalog,
        roots: &crate::analysis::FrequencySampleArtifactRootInventory,
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
        self.admit(frequency_bytes(step, catalog, roots, self.maximum_bytes)?)?;
        self.store
            .persist_acoustic_frequency(step, catalog, roots, source)
            .map_err(Into::into)
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
        let attachment = self.store.prepare_observations(run_id, completed, source)?;
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
            self.admit(additional)?;
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
        if declared != runs.keys().cloned().collect() {
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
        })
    }
}

pub(in crate::analysis) fn resident(
    value: &impl Serialize,
    maximum: u64,
) -> Result<u64, AnalysisPublicationError> {
    profile_serialized(
        value,
        CanonicalLimits::new(
            usize::try_from(maximum).map_err(|_| AnalysisPublicationError::ResourceLimit {
                message: "publication budget exceeds address space".into(),
            })?,
            100_000_000,
            65_536,
            64,
        ),
    )
    .map(|p| p.materialization_resident_bytes_estimate)
    .map_err(Into::into)
}
fn frequency_bytes(
    step: &AnalysisStudyStepRunData,
    catalog: &runmat_execution::value::PublishedObjectCatalog,
    roots: &impl Serialize,
    maximum: u64,
) -> Result<u64, AnalysisPublicationError> {
    let mut bytes = resident(&(step, catalog, roots), maximum)?;
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
    if declared.is_empty() || declared != descriptor.records.keys().cloned().collect() {
        return Err("manifest does not cover committed records".into());
    }
    Ok(())
}

pub(in crate::analysis::storage) mod reclamation;

mod outcome;
pub use outcome::AnalysisPublicationError;
