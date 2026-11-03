use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

use chrono::{DateTime, Utc};
use runmat_analysis_core::{AnalysisField, AnalysisFieldValues};
use runmat_analysis_fea::execution::{
    chunk_numerical_payload, AnalysisNumericalObjectKind, AnalysisObjectManifest,
    DEFAULT_ANALYSIS_CHUNK_BYTES,
};
use runmat_execution::Digest;
use runmat_filesystem::{AtomicCreateResult, DirEntry, FsFileType, FsMetadata};
use serde::{Deserialize, Serialize};

use runmat_analysis_fea::AnalysisRunIdentity;

use super::contracts::{
    AnalysisArtifactRecord, AnalysisFieldDescriptor, AnalysisFieldPageResult,
    AnalysisFieldPageValues, AnalysisFieldRequestOptions, AnalysisFieldSelection,
    AnalysisRunResult, AnalysisStudyStepRunData,
};
use super::render_tiles::{
    self, RenderFieldSummary, RenderTile, RenderTileIndex, RenderViewRequest, RenderViewSelection,
};

mod assessment;
mod initial_fields;
mod metadata_identity;
pub use initial_fields::{
    export_initial_field, resolve_initial_field_references, resolve_initial_fields,
};
use metadata_identity::persisted_run_digest;
mod discovery;
pub use assessment::{AssessmentRunEvidence, AssessmentStoreError};
mod filesystem;
mod frequency;
pub(super) mod frequency_page;
mod frequency_sample;
mod in_memory;
mod object_payload;
mod observations;
pub use observations::{
    load_observation_history_sample, load_observation_publication, persist_thermal_observations,
    ObservationHistoryLookup,
};

use frequency::PersistedStructuralFrequencyArtifact;
use frequency_sample::PersistedAcousticFrequencyArtifact;

pub(super) const ARTIFACT_SCHEMA_VERSION: &str = "analysis_run_artifact/2026-09-002";
const FIELD_IDENTITY_PREFIX: &[u8] = b"runmat-analysis-field/v1\0";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedRunArtifact {
    schema_version: String,
    created_at: String,
    op_version: String,
    run_digest: Digest,
    /// Untyped run content of records written before the typed metadata became the single copy;
    /// absent (null) for current records, whose content is `metadata` plus `observation_payload`.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    run: serde_json::Value,
    metadata: AnalysisRunResult,
    field_inventory: Vec<AnalysisField>,
    materialized_field_ids: Vec<String>,
    fields: Vec<PersistedFieldArtifact>,
    render_tiles: Option<RenderTileIndex>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    observation_histories: Vec<PersistedObservationHistory>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    original_fields: Option<runmat_analysis_fea::results::field::RecoveredFieldArtifactManifest>,
    /// Transient dynamic-balance vectors (applied, inertia, damping, elastic, residual per
    /// accepted state), stored as one chunked numerical object instead of inline JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dynamic_balance: Option<PersistedDynamicBalance>,
    /// The run's materialized observations and series reductions, chunked out of the metadata so
    /// the typed metadata is the record's only copy of the run content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    observation_payload: Option<AnalysisObjectManifest>,
}

impl PersistedRunArtifact {
    /// The authenticated run content as JSON: the legacy untyped copy when present, otherwise
    /// the typed metadata (the single copy of current records).
    fn run_projection(&self) -> Result<std::borrow::Cow<'_, serde_json::Value>, String> {
        if self.run.is_null() {
            serde_json::to_value(&self.metadata)
                .map(std::borrow::Cow::Owned)
                .map_err(|error| format!("failed to project analysis run metadata: {error}"))
        } else {
            Ok(std::borrow::Cow::Borrowed(&self.run))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedDynamicBalance {
    /// Degrees of freedom of each snapshot; each snapshot stores five vectors of that length.
    dof_counts: Vec<u64>,
    manifest: AnalysisObjectManifest,
}

/// Histories own sample-local field identities; they cannot be flattened into the
/// run's field-id index. Keep their canonical objects and authenticate their roots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedObservationHistory {
    identity: runmat_analysis_fea::results::observation::ObservationHistoryIdentity,
    observation_id: runmat_analysis_core::ObservationId,
    manifest: AnalysisObjectManifest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedFieldArtifact {
    field_id: String,
    shape: Vec<usize>,
    template: AnalysisField,
    manifest: AnalysisObjectManifest,
}

struct PreparedRunArtifact {
    persisted: PersistedRunArtifact,
    chunks: HashMap<Digest, Vec<u8>>,
}

#[derive(Clone)]
enum PersistedAnalysisArtifact {
    Standard(Box<PersistedRunArtifact>),
    StructuralFrequency(Box<PersistedStructuralFrequencyArtifact>),
    AcousticFrequency(Box<PersistedAcousticFrequencyArtifact>),
}

pub enum AnalysisFieldPageLookup {
    RunNotFound,
    FieldNotFound,
    FieldSelectionAmbiguous(Vec<String>),
    SampleSelectionRequired(Vec<runmat_analysis_fea::results::field::RecoveredFieldSample>),
    SampleNotFound,
    Page(Box<AnalysisFieldPageResult>),
}

pub(in crate::analysis) mod publication;
pub use publication::{
    AnalysisPublicationBuilder, AnalysisPublicationError, AnalysisPublicationReceipt,
    PreparedAnalysisPublication,
};
mod store_contract;
pub use store_contract::AnalysisArtifactStore;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisArtifactStoreConfig {
    InMemory,
    Filesystem { root: PathBuf },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnalysisArtifactRetentionConfig {
    pub max_runs: Option<usize>,
    pub max_runs_per_kind: Option<usize>,
}

pub struct InMemoryAnalysisArtifactStore {
    // Private builder staging is governed by its byte cap; retention applies at commit.
    retain_standalone: bool,
    publications: RwLock<HashMap<Digest, Vec<u8>>>,
    publication_runs: RwLock<HashMap<String, Digest>>,
    observations: RwLock<HashMap<String, observations::ObservationAttachment>>,
    runs: RwLock<HashMap<String, PersistedAnalysisArtifact>>,
    chunks: RwLock<HashMap<Digest, Arc<[u8]>>>,
    /// Staging only: field chunks streamed to the publication spill area. Records may
    /// reference them although their bytes are not resident here.
    streamed: RwLock<std::collections::HashSet<Digest>>,
}

impl Default for InMemoryAnalysisArtifactStore {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FilesystemAnalysisArtifactStore {
    provider: Arc<dyn runmat_filesystem::FsProvider>,
    root: PathBuf,
    manifests: RwLock<FilesystemManifestCache>,
}

#[derive(Default)]
struct FilesystemManifestCache {
    entries: HashMap<String, Arc<PersistedRunArtifact>>,
    order: VecDeque<String>,
    identities: HashMap<String, Digest>,
}

impl FilesystemAnalysisArtifactStore {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            provider: runmat_filesystem::current_provider(),
            manifests: RwLock::new(FilesystemManifestCache::default()),
        }
    }

    fn run_path(&self, run_id: &str) -> Result<PathBuf, String> {
        validate_run_id_token(run_id)?;
        Ok(self.root.join("runs").join(format!("{run_id}.json")))
    }
}

impl FilesystemAnalysisArtifactStore {
    fn objects_directory(&self) -> PathBuf {
        self.root.join("objects")
    }

    fn object_path(&self, digest: Digest) -> PathBuf {
        self.objects_directory().join(digest_hex(digest))
    }
}

fn decode_persisted_run(
    bytes: &[u8],
    expected_run_id: &str,
) -> Result<PersistedRunArtifact, String> {
    validate_run_id_token(expected_run_id)?;
    let persisted = serde_json::from_slice::<PersistedRunArtifact>(bytes)
        .map_err(|error| format!("failed to parse run artifact: {error}"))?;
    validate_persisted_run(&persisted, expected_run_id)?;
    Ok(persisted)
}

fn decode_persisted_artifact(
    bytes: &[u8],
    expected_run_id: &str,
) -> Result<PersistedAnalysisArtifact, String> {
    let value = serde_json::from_slice::<serde_json::Value>(bytes)
        .map_err(|error| format!("failed to parse analysis run artifact: {error}"))?;
    let schema = value
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "analysis run artifact omits its schema version".to_string())?;
    match schema {
        ARTIFACT_SCHEMA_VERSION => decode_persisted_run(bytes, expected_run_id)
            .map(Box::new)
            .map(PersistedAnalysisArtifact::Standard),
        frequency::FREQUENCY_ARTIFACT_SCHEMA_VERSION => {
            frequency::decode_artifact(bytes, expected_run_id)
                .map(Box::new)
                .map(PersistedAnalysisArtifact::StructuralFrequency)
        }
        frequency_sample::ACOUSTIC_FREQUENCY_ARTIFACT_SCHEMA_VERSION => {
            frequency_sample::decode_artifact(bytes, expected_run_id)
                .map(Box::new)
                .map(PersistedAnalysisArtifact::AcousticFrequency)
        }
        other => Err(format!(
            "unsupported analysis run artifact schema `{other}`"
        )),
    }
}

mod field_reader;
use field_reader::{field_page_from_artifact, materialize_run, persisted_field_inventory};
mod preparation;
use preparation::{prepare_run_artifact, prepare_run_artifact_with_admission};

fn same_persisted_content(left: &PersistedRunArtifact, right: &PersistedRunArtifact) -> bool {
    left.schema_version == right.schema_version
        && left.op_version == right.op_version
        && left.run_digest == right.run_digest
        && left.run == right.run
        && left.metadata == right.metadata
        && left.field_inventory == right.field_inventory
        && left.materialized_field_ids == right.materialized_field_ids
        && left.fields == right.fields
        && left.render_tiles == right.render_tiles
        && left.observation_histories == right.observation_histories
}

fn validate_chunk(
    descriptor: &runmat_analysis_fea::execution::AnalysisChunkDescriptor,
    bytes: &[u8],
) -> Result<(), String> {
    if bytes.len() as u64 != descriptor.encoded_length || Digest::sha256(bytes) != descriptor.digest
    {
        return Err(format!(
            "analysis field chunk {} differs from its manifest descriptor",
            descriptor.ordinal
        ));
    }
    Ok(())
}

fn encode_field_values(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn decode_field_values(bytes: &[u8]) -> Result<Vec<f64>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f64>()) {
        return Err("analysis field payload is not an integral number of f64 values".to_string());
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<f64>())
        .map(|chunk| {
            let mut encoded = [0_u8; std::mem::size_of::<f64>()];
            encoded.copy_from_slice(chunk);
            f64::from_bits(u64::from_le_bytes(encoded))
        })
        .collect())
}

fn field_logical_identity(field: &AnalysisField, payload_digest: Digest) -> Digest {
    let mut identity = Vec::with_capacity(
        FIELD_IDENTITY_PREFIX.len() + field.field_id.len() + field.shape.len() * 8 + 48,
    );
    identity.extend_from_slice(FIELD_IDENTITY_PREFIX);
    identity.extend_from_slice(&(field.field_id.len() as u64).to_le_bytes());
    identity.extend_from_slice(field.field_id.as_bytes());
    identity.extend_from_slice(&(field.shape.len() as u64).to_le_bytes());
    for dimension in &field.shape {
        identity.extend_from_slice(&(*dimension as u64).to_le_bytes());
    }
    identity.extend_from_slice(payload_digest.bytes());
    Digest::sha256(identity)
}

fn checked_element_count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| "analysis field shape element count overflow".to_string())
    })
}

fn validate_field_id(field_id: &str) -> Result<(), String> {
    if field_id.is_empty() || field_id.len() > 512 || field_id.chars().any(char::is_control) {
        return Err("analysis field ID must be 1-512 non-control characters".to_string());
    }
    Ok(())
}

fn digest_hex(digest: Digest) -> String {
    digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn compact_run_metadata(run: &serde_json::Value) -> Result<AnalysisRunResult, String> {
    // Persisted field shells intentionally have no numerical values. Remove the two
    // materialized observation collections before their strict numerical decoders run.
    let mut projection = run.clone();
    let numerical_run = projection
        .get_mut("run")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| "authenticated analysis run has no numerical run".to_string())?;
    numerical_run.insert("observations".into(), serde_json::Value::Array(Vec::new()));
    numerical_run.insert(
        "series_reductions".into(),
        serde_json::Value::Array(Vec::new()),
    );
    serde_json::from_value(projection)
        .map_err(|error| format!("invalid authenticated compact analysis run: {error}"))
}

fn validate_persisted_run(
    persisted: &PersistedRunArtifact,
    expected_run_id: &str,
) -> Result<(), String> {
    if persisted.schema_version != ARTIFACT_SCHEMA_VERSION {
        return Err(format!(
            "unsupported analysis run artifact schema `{}`",
            persisted.schema_version
        ));
    }
    let run_value = persisted.run_projection()?;
    if persisted_run_digest(
        &run_value,
        &persisted.field_inventory,
        &persisted.materialized_field_ids,
        persisted.render_tiles.as_ref(),
        &persisted.observation_histories,
        persisted.original_fields.as_ref(),
        persisted.dynamic_balance.as_ref(),
        persisted.observation_payload.as_ref(),
    )? != persisted.run_digest
    {
        return Err("analysis run metadata differs from its content digest".into());
    }
    if persisted.metadata.run_id != expected_run_id {
        return Err(format!(
            "analysis run artifact key `{expected_run_id}` does not match embedded run ID `{}`",
            persisted.metadata.run_id
        ));
    }
    let expected_op_version = run_operation_version(&persisted.metadata)?;
    if persisted.op_version != expected_op_version {
        return Err(format!(
            "analysis run artifact operation version `{}` does not match derived version `{expected_op_version}`",
            persisted.op_version
        ));
    }
    let mut history_ids = HashSet::new();
    for history in &persisted.observation_histories {
        history
            .manifest
            .validate()
            .map_err(|error| error.to_string())?;
        if history.manifest.kind != AnalysisNumericalObjectKind::History
            || history.manifest.logical_identity != history.identity.digest()
            || !history_ids.insert(&history.observation_id)
        {
            return Err(
                "observation history inventory has invalid or duplicated identities".into(),
            );
        }
    }
    if let Some(manifest) = &persisted.observation_payload {
        manifest.validate().map_err(|error| error.to_string())?;
        if manifest.kind != AnalysisNumericalObjectKind::History {
            return Err("observation payload manifest has the wrong numerical object kind".into());
        }
    }
    // A legacy record carries a second, untyped copy that must agree with the typed metadata.
    if !persisted.run.is_null() {
        let authenticated_scalars: Vec<
            runmat_analysis_fea::results::assessment::ScalarObservationEvidence,
        > = match persisted.run.get("observation_scalars") {
            Some(value) => serde_json::from_value(value.clone())
                .map_err(|error| format!("invalid scalar observation inventory: {error}"))?,
            None => Vec::new(),
        };
        if authenticated_scalars != persisted.metadata.observation_scalars {
            return Err(
                "scalar observation inventory differs from authenticated run content".into(),
            );
        }
        if compact_run_metadata(&persisted.run)? != persisted.metadata {
            return Err("compact analysis metadata differs from authenticated run content".into());
        }
    } else if !persisted.metadata.run.observations.is_empty()
        || !persisted.metadata.run.series_reductions.is_empty()
    {
        return Err("analysis metadata must not carry inline observation payloads".into());
    }
    validate_run_identity(&persisted.metadata)?;
    if persisted.metadata.render_topology.is_some() {
        return Err("render topology must be stored in tile objects".into());
    }
    if let Some(index) = &persisted.render_tiles {
        index.validate()?;
    }
    let mut expected_fields = BTreeMap::new();
    collect_persisted_field_shells(&run_value, &mut expected_fields)?;
    let mut inventory_fields = BTreeMap::new();
    for field in &persisted.field_inventory {
        validate_field_id(&field.field_id)?;
        if !matches!(
            &field.values,
            AnalysisFieldValues::HostF64(values) if values.is_empty()
        ) || inventory_fields
            .insert(field.field_id.clone(), field.shape.clone())
            .is_some()
        {
            return Err(format!(
                "analysis field `{}` has invalid or duplicate inventory metadata",
                field.field_id
            ));
        }
    }
    // Current records chunk their observations (and the field shells inside them) out of the
    // metadata, so the authenticated inventory covers the metadata shells and those of the
    // observation payload; legacy records list exactly the shells of their untyped copy.
    let inventory_matches = if persisted.run.is_null() {
        expected_fields
            .iter()
            .all(|(field_id, shape)| inventory_fields.get(field_id) == Some(shape))
    } else {
        inventory_fields == expected_fields
    };
    if !inventory_matches {
        return Err("analysis run field inventory differs from its metadata shells".into());
    }
    let expected_fields = if persisted.run.is_null() {
        inventory_fields
    } else {
        expected_fields
    };
    let mut stored_fields = BTreeMap::new();
    for field in &persisted.fields {
        validate_field_id(&field.field_id)?;
        if field.template.field_id != field.field_id
            || field.template.shape != field.shape
            || !matches!(
                &field.template.values,
                AnalysisFieldValues::HostF64(values) if values.is_empty()
            )
        {
            return Err(format!(
                "analysis field `{}` template differs from its persisted metadata",
                field.field_id
            ));
        }
        field
            .manifest
            .validate()
            .map_err(|error| format!("invalid field manifest for `{}`: {error}", field.field_id))?;
        if field.manifest.kind != AnalysisNumericalObjectKind::Field {
            return Err(format!(
                "analysis field `{}` manifest has the wrong numerical object kind",
                field.field_id
            ));
        }
        let value_count = checked_element_count(&field.shape)?;
        let expected_bytes = value_count
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| "analysis field byte length overflow".to_string())?;
        if field.manifest.encoded_length != expected_bytes as u64
            || field_logical_identity(
                &AnalysisField::host_f64(&field.field_id, field.template.layout.clone(), field.shape.clone(), Vec::new()),
                field.manifest.payload_digest,
            ) != field.manifest.logical_identity
        {
            return Err(format!(
                "analysis field `{}` manifest is not bound to its field metadata",
                field.field_id
            ));
        }
        if stored_fields
            .insert(field.field_id.clone(), field.shape.clone())
            .is_some()
        {
            return Err(format!(
                "analysis field `{}` appears more than once in the run manifest",
                field.field_id
            ));
        }
    }
    let stored_field_ids = stored_fields
        .keys()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if persisted.materialized_field_ids != stored_field_ids
        || stored_fields
            .iter()
            .any(|(field_id, shape)| expected_fields.get(field_id) != Some(shape))
    {
        return Err(
            "analysis run field inventory differs from its persisted manifests".to_string(),
        );
    }
    initial_fields::validate_catalog(persisted)?;
    Ok(())
}

fn collect_persisted_field_shells(
    value: &serde_json::Value,
    fields: &mut BTreeMap<String, Vec<usize>>,
) -> Result<(), String> {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                collect_persisted_field_shells(value, fields)?;
            }
        }
        serde_json::Value::Object(object) => {
            if object.contains_key("field_id")
                && object.contains_key("shape")
                && object.contains_key("values")
            {
                let field = serde_json::from_value::<AnalysisField>(serde_json::Value::Object(
                    object.clone(),
                ))
                .map_err(|error| format!("invalid persisted analysis field shell: {error}"))?;
                validate_field_id(&field.field_id)?;
                let AnalysisFieldValues::HostF64(values) = &field.values else {
                    return Err(format!(
                        "analysis field `{}` shell is not canonical host storage",
                        field.field_id
                    ));
                };
                if !values.is_empty() {
                    return Err(format!(
                        "analysis field `{}` shell is not empty",
                        field.field_id
                    ));
                }
                checked_element_count(&field.shape)?;
                if fields.insert(field.field_id.clone(), field.shape).is_some() {
                    return Err(format!(
                        "analysis field `{}` appears more than once in the run",
                        field.field_id
                    ));
                }
                return Ok(());
            }
            for value in object.values() {
                collect_persisted_field_shells(value, fields)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_owned_directory_metadata(
    metadata: FsMetadata,
    path: &Path,
    label: &str,
) -> Result<(), String> {
    if metadata.file_type() == FsFileType::Directory {
        Ok(())
    } else {
        Err(format!(
            "{label} must be a real directory and may not be a symlink: {}",
            path.display()
        ))
    }
}

fn run_operation_version(run: &AnalysisRunResult) -> Result<String, String> {
    super::run_operation_for_identity(&run.run.identity)
        .map(|operation| operation.version().to_string())
        .ok_or_else(|| "analysis run identity has no registered runtime operation".to_string())
}

fn artifact_record(
    run: &AnalysisRunResult,
    created_at: String,
) -> Result<AnalysisArtifactRecord, String> {
    Ok(AnalysisArtifactRecord {
        run_id: run.run_id.clone(),
        created_at,
        identity: run.run.identity.clone(),
        op_version: run_operation_version(run)?,
        field_ids: super::analysis_run_field_ids(run),
    })
}

fn validate_run_identity(run: &AnalysisRunResult) -> Result<(), String> {
    validate_run_id_token(&run.run_id)?;
    let identity = &run.run.identity;
    super::run_operation_for_identity(identity)
        .ok_or_else(|| "analysis run identity has no registered runtime operation".to_string())?;
    let expected_status =
        super::result_policy::classify_run_status(run.result_quality, run.numerically_acceptable);
    if run.run_status != expected_status {
        return Err(format!(
            "analysis run status {:?} does not match numerical outcome {:?}",
            run.run_status, expected_status
        ));
    }
    Ok(())
}

fn validate_run_id_token(run_id: &str) -> Result<(), String> {
    const MAX_RUN_ID_BYTES: usize = 128;
    if run_id.is_empty()
        || run_id.len() > MAX_RUN_ID_BYTES
        || !run_id.is_ascii()
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(format!(
            "analysis run ID must be a 1-{MAX_RUN_ID_BYTES} byte ASCII token containing only letters, digits, `_`, or `-`"
        ));
    }
    Ok(())
}

mod access;
mod retention;
use retention::prune_filesystem_runs;
#[derive(Debug)]
struct FilesystemRunArtifact {
    path: PathBuf,
    identity: AnalysisRunIdentity,
    run_id: String,
    created_at: Option<DateTime<Utc>>,
    modified_at: Option<DateTime<Utc>>,
}

impl FilesystemRunArtifact {
    fn recorded_at(&self) -> Option<DateTime<Utc>> {
        self.created_at.or(self.modified_at)
    }
}

fn parse_recorded_at(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|value| value.with_timezone(&Utc))
}

fn generated_run_id_order(run_id: &str) -> Option<(i64, u64)> {
    let suffix = run_id.strip_prefix("run_")?;
    let (timestamp_millis, sequence) = suffix.rsplit_once('_')?;
    Some((timestamp_millis.parse().ok()?, sequence.parse().ok()?))
}

fn sort_filesystem_artifacts_newest_first(artifacts: &mut [FilesystemRunArtifact]) {
    artifacts.sort_by(|a, b| {
        b.recorded_at()
            .cmp(&a.recorded_at())
            .then_with(|| generated_run_id_order(&b.run_id).cmp(&generated_run_id_order(&a.run_id)))
            .then_with(|| b.run_id.cmp(&a.run_id))
            .then_with(|| b.path.cmp(&a.path))
    });
}

fn global_store() -> &'static RwLock<Arc<dyn AnalysisArtifactStore>> {
    static STORE: OnceLock<RwLock<Arc<dyn AnalysisArtifactStore>>> = OnceLock::new();
    STORE.get_or_init(|| {
        let default = store_from_config(config_from_env());
        RwLock::new(default)
    })
}

fn retention_config() -> &'static RwLock<AnalysisArtifactRetentionConfig> {
    static CONFIG: OnceLock<RwLock<AnalysisArtifactRetentionConfig>> = OnceLock::new();
    CONFIG.get_or_init(|| RwLock::new(AnalysisArtifactRetentionConfig::default()))
}

static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_run_id() -> String {
    let seq = NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed);
    format!("run_{}_{}", Utc::now().timestamp_millis(), seq)
}

pub fn persist_run_result(run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String> {
    crate::analysis::fea_run_admission::FeaRunScope::check_publication_checked()
        .map_err(|denied| denied.to_string())?;
    // Every published field declares its layout; a layout that does not fit the stored
    // shape is a publisher defect and is never persisted.
    let mut layout_error = None;
    crate::analysis::operations::visit_analysis_result_fields(run, |field| {
        if layout_error.is_none() {
            layout_error = field.validate_layout().err();
        }
    });
    if let Some(error) = layout_error {
        return Err(format!("published field layout is invalid: {error}"));
    }
    let guard = capture_artifact_store()?;
    guard.persist_run(run)
}

pub(in crate::analysis) fn capture_artifact_store() -> Result<Arc<dyn AnalysisArtifactStore>, String>
{
    if let Some(store) = async_access::scoped_store() {
        return Ok(store);
    }
    global_store()
        .read()
        .map(|store| Arc::clone(&store))
        .map_err(|_| "analysis artifact store lock poisoned".to_owned())
}

mod source_access;
mod async_access;
pub(in crate::analysis) use async_access::scoped_artifact_provider;
pub use async_access::with_artifact_access;
pub use source_access::{
    load_acoustic_frequency_result, load_assessment_metadata, load_run_metadata, load_run_result,
    load_structural_frequency_result, resolve_run_attempt, run_attempts,
};

pub fn load_render_tile_index(run_id: &str) -> Result<Option<RenderTileIndex>, String> {
    let guard = capture_artifact_store()?;
    guard.load_render_tile_index(run_id)
}

pub fn plan_render_view(
    run_id: &str,
    view: Option<RenderViewRequest>,
) -> Result<Option<RenderViewSelection>, String> {
    let guard = capture_artifact_store()?;
    guard.plan_render_view(run_id, view)
}

pub fn load_render_tile(
    run_id: &str,
    mesh_index: usize,
    tile_index: usize,
) -> Result<Option<RenderTile>, String> {
    let guard = capture_artifact_store()?;
    guard.load_render_tile(run_id, mesh_index, tile_index)
}

pub fn load_render_lod_node(
    run_id: &str,
    mesh_index: usize,
    node_index: usize,
) -> Result<Option<RenderTile>, String> {
    let guard = capture_artifact_store()?;
    guard.load_render_lod_node(run_id, mesh_index, node_index)
}

pub fn load_render_topology(run_id: &str) -> Result<Option<super::AnalysisRenderTopology>, String> {
    let guard = capture_artifact_store()?;
    guard.load_render_topology(run_id)
}

pub fn load_field_inventory(run_id: &str) -> Result<Option<Vec<AnalysisField>>, String> {
    let guard = capture_artifact_store()?;
    guard.load_field_inventory(run_id)
}

pub fn load_field_page(
    run_id: &str,
    selection: &AnalysisFieldSelection,
    options: AnalysisFieldRequestOptions,
) -> Result<AnalysisFieldPageLookup, String> {
    let guard = capture_artifact_store()?;
    guard.load_field_page(run_id, selection, options)
}

pub fn list_run_results() -> Result<Vec<AnalysisRunResult>, String> {
    let guard = capture_artifact_store()?;
    guard.list_runs()
}

pub fn list_run_metadata() -> Result<Vec<AnalysisRunResult>, String> {
    let guard = capture_artifact_store()?;
    guard.list_run_metadata()
}

pub fn configure_artifact_store(config: AnalysisArtifactStoreConfig) -> Result<(), String> {
    let mut guard = global_store()
        .write()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    *guard = store_from_config(config);
    Ok(())
}

/// Install a retained host-owned store while preserving the exact previous
/// owner. Embedded hosts must serialize access and restore it on scope exit;
/// constructing a fresh in-memory store per command would discard results.
pub fn replace_artifact_store(
    store: Arc<dyn AnalysisArtifactStore>,
) -> Result<Arc<dyn AnalysisArtifactStore>, String> {
    let mut guard = global_store()
        .write()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    Ok(std::mem::replace(&mut *guard, store))
}

/// Scoped host counterpart of `configure_artifact_retention`.
pub fn replace_artifact_retention(
    config: AnalysisArtifactRetentionConfig,
) -> Result<AnalysisArtifactRetentionConfig, String> {
    let mut guard = retention_config()
        .write()
        .map_err(|_| "analysis artifact retention config lock poisoned".to_string())?;
    Ok(std::mem::replace(&mut *guard, config))
}

pub fn configure_artifact_retention(config: AnalysisArtifactRetentionConfig) -> Result<(), String> {
    let mut guard = retention_config()
        .write()
        .map_err(|_| "analysis artifact retention config lock poisoned".to_string())?;
    *guard = config;
    Ok(())
}

pub fn configure_artifact_store_from_env() -> Result<(), String> {
    configure_artifact_store(config_from_env())
}

fn store_from_config(config: AnalysisArtifactStoreConfig) -> Arc<dyn AnalysisArtifactStore> {
    match config {
        AnalysisArtifactStoreConfig::InMemory => Arc::new(InMemoryAnalysisArtifactStore::new()),
        AnalysisArtifactStoreConfig::Filesystem { root } => {
            Arc::new(FilesystemAnalysisArtifactStore::new(root))
        }
    }
}

fn config_from_env() -> AnalysisArtifactStoreConfig {
    let mode = std::env::var("RUNMAT_FEA_ARTIFACT_STORE")
        .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_STORE"))
        .unwrap_or_else(|_| "filesystem".to_string())
        .to_lowercase();
    if mode == "filesystem" {
        let root = std::env::var("RUNMAT_FEA_ARTIFACT_ROOT")
            .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_ROOT"))
            .map(PathBuf::from)
            .unwrap_or_else(|_| default_filesystem_artifact_root());
        AnalysisArtifactStoreConfig::Filesystem { root }
    } else {
        AnalysisArtifactStoreConfig::InMemory
    }
}

pub fn default_filesystem_artifact_root() -> PathBuf {
    PathBuf::from(".artifacts")
}

#[cfg(test)]
pub fn set_artifact_store_for_tests(store: Arc<dyn AnalysisArtifactStore>) {
    let mut guard = global_store()
        .write()
        .expect("analysis artifact store lock poisoned");
    *guard = store;
}

#[cfg(test)]
pub fn reset_artifact_store_for_tests() {
    let mut guard = global_store()
        .write()
        .expect("analysis artifact store lock poisoned");
    *guard = Arc::new(InMemoryAnalysisArtifactStore::new());
    *retention_config()
        .write()
        .expect("analysis artifact retention config lock poisoned") =
        AnalysisArtifactRetentionConfig::default();
}

#[cfg(test)]
pub(crate) fn artifact_store_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
#[path = "storage/tests.rs"]
mod tests;

#[cfg(test)]
pub(crate) mod test_support;
