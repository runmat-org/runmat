use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

use chrono::Utc;
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

mod frequency;
mod frequency_page;

use frequency::PersistedStructuralFrequencyArtifact;

const ARTIFACT_SCHEMA_VERSION: &str = "analysis_run_artifact/2026-09-validated-field-shells";
const FIELD_IDENTITY_PREFIX: &[u8] = b"runmat-analysis-field/v1\0";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedRunArtifact {
    schema_version: String,
    created_at: String,
    op_version: String,
    run_digest: Digest,
    run: serde_json::Value,
    metadata: AnalysisRunResult,
    field_inventory: Vec<AnalysisField>,
    materialized_field_ids: Vec<String>,
    fields: Vec<PersistedFieldArtifact>,
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
}

pub enum AnalysisFieldPageLookup {
    RunNotFound,
    FieldNotFound,
    FieldSelectionAmbiguous(Vec<String>),
    SampleSelectionRequired(Vec<runmat_analysis_fea::results::field::RecoveredFieldSample>),
    SampleNotFound,
    Page(Box<AnalysisFieldPageResult>),
}

pub trait AnalysisArtifactStore: Send + Sync {
    fn persist_run(&self, run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String>;
    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String>;
    fn load_run_metadata(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String>;
    fn load_field_inventory(&self, run_id: &str) -> Result<Option<Vec<AnalysisField>>, String>;
    fn load_field_page(
        &self,
        run_id: &str,
        selection: &AnalysisFieldSelection,
        options: AnalysisFieldRequestOptions,
    ) -> Result<AnalysisFieldPageLookup, String>;
    fn list_run_metadata(&self) -> Result<Vec<AnalysisRunResult>, String>;
    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String>;
    fn persist_structural_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        object_catalog: &runmat_execution::value::PublishedObjectCatalog,
        artifact_roots: &crate::analysis::StructuralFrequencyArtifactRootIndex,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String>;
    fn load_structural_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String>;
}

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
    runs: RwLock<HashMap<String, PersistedAnalysisArtifact>>,
    chunks: RwLock<HashMap<Digest, Arc<[u8]>>>,
}

impl InMemoryAnalysisArtifactStore {
    pub fn new() -> Self {
        Self {
            runs: RwLock::new(HashMap::new()),
            chunks: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryAnalysisArtifactStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalysisArtifactStore for InMemoryAnalysisArtifactStore {
    fn persist_run(&self, run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String> {
        validate_run_identity(run)?;
        let prepared = prepare_run_artifact(run, Utc::now().to_rfc3339())?;
        let mut guard = self
            .runs
            .write()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        if let Some(existing) = guard.get(&run.run_id) {
            let PersistedAnalysisArtifact::Standard(existing) = existing else {
                return Err("analysis run identity already contains another result kind".into());
            };
            validate_persisted_run(existing, &run.run_id)?;
            if !same_persisted_content(existing, &prepared.persisted) {
                return Err("analysis run identity already contains different content".into());
            }
            return artifact_record(run, existing.created_at.clone());
        }
        let created_at = prepared.persisted.created_at.clone();
        let mut chunks = self
            .chunks
            .write()
            .map_err(|_| "analysis field chunk store lock poisoned".to_string())?;
        for (digest, bytes) in prepared.chunks {
            match chunks.get(&digest) {
                Some(existing) if existing.as_ref() != bytes.as_slice() => {
                    return Err("analysis field chunk digest collision".to_string());
                }
                Some(_) => {}
                None => {
                    chunks.insert(digest, Arc::from(bytes));
                }
            }
        }
        guard.insert(
            run.run_id.clone(),
            PersistedAnalysisArtifact::Standard(Box::new(prepared.persisted)),
        );
        artifact_record(run, created_at)
    }

    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        validate_run_id_token(run_id)?;
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        let Some(PersistedAnalysisArtifact::Standard(persisted)) = guard.get(run_id) else {
            return Ok(None);
        };
        validate_persisted_run(persisted, run_id)?;
        let chunks = self
            .chunks
            .read()
            .map_err(|_| "analysis field chunk store lock poisoned".to_string())?;
        Ok(Some(materialize_run(persisted, |digest| {
            Ok(chunks.get(&digest).map(|bytes| bytes.to_vec()))
        })?))
    }

    fn load_run_metadata(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        validate_run_id_token(run_id)?;
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        let Some(PersistedAnalysisArtifact::Standard(persisted)) = guard.get(run_id) else {
            return Ok(None);
        };
        validate_persisted_run(persisted, run_id)?;
        Ok(Some(persisted.metadata.clone()))
    }

    fn load_field_inventory(&self, run_id: &str) -> Result<Option<Vec<AnalysisField>>, String> {
        validate_run_id_token(run_id)?;
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        let Some(PersistedAnalysisArtifact::Standard(persisted)) = guard.get(run_id) else {
            return Ok(None);
        };
        validate_persisted_run(persisted, run_id)?;
        Ok(Some(persisted_field_inventory(persisted)))
    }

    fn load_field_page(
        &self,
        run_id: &str,
        selection: &AnalysisFieldSelection,
        options: AnalysisFieldRequestOptions,
    ) -> Result<AnalysisFieldPageLookup, String> {
        validate_run_id_token(run_id)?;
        let runs = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        let Some(persisted) = runs.get(run_id) else {
            return Ok(AnalysisFieldPageLookup::RunNotFound);
        };
        let chunks = self
            .chunks
            .read()
            .map_err(|_| "analysis field chunk store lock poisoned".to_string())?;
        match persisted {
            PersistedAnalysisArtifact::Standard(persisted) => {
                validate_persisted_run(persisted, run_id)?;
                field_page_from_artifact(persisted, selection, options, |digest| {
                    Ok(chunks.get(&digest).map(|bytes| bytes.to_vec()))
                })
            }
            PersistedAnalysisArtifact::StructuralFrequency(artifact) => {
                frequency::field_page(artifact, selection, options, |digest| {
                    Ok(chunks.get(&digest).map(|bytes| bytes.to_vec()))
                })
            }
        }
    }

    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String> {
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        guard
            .iter()
            .filter_map(|(run_id, stored)| match stored {
                PersistedAnalysisArtifact::Standard(persisted) => Some((run_id, persisted)),
                PersistedAnalysisArtifact::StructuralFrequency(_) => None,
            })
            .map(|(run_id, persisted)| {
                validate_persisted_run(persisted, run_id)?;
                let chunks = self
                    .chunks
                    .read()
                    .map_err(|_| "analysis field chunk store lock poisoned".to_string())?;
                materialize_run(persisted, |digest| {
                    Ok(chunks.get(&digest).map(|bytes| bytes.to_vec()))
                })
            })
            .collect()
    }

    fn list_run_metadata(&self) -> Result<Vec<AnalysisRunResult>, String> {
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        guard
            .iter()
            .filter_map(|(run_id, stored)| match stored {
                PersistedAnalysisArtifact::Standard(persisted) => Some((run_id, persisted)),
                PersistedAnalysisArtifact::StructuralFrequency(_) => None,
            })
            .map(|(run_id, persisted)| {
                validate_persisted_run(persisted, run_id)?;
                Ok(persisted.metadata.clone())
            })
            .collect()
    }

    fn persist_structural_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        object_catalog: &runmat_execution::value::PublishedObjectCatalog,
        artifact_roots: &crate::analysis::StructuralFrequencyArtifactRootIndex,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String> {
        let prepared = frequency::prepare_artifact(
            step,
            object_catalog,
            artifact_roots,
            Utc::now().to_rfc3339(),
        )?;
        let objects = frequency::ingest_objects(&prepared, source)?;
        let mut runs = self
            .runs
            .write()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        if let Some(existing) = runs.get(&step.run_id) {
            let PersistedAnalysisArtifact::StructuralFrequency(existing) = existing else {
                return Err("analysis run identity already contains another result kind".into());
            };
            frequency::validate_artifact(existing, &step.run_id)?;
            if !frequency::same_content(existing, &prepared) {
                return Err("analysis run identity already contains different content".into());
            }
            return Ok(frequency::artifact_record(existing));
        }
        let mut stored_objects = self
            .chunks
            .write()
            .map_err(|_| "analysis object store lock poisoned".to_string())?;
        for (digest, bytes) in objects {
            match stored_objects.get(&digest) {
                Some(existing) if existing.as_ref() != bytes.as_slice() => {
                    return Err("analysis object digest collision".into());
                }
                Some(_) => {}
                None => {
                    stored_objects.insert(digest, Arc::from(bytes));
                }
            }
        }
        let record = frequency::artifact_record(&prepared);
        runs.insert(
            step.run_id.clone(),
            PersistedAnalysisArtifact::StructuralFrequency(Box::new(prepared)),
        );
        Ok(record)
    }

    fn load_structural_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String> {
        validate_run_id_token(run_id)?;
        let runs = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        let Some(PersistedAnalysisArtifact::StructuralFrequency(artifact)) = runs.get(run_id)
        else {
            return Ok(None);
        };
        frequency::validate_artifact(artifact, run_id)?;
        Ok(Some(artifact.step.clone()))
    }
}

pub struct FilesystemAnalysisArtifactStore {
    root: PathBuf,
}

impl FilesystemAnalysisArtifactStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn run_path(&self, run_id: &str) -> Result<PathBuf, String> {
        validate_run_id_token(run_id)?;
        Ok(self.root.join("runs").join(format!("{run_id}.json")))
    }

    fn ensure_runs_directory(&self) -> Result<(), String> {
        ensure_owned_directory(&self.root, "analysis artifact root")?;
        let runs = self.root.join("runs");
        ensure_owned_directory(&runs, "analysis artifact runs directory")?;
        Ok(())
    }

    fn objects_directory(&self) -> PathBuf {
        self.root.join("objects")
    }

    fn object_path(&self, digest: Digest) -> PathBuf {
        self.objects_directory().join(digest_hex(digest))
    }

    fn ensure_objects_directory(&self) -> Result<(), String> {
        ensure_owned_directory(
            &self.objects_directory(),
            "analysis artifact objects directory",
        )
    }

    fn read_chunk(&self, digest: Digest) -> Result<Option<Vec<u8>>, String> {
        let path = self.object_path(digest);
        match fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {}
            Ok(metadata) => {
                return Err(format!(
                    "analysis field chunk must be a regular file, found {:?}",
                    metadata.file_type()
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to inspect analysis field chunk: {error}")),
        }
        let bytes =
            fs_read(path).map_err(|error| format!("failed to read field chunk: {error}"))?;
        if Digest::sha256(&bytes) != digest {
            return Err("analysis field chunk differs from its content address".to_string());
        }
        Ok(Some(bytes))
    }

    fn list_persisted_runs(&self) -> Result<Vec<PersistedAnalysisArtifact>, String> {
        let runs_dir = self.root.join("runs");
        if !owned_directory_exists(&self.root, "analysis artifact root")?
            || !owned_directory_exists(&runs_dir, "analysis artifact runs directory")?
        {
            return Ok(Vec::new());
        }
        let mut runs = Vec::new();
        for entry in
            fs_read_dir(&runs_dir).map_err(|error| format!("failed to scan artifacts: {error}"))?
        {
            let path = entry.path().to_path_buf();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            if entry.file_type() != FsFileType::File {
                return Err(format!(
                    "analysis run artifact must be a regular file: {}",
                    path.display()
                ));
            }
            let run_id = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| {
                    format!(
                        "analysis run artifact filename is invalid: {}",
                        path.display()
                    )
                })?;
            validate_run_id_token(run_id)?;
            let bytes =
                fs_read(&path).map_err(|error| format!("failed to read run artifact: {error}"))?;
            runs.push(decode_persisted_artifact(&bytes, run_id)?);
        }
        Ok(runs)
    }
}

impl AnalysisArtifactStore for FilesystemAnalysisArtifactStore {
    fn persist_run(&self, run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String> {
        validate_run_identity(run)?;
        let path = self.run_path(&run.run_id)?;
        self.ensure_runs_directory()?;
        self.ensure_objects_directory()?;
        let prepared = prepare_run_artifact(run, Utc::now().to_rfc3339())?;
        for (digest, chunk) in &prepared.chunks {
            match runmat_filesystem::atomic_create_new(self.object_path(*digest), chunk)
                .map_err(|error| format!("failed to publish immutable field chunk: {error}"))?
            {
                AtomicCreateResult::Published { .. } => {}
                AtomicCreateResult::Existing(existing) if existing == *chunk => {}
                AtomicCreateResult::Existing(_) => {
                    return Err("analysis field chunk digest collision".to_string());
                }
            }
        }
        let bytes = serde_json::to_vec_pretty(&prepared.persisted)
            .map_err(|err| format!("failed to encode run artifact: {err}"))?;
        match runmat_filesystem::atomic_create_new(&path, &bytes)
            .map_err(|error| format!("failed to publish immutable run artifact: {error}"))?
        {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(existing_bytes) => {
                let PersistedAnalysisArtifact::Standard(existing) =
                    decode_persisted_artifact(&existing_bytes, &run.run_id)?
                else {
                    return Err("analysis run identity already contains another result kind".into());
                };
                if !same_persisted_content(&existing, &prepared.persisted) {
                    return Err("analysis run identity already contains different content".into());
                }
                return artifact_record(run, existing.created_at);
            }
        }
        prune_filesystem_runs(&self.root)?;

        artifact_record(run, prepared.persisted.created_at)
    }

    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        let path = self.run_path(run_id)?;
        if !owned_directory_exists(&self.root, "analysis artifact root")?
            || !owned_directory_exists(&self.root.join("runs"), "analysis artifact runs directory")?
        {
            return Ok(None);
        }
        match fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {}
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to inspect run artifact: {error}")),
        }
        let bytes = fs_read(&path).map_err(|err| format!("failed to read run artifact: {err}"))?;
        let PersistedAnalysisArtifact::Standard(persisted) =
            decode_persisted_artifact(&bytes, run_id)?
        else {
            return Ok(None);
        };
        Ok(Some(materialize_run(&persisted, |digest| {
            self.read_chunk(digest)
        })?))
    }

    fn load_run_metadata(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        let path = self.run_path(run_id)?;
        if !owned_directory_exists(&self.root, "analysis artifact root")?
            || !owned_directory_exists(&self.root.join("runs"), "analysis artifact runs directory")?
        {
            return Ok(None);
        }
        match fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {}
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to inspect run artifact: {error}")),
        }
        let bytes = fs_read(&path).map_err(|err| format!("failed to read run artifact: {err}"))?;
        let PersistedAnalysisArtifact::Standard(persisted) =
            decode_persisted_artifact(&bytes, run_id)?
        else {
            return Ok(None);
        };
        Ok(Some(persisted.metadata))
    }

    fn load_field_inventory(&self, run_id: &str) -> Result<Option<Vec<AnalysisField>>, String> {
        let path = self.run_path(run_id)?;
        if !owned_directory_exists(&self.root, "analysis artifact root")?
            || !owned_directory_exists(&self.root.join("runs"), "analysis artifact runs directory")?
        {
            return Ok(None);
        }
        let bytes = match fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {
                fs_read(&path).map_err(|error| format!("failed to read run artifact: {error}"))?
            }
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to inspect run artifact: {error}")),
        };
        let PersistedAnalysisArtifact::Standard(persisted) =
            decode_persisted_artifact(&bytes, run_id)?
        else {
            return Ok(None);
        };
        Ok(Some(persisted_field_inventory(&persisted)))
    }

    fn load_field_page(
        &self,
        run_id: &str,
        selection: &AnalysisFieldSelection,
        options: AnalysisFieldRequestOptions,
    ) -> Result<AnalysisFieldPageLookup, String> {
        let path = self.run_path(run_id)?;
        if !owned_directory_exists(&self.root, "analysis artifact root")?
            || !owned_directory_exists(&self.root.join("runs"), "analysis artifact runs directory")?
        {
            return Ok(AnalysisFieldPageLookup::RunNotFound);
        }
        let bytes = match fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {
                fs_read(&path).map_err(|error| format!("failed to read run artifact: {error}"))?
            }
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok(AnalysisFieldPageLookup::RunNotFound);
            }
            Err(error) => return Err(format!("failed to inspect run artifact: {error}")),
        };
        match decode_persisted_artifact(&bytes, run_id)? {
            PersistedAnalysisArtifact::Standard(persisted) => {
                field_page_from_artifact(&persisted, selection, options, |digest| {
                    self.read_chunk(digest)
                })
            }
            PersistedAnalysisArtifact::StructuralFrequency(artifact) => {
                frequency::field_page(&artifact, selection, options, |digest| {
                    self.read_chunk(digest)
                })
            }
        }
    }

    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String> {
        self.list_persisted_runs()?
            .into_iter()
            .filter_map(|artifact| match artifact {
                PersistedAnalysisArtifact::Standard(persisted) => Some(persisted),
                PersistedAnalysisArtifact::StructuralFrequency(_) => None,
            })
            .map(|persisted| materialize_run(&persisted, |digest| self.read_chunk(digest)))
            .collect()
    }

    fn list_run_metadata(&self) -> Result<Vec<AnalysisRunResult>, String> {
        Ok(self
            .list_persisted_runs()?
            .into_iter()
            .filter_map(|artifact| match artifact {
                PersistedAnalysisArtifact::Standard(persisted) => Some(persisted.metadata),
                PersistedAnalysisArtifact::StructuralFrequency(_) => None,
            })
            .collect())
    }

    fn persist_structural_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        object_catalog: &runmat_execution::value::PublishedObjectCatalog,
        artifact_roots: &crate::analysis::StructuralFrequencyArtifactRootIndex,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String> {
        let path = self.run_path(&step.run_id)?;
        self.ensure_runs_directory()?;
        self.ensure_objects_directory()?;
        let prepared = frequency::prepare_artifact(
            step,
            object_catalog,
            artifact_roots,
            Utc::now().to_rfc3339(),
        )?;
        let objects = frequency::ingest_objects(&prepared, source)?;
        for (digest, bytes) in objects {
            match runmat_filesystem::atomic_create_new(self.object_path(digest), &bytes)
                .map_err(|error| format!("failed to publish immutable analysis object: {error}"))?
            {
                AtomicCreateResult::Published { .. } => {}
                AtomicCreateResult::Existing(existing) if existing == bytes => {}
                AtomicCreateResult::Existing(_) => {
                    return Err("analysis object digest collision".into());
                }
            }
        }
        let bytes = serde_json::to_vec_pretty(&prepared)
            .map_err(|error| format!("failed to encode structural frequency artifact: {error}"))?;
        match runmat_filesystem::atomic_create_new(&path, &bytes)
            .map_err(|error| format!("failed to publish immutable run artifact: {error}"))?
        {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(existing_bytes) => {
                let PersistedAnalysisArtifact::StructuralFrequency(existing) =
                    decode_persisted_artifact(&existing_bytes, &step.run_id)?
                else {
                    return Err("analysis run identity already contains another result kind".into());
                };
                if !frequency::same_content(&existing, &prepared) {
                    return Err("analysis run identity already contains different content".into());
                }
                return Ok(frequency::artifact_record(&existing));
            }
        }
        prune_filesystem_runs(&self.root)?;
        Ok(frequency::artifact_record(&prepared))
    }

    fn load_structural_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String> {
        let path = self.run_path(run_id)?;
        if !owned_directory_exists(&self.root, "analysis artifact root")?
            || !owned_directory_exists(&self.root.join("runs"), "analysis artifact runs directory")?
        {
            return Ok(None);
        }
        let bytes = match fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {
                fs_read(&path).map_err(|error| {
                    format!("failed to read structural frequency artifact: {error}")
                })?
            }
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to inspect run artifact: {error}")),
        };
        let PersistedAnalysisArtifact::StructuralFrequency(artifact) =
            decode_persisted_artifact(&bytes, run_id)?
        else {
            return Ok(None);
        };
        Ok(Some(artifact.step))
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
        other => Err(format!(
            "unsupported analysis run artifact schema `{other}`"
        )),
    }
}

fn prepare_run_artifact(
    run: &AnalysisRunResult,
    created_at: String,
) -> Result<PreparedRunArtifact, String> {
    let mut fields = BTreeMap::<String, PersistedFieldArtifact>::new();
    let mut chunks = HashMap::new();
    let mut seen_field_ids = HashSet::new();
    let mut preparation_error = None;
    super::operations::visit_analysis_result_fields(run, |field| {
        if preparation_error.is_some() {
            return;
        }
        if let Err(error) = validate_field_id(&field.field_id) {
            preparation_error = Some(error);
            return;
        }
        if !seen_field_ids.insert(field.field_id.clone()) {
            preparation_error = Some(format!(
                "analysis field ID `{}` appears more than once in one run",
                field.field_id
            ));
            return;
        }
        let expected_count = match checked_element_count(&field.shape) {
            Ok(count) => count,
            Err(error) => {
                preparation_error = Some(error);
                return;
            }
        };
        match &field.values {
            AnalysisFieldValues::HostF64(values) => {
                if values.len() != expected_count {
                    preparation_error = Some(format!(
                        "analysis field `{}` shape requires {expected_count} values but contains {}",
                        field.field_id,
                        values.len()
                    ));
                    return;
                }
                if values.is_empty() {
                    return;
                }
                let payload = encode_field_values(values);
                let payload_digest = Digest::sha256(&payload);
                let logical_identity = field_logical_identity(field, payload_digest);
                let chunked = match chunk_numerical_payload(
                    AnalysisNumericalObjectKind::Field,
                    logical_identity,
                    &payload,
                    DEFAULT_ANALYSIS_CHUNK_BYTES,
                ) {
                    Ok(chunked) => chunked,
                    Err(error) => {
                        preparation_error = Some(format!(
                            "failed to chunk analysis field `{}`: {error}",
                            field.field_id
                        ));
                        return;
                    }
                };
                let artifact = PersistedFieldArtifact {
                    field_id: field.field_id.clone(),
                    shape: field.shape.clone(),
                    template: AnalysisField::host_f64(
                        &field.field_id,
                        field.shape.clone(),
                        Vec::new(),
                    ),
                    manifest: chunked.manifest.clone(),
                };
                for (descriptor, bytes) in chunked.manifest.chunks.iter().zip(chunked.chunks) {
                    match chunks.get(&descriptor.digest) {
                        Some(existing) if existing != &bytes => {
                            preparation_error =
                                Some("analysis field chunk digest collision".to_string());
                            return;
                        }
                        Some(_) => {}
                        None => {
                            chunks.insert(descriptor.digest, bytes);
                        }
                    }
                }
                fields.insert(field.field_id.clone(), artifact);
            }
            AnalysisFieldValues::DeviceRef(reference) => {
                if reference.element_count != expected_count {
                    preparation_error = Some(format!(
                        "device field `{}` shape requires {expected_count} values but its reference declares {}",
                        field.field_id, reference.element_count
                    ));
                } else {
                    preparation_error = Some(format!(
                        "device field `{}` is a process-local reference; materialize it through typed runtime value admission before persistence",
                        field.field_id
                    ));
                }
            }
        }
    });
    if let Some(error) = preparation_error {
        return Err(error);
    }

    let mut stripped_run = run.clone();
    super::operations::visit_analysis_result_fields_mut(&mut stripped_run, |field| {
        if let AnalysisFieldValues::HostF64(values) = &mut field.values {
            if !values.is_empty() {
                values.clear();
            }
        }
    });
    let field_inventory = super::operations::collect_analysis_result_fields(&stripped_run);
    let run_value = serde_json::to_value(&stripped_run)
        .map_err(|error| format!("failed to encode stripped analysis run: {error}"))?;
    let materialized_field_ids = fields.keys().cloned().collect::<Vec<_>>();
    let run_digest = persisted_run_digest(&run_value, &field_inventory, &materialized_field_ids)?;
    let mut metadata = stripped_run;
    metadata.run.observations.clear();
    metadata.run.series_reductions.clear();
    Ok(PreparedRunArtifact {
        persisted: PersistedRunArtifact {
            schema_version: ARTIFACT_SCHEMA_VERSION.to_string(),
            created_at,
            op_version: run_operation_version(run)?,
            run_digest,
            run: run_value,
            metadata,
            field_inventory,
            materialized_field_ids,
            fields: fields.into_values().collect(),
        },
        chunks,
    })
}

fn same_persisted_content(left: &PersistedRunArtifact, right: &PersistedRunArtifact) -> bool {
    left.schema_version == right.schema_version
        && left.op_version == right.op_version
        && left.run_digest == right.run_digest
        && left.run == right.run
        && left.metadata == right.metadata
        && left.field_inventory == right.field_inventory
        && left.materialized_field_ids == right.materialized_field_ids
        && left.fields == right.fields
}

fn materialize_run(
    persisted: &PersistedRunArtifact,
    mut read_chunk: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<AnalysisRunResult, String> {
    let mut values = HashMap::with_capacity(persisted.fields.len());
    for field in &persisted.fields {
        let payload = read_field_payload(field, &mut read_chunk)?;
        values.insert(
            field.field_id.clone(),
            (field.shape.clone(), decode_field_values(&payload)?),
        );
    }
    let mut run = persisted.run.clone();
    hydrate_field_values(&mut run, &values)?;
    let run = serde_json::from_value::<AnalysisRunResult>(run)
        .map_err(|error| format!("failed to decode hydrated analysis run: {error}"))?;
    validate_run_identity(&run)?;
    Ok(run)
}

fn hydrate_field_values(
    value: &mut serde_json::Value,
    fields: &HashMap<String, (Vec<usize>, Vec<f64>)>,
) -> Result<(), String> {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                hydrate_field_values(value, fields)?;
            }
        }
        serde_json::Value::Object(object) => {
            let field_id = object
                .get("field_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            if let Some(field_id) = field_id {
                if let Some((shape, values)) = fields.get(&field_id) {
                    let stored_shape = object
                        .get("shape")
                        .cloned()
                        .ok_or_else(|| format!("analysis field `{field_id}` omits its shape"))?;
                    if stored_shape
                        != serde_json::to_value(shape).map_err(|error| error.to_string())?
                    {
                        return Err(format!(
                            "analysis field `{field_id}` shape differs from its persisted manifest"
                        ));
                    }
                    object.insert(
                        "values".to_string(),
                        serde_json::to_value(AnalysisFieldValues::HostF64(values.clone()))
                            .map_err(|error| error.to_string())?,
                    );
                    return Ok(());
                }
            }
            for value in object.values_mut() {
                hydrate_field_values(value, fields)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn read_field_payload(
    field: &PersistedFieldArtifact,
    read_chunk: &mut impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<Vec<u8>, String> {
    let capacity = usize::try_from(field.manifest.encoded_length)
        .map_err(|_| "analysis field byte length is not representable".to_string())?;
    let mut payload = Vec::with_capacity(capacity);
    for descriptor in &field.manifest.chunks {
        let bytes = read_chunk(descriptor.digest)?.ok_or_else(|| {
            format!(
                "analysis field `{}` is missing chunk {}",
                field.field_id, descriptor.ordinal
            )
        })?;
        validate_chunk(descriptor, &bytes)?;
        payload.extend_from_slice(&bytes);
    }
    if payload.len() as u64 != field.manifest.encoded_length
        || Digest::sha256(&payload) != field.manifest.payload_digest
    {
        return Err(format!(
            "analysis field `{}` payload differs from its manifest",
            field.field_id
        ));
    }
    Ok(payload)
}

fn field_page_from_artifact(
    persisted: &PersistedRunArtifact,
    selection: &AnalysisFieldSelection,
    options: AnalysisFieldRequestOptions,
    mut read_chunk: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<AnalysisFieldPageLookup, String> {
    let field_id = &selection.field_id;
    validate_field_id(field_id)?;
    if selection.sample.is_some() {
        return Ok(AnalysisFieldPageLookup::SampleNotFound);
    }
    let template = match select_persisted_field(&persisted.fields, field_id) {
        Ok(Some(template)) => template,
        Ok(None) => return Ok(AnalysisFieldPageLookup::FieldNotFound),
        Err(available) => return Ok(AnalysisFieldPageLookup::FieldSelectionAmbiguous(available)),
    };
    let selected_field_id = template.field_id.clone();
    let mut descriptor = AnalysisFieldDescriptor::from_field(&template);
    if descriptor.storage == super::contracts::AnalysisFieldStorage::HostF64 {
        descriptor.storage_ref.kind = "chunked_field".to_string();
    }
    let total_count = checked_element_count(&template.shape)?;
    let offset = options.offset.unwrap_or(0).min(total_count);
    let limit = options
        .limit
        .unwrap_or(super::contracts::ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT)
        .min(super::contracts::ANALYSIS_FIELD_DEFAULT_PAGE_SIZE);
    let count = limit.min(total_count.saturating_sub(offset));
    if count == 0 {
        return Ok(AnalysisFieldPageLookup::Page(Box::new(
            AnalysisFieldPageResult {
                run_id: persisted.metadata.run_id.clone(),
                field_id: selected_field_id,
                descriptor,
                values: AnalysisFieldPageValues::Real(Vec::new()),
                offset,
                count: 0,
                total_count,
                truncated: offset > 0 || total_count > 0,
            },
        )));
    }
    let Some(stored) = persisted
        .fields
        .iter()
        .find(|field| field.field_id == selected_field_id)
    else {
        return Err(format!(
            "analysis field `{field_id}` is device-resident and has no materialized host artifact"
        ));
    };
    let start_byte = offset
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| "analysis field page byte offset overflow".to_string())?;
    let end_byte = offset
        .checked_add(count)
        .and_then(|end| end.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| "analysis field page byte end overflow".to_string())?;
    let mut page = Vec::with_capacity(count * std::mem::size_of::<f64>());
    let mut chunk_start = 0_usize;
    for chunk in &stored.manifest.chunks {
        let chunk_len = usize::try_from(chunk.encoded_length)
            .map_err(|_| "analysis field chunk length is not representable".to_string())?;
        let chunk_end = chunk_start
            .checked_add(chunk_len)
            .ok_or_else(|| "analysis field chunk range overflow".to_string())?;
        let overlap_start = start_byte.max(chunk_start);
        let overlap_end = end_byte.min(chunk_end);
        if overlap_start < overlap_end {
            let bytes = read_chunk(chunk.digest)?.ok_or_else(|| {
                format!(
                    "analysis field `{field_id}` is missing chunk {}",
                    chunk.ordinal
                )
            })?;
            validate_chunk(chunk, &bytes)?;
            page.extend_from_slice(&bytes[overlap_start - chunk_start..overlap_end - chunk_start]);
        }
        chunk_start = chunk_end;
        if chunk_start >= end_byte {
            break;
        }
    }
    let values = decode_field_values(&page)?;
    if values.len() != count {
        return Err(format!(
            "analysis field `{field_id}` page is incomplete: expected {count} values, found {}",
            values.len()
        ));
    }
    Ok(AnalysisFieldPageLookup::Page(Box::new(
        AnalysisFieldPageResult {
            run_id: persisted.metadata.run_id.clone(),
            field_id: selected_field_id,
            descriptor,
            values: AnalysisFieldPageValues::Real(values),
            offset,
            count,
            total_count,
            truncated: offset > 0 || offset.saturating_add(count) < total_count,
        },
    )))
}

fn select_persisted_field(
    fields: &[PersistedFieldArtifact],
    requested: &str,
) -> Result<Option<AnalysisField>, Vec<String>> {
    if let Some(field) = fields.iter().find(|field| field.field_id == requested) {
        return Ok(Some(field.template.clone()));
    }
    let mut matches = fields
        .iter()
        .filter(|field| super::field_selection::field_id_matches(&field.field_id, requested));
    let Some(first) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_none() {
        return Ok(Some(first.template.clone()));
    }
    Err(fields
        .iter()
        .filter(|field| super::field_selection::field_id_matches(&field.field_id, requested))
        .map(|field| field.field_id.clone())
        .collect())
}

fn persisted_field_inventory(persisted: &PersistedRunArtifact) -> Vec<AnalysisField> {
    persisted.field_inventory.clone()
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
    if persisted_run_digest(
        &persisted.run,
        &persisted.field_inventory,
        &persisted.materialized_field_ids,
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
    validate_run_identity(&persisted.metadata)?;
    let mut expected_fields = BTreeMap::new();
    collect_persisted_field_shells(&persisted.run, &mut expected_fields)?;
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
    if inventory_fields != expected_fields {
        return Err("analysis run field inventory differs from its metadata shells".into());
    }
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
                &AnalysisField::host_f64(&field.field_id, field.shape.clone(), Vec::new()),
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
    let stored_field_ids = stored_fields.keys().cloned().collect::<Vec<_>>();
    if persisted.materialized_field_ids != stored_field_ids
        || stored_fields
            .iter()
            .any(|(field_id, shape)| expected_fields.get(field_id) != Some(shape))
    {
        return Err(
            "analysis run field inventory differs from its persisted manifests".to_string(),
        );
    }
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

fn persisted_run_digest(
    value: &serde_json::Value,
    field_inventory: &[AnalysisField],
    materialized_field_ids: &[String],
) -> Result<Digest, String> {
    serde_json::to_vec(&(value, field_inventory, materialized_field_ids))
        .map(Digest::sha256)
        .map_err(|error| format!("failed to encode analysis run metadata: {error}"))
}

pub(super) fn ensure_owned_directory(path: &Path, label: &str) -> Result<(), String> {
    match fs_symlink_metadata(path) {
        Ok(metadata) => return validate_owned_directory_metadata(metadata, path, label),
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(format!("failed to inspect {label}: {error}")),
    }
    fs_create_dir_all(path).map_err(|error| format!("failed to create {label}: {error}"))?;
    let metadata = fs_symlink_metadata(path)
        .map_err(|error| format!("failed to inspect created {label}: {error}"))?;
    validate_owned_directory_metadata(metadata, path, label)
}

pub(super) fn owned_directory_exists(path: &Path, label: &str) -> Result<bool, String> {
    match fs_symlink_metadata(path) {
        Ok(metadata) => {
            validate_owned_directory_metadata(metadata, path, label)?;
            Ok(true)
        }
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!("failed to inspect {label}: {error}")),
    }
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
    let (expected_publishable, expected_publication_reasons) =
        super::result_policy::evaluate_production_publication(&run.run, run.numerically_acceptable);
    if run.production_publishable != expected_publishable {
        return Err(
            "production publication must require both numerical acceptance and a production-qualified formulation"
                .to_string(),
        );
    }
    if run.production_publication_reasons != expected_publication_reasons {
        return Err(
            "production publication reasons do not match the failed publication gates".to_string(),
        );
    }
    let expected_status =
        super::result_policy::classify_run_status(run.result_quality, run.numerically_acceptable);
    if run.run_status != expected_status {
        return Err(format!(
            "analysis run status {:?} does not match numerical outcome {:?}",
            run.run_status, expected_status
        ));
    }
    if !run.provenance.fallback_events.is_empty() && !identity.permits_fallback() {
        return Err(
            "analysis run identity forbids the execution fallbacks recorded by provenance"
                .to_string(),
        );
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

fn prune_filesystem_runs(root: &Path) -> Result<(), String> {
    let retention = current_retention_config();
    let max_runs = retention.max_runs.unwrap_or_else(|| {
        std::env::var("RUNMAT_FEA_ARTIFACT_MAX_RUNS")
            .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS"))
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
    });
    let max_runs_per_kind = retention.max_runs_per_kind.unwrap_or_else(|| {
        std::env::var("RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND")
            .or_else(|_| std::env::var("RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS_PER_KIND"))
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
    });
    if max_runs == 0 && max_runs_per_kind == 0 {
        return Ok(());
    }

    let runs_dir = root.join("runs");
    ensure_owned_directory(root, "analysis artifact root")?;
    ensure_owned_directory(&runs_dir, "analysis artifact runs directory")?;
    let mut artifacts = Vec::new();
    for entry in fs_read_dir(&runs_dir).map_err(|err| format!("failed to scan artifacts: {err}"))? {
        let path = entry.path().to_path_buf();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        if entry.file_type() != FsFileType::File {
            return Err(format!(
                "analysis run artifact must be a regular file: {}",
                path.display()
            ));
        }
        let run_id = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| {
                format!(
                    "analysis run artifact filename is invalid: {}",
                    path.display()
                )
            })?;
        let bytes = fs_read(&path).map_err(|err| format!("failed to read artifact file: {err}"))?;
        let Ok(persisted) = decode_persisted_artifact(&bytes, run_id) else {
            continue;
        };
        let identity = match persisted {
            PersistedAnalysisArtifact::Standard(persisted) => persisted.metadata.run.identity,
            PersistedAnalysisArtifact::StructuralFrequency(persisted) => persisted.step.identity,
        };
        let modified = fs_modified(&path).ok().flatten();
        artifacts.push((path, identity, modified));
    }
    artifacts.sort_by(|a, b| b.2.cmp(&a.2));

    let mut to_remove = Vec::new();
    if max_runs_per_kind > 0 {
        let mut per_kind_counts: HashMap<AnalysisRunIdentity, usize> = HashMap::new();
        for (path, identity, _modified) in &artifacts {
            let count = per_kind_counts.entry(identity.clone()).or_default();
            *count += 1;
            if *count > max_runs_per_kind {
                to_remove.push(path.clone());
            }
        }
    }
    if max_runs > 0 {
        for (index, (path, _identity, _modified)) in artifacts.iter().enumerate() {
            if index >= max_runs {
                to_remove.push(path.clone());
            }
        }
    }
    to_remove.sort();
    to_remove.dedup();
    for path in to_remove {
        let _ = fs_remove_file(path);
    }
    Ok(())
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

fn current_retention_config() -> AnalysisArtifactRetentionConfig {
    retention_config()
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_run_id() -> String {
    let seq = NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed);
    format!("run_{}_{}", Utc::now().timestamp_millis(), seq)
}

pub fn persist_run_result(run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.persist_run(run)
}

pub(in crate::analysis) fn persist_structural_frequency_result(
    step: &AnalysisStudyStepRunData,
    persistence: &crate::analysis::frequency_result::StructuralFrequencyPersistence,
) -> Result<AnalysisArtifactRecord, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.persist_structural_frequency(
        step,
        &persistence.object_catalog,
        &persistence.artifact_roots,
        persistence.source.as_ref(),
    )
}

pub fn load_run_result(run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.load_run(run_id)
}

pub fn load_structural_frequency_result(
    run_id: &str,
) -> Result<Option<AnalysisStudyStepRunData>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.load_structural_frequency(run_id)
}

pub fn load_run_metadata(run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.load_run_metadata(run_id)
}

pub fn load_field_inventory(run_id: &str) -> Result<Option<Vec<AnalysisField>>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.load_field_inventory(run_id)
}

pub fn load_field_page(
    run_id: &str,
    selection: &AnalysisFieldSelection,
    options: AnalysisFieldRequestOptions,
) -> Result<AnalysisFieldPageLookup, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.load_field_page(run_id, selection, options)
}

pub fn list_run_results() -> Result<Vec<AnalysisRunResult>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.list_runs()
}

pub fn list_run_metadata() -> Result<Vec<AnalysisRunResult>, String> {
    let guard = global_store()
        .read()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    guard.list_run_metadata()
}

pub fn configure_artifact_store(config: AnalysisArtifactStoreConfig) -> Result<(), String> {
    let mut guard = global_store()
        .write()
        .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
    *guard = store_from_config(config);
    Ok(())
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
    PathBuf::from("artifacts")
}

fn fs_create_dir_all(path: impl Into<PathBuf>) -> std::io::Result<()> {
    runmat_filesystem::create_dir_all(path.into())
}

fn fs_read(path: impl Into<PathBuf>) -> std::io::Result<Vec<u8>> {
    runmat_filesystem::read(path.into())
}

fn fs_remove_file(path: impl Into<PathBuf>) -> std::io::Result<()> {
    match runmat_filesystem::remove_file(path.into()) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn fs_read_dir(path: impl Into<PathBuf>) -> std::io::Result<Vec<DirEntry>> {
    runmat_filesystem::read_dir(path.into())
}

fn fs_symlink_metadata(path: impl Into<PathBuf>) -> std::io::Result<FsMetadata> {
    runmat_filesystem::symlink_metadata(path.into())
}

fn fs_modified(path: impl Into<PathBuf>) -> std::io::Result<Option<std::time::SystemTime>> {
    runmat_filesystem::metadata(path.into()).map(|metadata| metadata.modified())
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
