use std::borrow::Borrow;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use runmat_execution::{Digest, JobHandle, JobId, OutputContract, RunId};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::driver_files::marker_path;
use super::filesystem::{
    atomic_json, atomic_write, io_error, read_json, read_optional_json, secure_directory,
    unix_millis,
};
use super::model::{
    BatchSubmission, DriverCompletion, DriverMarker, GraphBatchSubmission, JobAttachment,
    LocalJobRecord, LocalJobState, LocalJobWorkloadKind, ProgramBatchSubmission,
    MAX_RETENTION_MILLIS, MIN_RETENTION_MILLIS,
};
use crate::{NativeExecutionError, NativeExecutionResult};

#[cfg(test)]
#[path = "store/retention_tests.rs"]
mod retention_tests;

const ATTACH_CHUNK_BYTES: usize = 1024 * 1024;

struct LocalJobRequestIdentity {
    request_digest: Digest,
    idempotency_key_digest: Option<Digest>,
    local_fea_organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
}

impl LocalJobRequestIdentity {
    fn new(
        request_digest: Digest,
        idempotency_key_digest: Option<Digest>,
        local_fea_organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
    ) -> Self {
        Self {
            request_digest: bind_organization(request_digest, local_fea_organization),
            idempotency_key_digest,
            local_fea_organization,
        }
    }
}

pub(super) fn bind_organization(
    request: Digest,
    organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
) -> Digest {
    use runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization;
    let selector = match organization {
        FeaLocalOrganization::Default => return request,
        FeaLocalOrganization::Selected(org) => [b"selected\0".as_slice(), org.as_bytes()].concat(),
        FeaLocalOrganization::Invalid => b"invalid".to_vec(),
    };
    Digest::sha256(
        [
            b"runmat-local-job-account-selector-v1\0".as_slice(),
            request.bytes(),
            &selector,
        ]
        .concat(),
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SupervisorPaths {
    pub root: PathBuf,
    pub jobs: PathBuf,
    pub token: PathBuf,
    pub lock: PathBuf,
    #[cfg(unix)]
    pub socket: PathBuf,
    #[cfg(windows)]
    pub pipe: String,
}

impl SupervisorPaths {
    pub fn new(root: PathBuf) -> NativeExecutionResult<Self> {
        if !root.is_absolute() {
            return Err(NativeExecutionError::Configuration(
                "local supervisor state root must be absolute".into(),
            ));
        }
        #[cfg(windows)]
        let pipe = {
            let digest = Digest::sha256(root.to_string_lossy().as_bytes()).to_string();
            format!(r"\\.\pipe\runmat-execution-{digest}")
        };
        Ok(Self {
            jobs: root.join("jobs"),
            token: root.join("auth-token"),
            lock: root.join("supervisor.lock"),
            #[cfg(unix)]
            socket: root.join("supervisor.sock"),
            #[cfg(windows)]
            pipe,
            root,
        })
    }

    pub fn platform_default() -> NativeExecutionResult<Self> {
        if let Some(root) = std::env::var_os("RUNMAT_EXECUTION_STATE_DIR") {
            return Self::new(PathBuf::from(root));
        }
        let root = dirs::data_local_dir()
            .ok_or_else(|| {
                NativeExecutionError::Configuration(
                    "the per-user local data directory is unavailable".into(),
                )
            })?
            .join("runmat")
            .join("execution");
        Self::new(root)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PersistedRequest<W = PersistedWorkload> {
    #[serde(default)]
    pub(super) local_fea_organization:
        runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
    pub(super) schema_version: u16,
    pub(super) handle: JobHandle,
    pub(super) request_digest: Digest,
    pub(super) workload: W,
    idempotency_key_digest: Option<Digest>,
    pub(super) retention_millis: u64,
}

impl PersistedRequest {
    /// The stored digest field is not itself proof of the workload tag: Show
    /// and InspectKind must bind that tag to the exact persisted request before
    /// choosing the Direct or Graph result decoder.
    fn validate_content_digest(&self, job_dir: &Path) -> NativeExecutionResult<()> {
        let digest = match &self.workload {
            PersistedWorkload::Script {
                source_digest,
                source_name,
                arguments,
                working_directory,
            } => {
                if Path::new(source_name).components().count() != 1
                    || !matches!(
                        Path::new(source_name).components().next(),
                        Some(std::path::Component::Normal(_))
                    )
                {
                    return Err(NativeExecutionError::Protocol(
                        "durable script source name is not local to its job".into(),
                    ));
                }
                let source = fs::read(job_dir.join(source_name)).map_err(io_error)?;
                if Digest::sha256(&source) != *source_digest {
                    return Err(NativeExecutionError::Protocol(
                        "durable script source differs from its frozen request".into(),
                    ));
                }
                BatchSubmission {
                    source_name: source_name.clone(),
                    source,
                    arguments: arguments.clone(),
                    working_directory: working_directory.clone(),
                    idempotency_key: None,
                    retention_millis: self.retention_millis,
                }
                .request_digest()?
            }
            PersistedWorkload::Program { submission } => submission.request_digest()?,
            PersistedWorkload::Graph { submission } => submission.request_digest()?,
        };
        if bind_organization(digest, self.local_fea_organization) != self.request_digest {
            return Err(NativeExecutionError::Protocol(
                "durable job workload differs from its request digest".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub(super) enum PersistedWorkload<P = Box<ProgramBatchSubmission>> {
    Script {
        source_digest: Digest,
        source_name: String,
        arguments: Vec<String>,
        working_directory: String,
    },
    Program {
        submission: P,
    },
    Graph {
        submission: Box<GraphBatchSubmission>,
    },
}

impl<P: Borrow<ProgramBatchSubmission>> PersistedWorkload<P> {
    fn kind(&self) -> LocalJobWorkloadKind {
        match self {
            Self::Script { .. } => LocalJobWorkloadKind::Script,
            Self::Program { .. } => LocalJobWorkloadKind::Program,
            Self::Graph { .. } => LocalJobWorkloadKind::Graph,
        }
    }

    fn digest(&self) -> Digest {
        match self {
            Self::Script { source_digest, .. } => *source_digest,
            Self::Program { submission } => submission.borrow().artifact.id.0,
            Self::Graph { submission } => submission.request.artifact.id.0,
        }
    }
}

pub(super) struct JobStore {
    paths: SupervisorPaths,
}

impl JobStore {
    pub(super) fn open(paths: SupervisorPaths) -> NativeExecutionResult<Self> {
        secure_directory(&paths.root)?;
        secure_directory(&paths.jobs)?;
        Ok(Self { paths })
    }

    pub(super) fn create_script(
        &self,
        submission: BatchSubmission,
        now: u64,
        organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        submission.validate()?;
        let request_digest = submission.request_digest()?;
        let idempotency_key_digest = submission.idempotency_digest();
        let workload = PersistedWorkload::<Box<ProgramBatchSubmission>>::Script {
            source_digest: Digest::sha256(&submission.source),
            source_name: submission.source_name.clone(),
            arguments: submission.arguments.clone(),
            working_directory: submission.working_directory.clone(),
        };
        self.create_record(
            LocalJobRequestIdentity::new(request_digest, idempotency_key_digest, organization),
            submission.retention_millis,
            workload,
            now,
            Some((&submission.source_name, &submission.source)),
        )
    }

    pub(super) fn create_program(
        &self,
        submission: &ProgramBatchSubmission,
        now: u64,
        organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        submission.validate()?;
        let request_digest = submission.request_digest()?;
        let idempotency_key_digest = submission.idempotency_digest();
        self.create_record(
            LocalJobRequestIdentity::new(request_digest, idempotency_key_digest, organization),
            submission.retention_millis,
            PersistedWorkload::Program { submission },
            now,
            None,
        )
    }

    #[cfg(all(test, unix))]
    pub(super) fn create_graph(
        &self,
        submission: GraphBatchSubmission,
        now: u64,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        self.create_graph_validated(submission.into_validated()?, now, Default::default())
    }

    pub(super) fn create_graph_validated(
        &self,
        validated: super::model::ValidatedGraphBatchSubmission,
        now: u64,
        organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        let super::model::ValidatedGraphBatchSubmission {
            submission,
            request_digest,
            idempotency_key_digest,
        } = validated;
        self.create_record(
            LocalJobRequestIdentity::new(request_digest, idempotency_key_digest, organization),
            submission.retention_millis,
            PersistedWorkload::<Box<ProgramBatchSubmission>>::Graph {
                submission: Box::new(submission),
            },
            now,
            None,
        )
    }

    fn create_record<P: Borrow<ProgramBatchSubmission> + Serialize>(
        &self,
        identity: LocalJobRequestIdentity,
        retention_millis: u64,
        workload: PersistedWorkload<P>,
        now: u64,
        frozen_source: Option<(&str, &[u8])>,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        let LocalJobRequestIdentity {
            request_digest,
            idempotency_key_digest,
            local_fea_organization,
        } = identity;
        if let Some(key_digest) = idempotency_key_digest {
            if let Some(record) = self
                .list()?
                .into_iter()
                .find(|record| record.idempotency_key_digest == Some(key_digest))
            {
                if record.request_digest != request_digest {
                    return Err(NativeExecutionError::Protocol(
                        "idempotency key was already used for a different batch request".into(),
                    ));
                }
                return Ok((record, false));
            }
        }
        let nonce = Uuid::new_v4();
        let job_id = JobId::derive(&[request_digest.bytes(), nonce.as_bytes()]);
        let run_id = RunId::derive(&[job_id.bytes(), b"local-batch"]);
        let requested_outputs = match &workload {
            PersistedWorkload::Script { .. } => 0,
            PersistedWorkload::Program { submission } => submission.borrow().requested_outputs,
            PersistedWorkload::Graph { submission } => submission.request.requested_outputs,
        };
        let handle = JobHandle {
            id: job_id,
            run_id,
            generation: 1,
            outputs: OutputContract {
                requested_outputs,
                owned_work: None,
            },
        };
        let job_dir = self.job_dir(job_id);
        secure_directory(&job_dir)?;
        if let Some((source_name, source)) = frozen_source {
            atomic_write(&job_dir.join(source_name), source)?;
        }
        let request = PersistedRequest {
            local_fea_organization,
            schema_version: 1,
            handle: handle.clone(),
            request_digest,
            workload: &workload,
            idempotency_key_digest,
            retention_millis,
        };
        atomic_json(&job_dir.join("request.json"), &request)?;
        let mut record = LocalJobRecord {
            local_fea_organization,
            schema_version: 1,
            handle,
            workload_kind: workload.kind(),
            state: LocalJobState::Queued,
            request_digest,
            workload_digest: workload.digest(),
            idempotency_key_digest,
            submitted_unix_millis: now,
            updated_unix_millis: now,
            retain_until_unix_millis: now.saturating_add(retention_millis),
            driver_process_id: None,
            exit_code: None,
            message: None,
        };
        self.write_record(&mut record)?;
        Ok((record, true))
    }

    pub(super) fn record(&self, job_id: JobId) -> NativeExecutionResult<LocalJobRecord> {
        let mut record: LocalJobRecord = read_json(&self.job_dir(job_id).join("record.json"))?;
        if record.handle.id != job_id {
            return Err(NativeExecutionError::Protocol(
                "durable job record is stored under the wrong identity".into(),
            ));
        }
        let request: PersistedRequest = read_json(&self.job_dir(job_id).join("request.json"))?;
        if request.schema_version != 1
            || request.handle != record.handle
            || request.request_digest != record.request_digest
            || request.local_fea_organization != record.local_fea_organization
        {
            return Err(NativeExecutionError::Protocol(
                "durable job workload differs from its stored identity".into(),
            ));
        }
        request.validate_content_digest(&self.job_dir(job_id))?;
        let selected_kind = request.workload.kind();
        if record.workload_kind == LocalJobWorkloadKind::Legacy {
            record.workload_kind = selected_kind;
        } else if record.workload_kind != selected_kind {
            return Err(NativeExecutionError::Protocol(
                "durable job workload kind differs from its persisted request".into(),
            ));
        }
        record.validate()?;
        Ok(record)
    }

    pub(super) fn list(&self) -> NativeExecutionResult<Vec<LocalJobRecord>> {
        let mut records = Vec::new();
        for entry in fs::read_dir(&self.paths.jobs).map_err(io_error)? {
            let entry = entry.map_err(io_error)?;
            if !entry.file_type().map_err(io_error)?.is_dir() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let Ok(job_id) = name.parse::<JobId>() else {
                continue;
            };
            records.push(self.recover_record(job_id)?);
        }
        records.sort_by_key(|record| (record.submitted_unix_millis, record.handle.id));
        Ok(records)
    }

    pub(super) fn write_record(&self, record: &mut LocalJobRecord) -> NativeExecutionResult<()> {
        if record.state.is_terminal() {
            // Read the original policy without constructing or cloning its program
            // payload. Retention starts at terminal persistence, not submission;
            // merely inspecting an unchanged terminal record does not renew it.
            let request: PersistedRequest<serde::de::IgnoredAny> =
                read_json(&self.job_dir(record.handle.id).join("request.json"))?;
            if request.schema_version != 1
                || request.handle != record.handle
                || request.request_digest != record.request_digest
                || request.local_fea_organization != record.local_fea_organization
                || !(MIN_RETENTION_MILLIS..=MAX_RETENTION_MILLIS)
                    .contains(&request.retention_millis)
            {
                return Err(NativeExecutionError::Protocol(
                    "terminal retention differs from the original job policy".into(),
                ));
            }
            let terminal_expiry = record
                .updated_unix_millis
                .checked_add(request.retention_millis)
                .ok_or_else(|| {
                    NativeExecutionError::Protocol("terminal retention deadline overflows".into())
                })?;
            record.retain_until_unix_millis = record.retain_until_unix_millis.max(terminal_expiry);
        }
        record.validate()?;
        atomic_json(&self.job_dir(record.handle.id).join("record.json"), record)
    }

    pub(super) fn completion(
        &self,
        job_id: JobId,
    ) -> NativeExecutionResult<Option<DriverCompletion>> {
        if self.retained_submission(job_id)?.is_some() {
            return Ok(None);
        }
        let path = self.job_dir(job_id).join("completion.json");
        let Some(completion) = read_optional_json::<DriverCompletion>(&path)? else {
            return Ok(None);
        };
        completion.validate()?;
        let request: PersistedRequest = read_json(&self.job_dir(job_id).join("request.json"))?;
        match (&request.workload, &completion.response) {
            (PersistedWorkload::Script { .. }, None) => {}
            // A detached Graph driver that fails before execution commits no response.
            (PersistedWorkload::Graph { .. }, None) if !completion.success => {}
            (PersistedWorkload::Graph { submission }, Some(response)) => submission
                .request
                .validate_response(response)
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?,
            _ => {
                return Err(NativeExecutionError::Protocol(
                    "durable completion differs from its persisted workload kind".into(),
                ));
            }
        }
        Ok(Some(completion))
    }

    pub(super) fn driver_marker(
        &self,
        job_id: JobId,
    ) -> NativeExecutionResult<Option<DriverMarker>> {
        let attempt_id = self
            .program_assignment(job_id)?
            .map(|assignment| assignment.request.id);
        let path = marker_path(&self.job_dir(job_id), attempt_id);
        let Some(marker) = read_optional_json::<DriverMarker>(&path)? else {
            return Ok(None);
        };
        if marker.schema_version != if marker.attempt_id.is_some() { 3 } else { 2 }
            || marker.process_id == 0
            || marker.attempt_id != attempt_id
        {
            return Err(NativeExecutionError::Protocol(
                "durable driver marker is malformed".into(),
            ));
        }
        Ok(Some(marker))
    }

    pub(super) fn attachment(
        &self,
        job_id: JobId,
        stdout_offset: u64,
        stderr_offset: u64,
    ) -> NativeExecutionResult<JobAttachment> {
        let job_dir = self.job_dir(job_id);
        let (stdout, next_stdout_offset) = read_chunk(&job_dir.join("stdout.log"), stdout_offset)?;
        let (stderr, next_stderr_offset) = read_chunk(&job_dir.join("stderr.log"), stderr_offset)?;
        Ok(JobAttachment {
            record: self.record(job_id)?,
            stdout,
            stderr,
            next_stdout_offset,
            next_stderr_offset,
            // Graph results are committed with their driver completion; portable
            // program responses are attached from their provider report.
            response: self
                .completion(job_id)?
                .and_then(|completion| completion.response),
            resources: None,
            execution: None,
            latest_progress: None,
        })
    }

    pub(super) fn stdout_path(&self, job_id: JobId) -> PathBuf {
        self.job_dir(job_id).join("stdout.log")
    }

    pub(super) fn stderr_path(&self, job_id: JobId) -> PathBuf {
        self.job_dir(job_id).join("stderr.log")
    }

    pub(super) fn job_dir(&self, job_id: JobId) -> PathBuf {
        self.paths.jobs.join(job_id.to_string())
    }

    pub(super) fn gc(
        &self,
        now: u64,
        mut can_collect: impl FnMut(JobId) -> NativeExecutionResult<bool>,
    ) -> NativeExecutionResult<Vec<JobId>> {
        let mut collected = Vec::new();
        for record in self
            .list()?
            .into_iter()
            .filter(|record| record.state.is_terminal() && record.retain_until_unix_millis <= now)
        {
            if can_collect(record.handle.id)? {
                fs::remove_dir_all(self.job_dir(record.handle.id)).map_err(io_error)?;
                collected.push(record.handle.id);
            }
        }
        Ok(collected)
    }

    fn recover_record(&self, job_id: JobId) -> NativeExecutionResult<LocalJobRecord> {
        match self.record(job_id) {
            Ok(record) => Ok(record),
            Err(record_error) => {
                let request: PersistedRequest =
                    match read_json(&self.job_dir(job_id).join("request.json")) {
                        Ok(request) => request,
                        Err(_) => return Err(record_error),
                    };
                if request.schema_version != 1 || request.handle.id != job_id {
                    return Err(record_error);
                }
                let now = unix_millis();
                let corrupt = self
                    .job_dir(job_id)
                    .join(format!("record.corrupt-{}.json", Uuid::new_v4().simple()));
                let _ = fs::rename(self.job_dir(job_id).join("record.json"), corrupt);
                let mut record = LocalJobRecord {
                    local_fea_organization: request.local_fea_organization,
                    schema_version: 1,
                    handle: request.handle,
                    workload_kind: request.workload.kind(),
                    state: LocalJobState::Indeterminate,
                    request_digest: request.request_digest,
                    workload_digest: request.workload.digest(),
                    idempotency_key_digest: request.idempotency_key_digest,
                    submitted_unix_millis: now,
                    updated_unix_millis: now,
                    retain_until_unix_millis: now.saturating_add(request.retention_millis),
                    driver_process_id: None,
                    exit_code: None,
                    message: Some(
                        "job metadata was corrupt; execution state is indeterminate".into(),
                    ),
                };
                self.write_record(&mut record)?;
                Ok(record)
            }
        }
    }
}

fn read_chunk(path: &Path, offset: u64) -> NativeExecutionResult<(Vec<u8>, u64)> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok((Vec::new(), offset))
        }
        Err(error) => return Err(io_error(error)),
    };
    let length = file.metadata().map_err(io_error)?.len();
    let start = offset.min(length);
    file.seek(SeekFrom::Start(start)).map_err(io_error)?;
    let mut bytes = Vec::new();
    file.take(ATTACH_CHUNK_BYTES as u64)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    let next_offset = start.saturating_add(bytes.len() as u64);
    Ok((bytes, next_offset))
}
