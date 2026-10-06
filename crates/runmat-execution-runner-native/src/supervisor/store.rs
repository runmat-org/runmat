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
    BatchSubmission, DriverCompletion, DriverMarker, JobAttachment, LocalJobRecord, LocalJobState,
    ProgramBatchSubmission,
};
use crate::{NativeExecutionError, NativeExecutionResult};

const ATTACH_CHUNK_BYTES: usize = 1024 * 1024;

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
    pub(super) schema_version: u16,
    pub(super) handle: JobHandle,
    pub(super) request_digest: Digest,
    pub(super) workload: W,
    idempotency_key_digest: Option<Digest>,
    pub(super) retention_millis: u64,
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
}

impl<P: Borrow<ProgramBatchSubmission>> PersistedWorkload<P> {
    fn digest(&self) -> Digest {
        match self {
            Self::Script { source_digest, .. } => *source_digest,
            Self::Program { submission } => submission.borrow().artifact.id.0,
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
            request_digest,
            idempotency_key_digest,
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
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        submission.validate()?;
        let request_digest = submission.request_digest()?;
        let idempotency_key_digest = submission.idempotency_digest();
        self.create_record(
            request_digest,
            idempotency_key_digest,
            submission.retention_millis,
            PersistedWorkload::Program { submission },
            now,
            None,
        )
    }

    fn create_record<P: Borrow<ProgramBatchSubmission> + Serialize>(
        &self,
        request_digest: Digest,
        idempotency_key_digest: Option<Digest>,
        retention_millis: u64,
        workload: PersistedWorkload<P>,
        now: u64,
        frozen_source: Option<(&str, &[u8])>,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
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
        };
        let handle = JobHandle {
            id: job_id,
            run_id,
            generation: 1,
            outputs: OutputContract { requested_outputs },
        };
        let job_dir = self.job_dir(job_id);
        secure_directory(&job_dir)?;
        if let Some((source_name, source)) = frozen_source {
            atomic_write(&job_dir.join(source_name), source)?;
        }
        let request = PersistedRequest {
            schema_version: 1,
            handle: handle.clone(),
            request_digest,
            workload: &workload,
            idempotency_key_digest,
            retention_millis,
        };
        atomic_json(&job_dir.join("request.json"), &request)?;
        let record = LocalJobRecord {
            schema_version: 1,
            handle,
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
        self.write_record(&record)?;
        Ok((record, true))
    }

    pub(super) fn record(&self, job_id: JobId) -> NativeExecutionResult<LocalJobRecord> {
        let record: LocalJobRecord = read_json(&self.job_dir(job_id).join("record.json"))?;
        if record.handle.id != job_id {
            return Err(NativeExecutionError::Protocol(
                "durable job record is stored under the wrong identity".into(),
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

    pub(super) fn write_record(&self, record: &LocalJobRecord) -> NativeExecutionResult<()> {
        record.validate()?;
        atomic_json(&self.job_dir(record.handle.id).join("record.json"), record)
    }

    pub(super) fn completion(
        &self,
        job_id: JobId,
    ) -> NativeExecutionResult<Option<DriverCompletion>> {
        if self.program_submission(job_id)?.is_some() {
            return Ok(None);
        }
        let path = self.job_dir(job_id).join("completion.json");
        let Some(completion) = read_optional_json::<DriverCompletion>(&path)? else {
            return Ok(None);
        };
        completion.validate()?;
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
            response: None,
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
                let record = LocalJobRecord {
                    schema_version: 1,
                    handle: request.handle,
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
                self.write_record(&record)?;
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
