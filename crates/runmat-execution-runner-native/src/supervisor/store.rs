use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use runmat_execution::{Digest, JobHandle, JobId, OutputContract, RunId};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::filesystem::{
    atomic_json, atomic_write, io_error, read_json, secure_directory, unix_millis,
};
use super::model::{
    BatchDriverInvocation, BatchSubmission, DriverCompletion, DriverMarker, JobAttachment,
    LocalJobRecord, LocalJobState, ProgramBatchSubmission,
};
use crate::{NativeExecutionError, NativeExecutionResult};

const ATTACH_CHUNK_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug)]
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
struct PersistedRequest {
    schema_version: u16,
    handle: JobHandle,
    request_digest: Digest,
    workload: PersistedWorkload,
    idempotency_key_digest: Option<Digest>,
    retention_millis: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
enum PersistedWorkload {
    Script {
        source_digest: Digest,
        source_name: String,
        arguments: Vec<String>,
        working_directory: String,
    },
    Program {
        submission: Box<ProgramBatchSubmission>,
    },
}

impl PersistedWorkload {
    fn digest(&self) -> Digest {
        match self {
            Self::Script { source_digest, .. } => *source_digest,
            Self::Program { submission } => submission.artifact.id.0,
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
        let workload = PersistedWorkload::Script {
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
        submission: ProgramBatchSubmission,
        now: u64,
    ) -> NativeExecutionResult<(LocalJobRecord, bool)> {
        submission.validate()?;
        let request_digest = submission.request_digest()?;
        let idempotency_key_digest = submission.idempotency_digest();
        self.create_record(
            request_digest,
            idempotency_key_digest,
            submission.retention_millis,
            PersistedWorkload::Program {
                submission: Box::new(submission),
            },
            now,
            None,
        )
    }

    fn create_record(
        &self,
        request_digest: Digest,
        idempotency_key_digest: Option<Digest>,
        retention_millis: u64,
        workload: PersistedWorkload,
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
            PersistedWorkload::Program { submission } => submission.requested_outputs,
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
            workload: workload.clone(),
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
        let path = self.job_dir(job_id).join("completion.json");
        if !path.exists() {
            return Ok(None);
        }
        let completion: DriverCompletion = read_json(&path)?;
        if completion.schema_version != 1
            || completion
                .message
                .as_ref()
                .is_some_and(|value| value.len() > 16 * 1024)
        {
            return Err(NativeExecutionError::Protocol(
                "durable driver completion is malformed".into(),
            ));
        }
        Ok(Some(completion))
    }

    pub(super) fn driver_marker(
        &self,
        job_id: JobId,
    ) -> NativeExecutionResult<Option<DriverMarker>> {
        let path = self.job_dir(job_id).join("driver.json");
        if !path.exists() {
            return Ok(None);
        }
        let marker: DriverMarker = read_json(&path)?;
        if marker.schema_version != 1 || marker.process_id == 0 {
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
            value: self
                .completion(job_id)?
                .and_then(|completion| completion.value),
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

    pub(super) fn gc(&self, now: u64) -> NativeExecutionResult<Vec<JobId>> {
        let expired = self
            .list()?
            .into_iter()
            .filter(|record| record.state.is_terminal() && record.retain_until_unix_millis <= now)
            .map(|record| record.handle.id)
            .collect::<Vec<_>>();
        for job_id in &expired {
            fs::remove_dir_all(self.job_dir(*job_id)).map_err(io_error)?;
        }
        Ok(expired)
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

pub(super) fn write_completion(
    job_dir: &Path,
    completion: &DriverCompletion,
) -> NativeExecutionResult<()> {
    atomic_json(&job_dir.join("completion.json"), completion)
}

pub(super) fn write_driver_marker(job_dir: &Path, process_id: u32) -> NativeExecutionResult<()> {
    if process_id == 0 {
        return Err(NativeExecutionError::Protocol(
            "durable driver process id must be non-zero".into(),
        ));
    }
    atomic_json(
        &job_dir.join("driver.json"),
        &DriverMarker {
            schema_version: 1,
            process_id,
        },
    )
}

pub(super) fn load_driver_invocation(
    job_dir: &Path,
) -> NativeExecutionResult<BatchDriverInvocation> {
    if !job_dir.is_absolute() {
        return Err(NativeExecutionError::Configuration(
            "durable driver job directory must be absolute".into(),
        ));
    }
    let request: PersistedRequest = read_json(&job_dir.join("request.json"))?;
    let expected_name = request.handle.id.to_string();
    if request.schema_version != 1
        || job_dir.file_name().and_then(|name| name.to_str()) != Some(expected_name.as_str())
    {
        return Err(NativeExecutionError::Protocol(
            "durable driver request identity is invalid".into(),
        ));
    }
    match request.workload {
        PersistedWorkload::Script {
            source_digest,
            source_name,
            arguments,
            working_directory,
        } => {
            let source_path = job_dir.join(&source_name);
            let source = fs::read(&source_path).map_err(io_error)?;
            if Digest::sha256(&source) != source_digest {
                return Err(NativeExecutionError::Protocol(
                    "durable driver source no longer matches its frozen request".into(),
                ));
            }
            let submission = BatchSubmission {
                source_name,
                source,
                arguments,
                working_directory,
                idempotency_key: None,
                retention_millis: request.retention_millis,
            };
            submission.validate()?;
            if submission.request_digest()? != request.request_digest {
                return Err(NativeExecutionError::Protocol(
                    "durable driver request no longer matches its identity".into(),
                ));
            }
            Ok(BatchDriverInvocation::Script {
                job_directory: job_dir.to_path_buf(),
                source_path,
                arguments: submission.arguments,
                working_directory: PathBuf::from(submission.working_directory),
            })
        }
        PersistedWorkload::Program { mut submission } => {
            submission.idempotency_key = None;
            submission.validate()?;
            if submission.request_digest()? != request.request_digest {
                return Err(NativeExecutionError::Protocol(
                    "durable program request no longer matches its identity".into(),
                ));
            }
            Ok(BatchDriverInvocation::Program {
                job_directory: job_dir.to_path_buf(),
                submission,
            })
        }
    }
}

fn read_chunk(path: &Path, offset: u64) -> NativeExecutionResult<(Vec<u8>, u64)> {
    if !path.exists() {
        return Ok((Vec::new(), offset));
    }
    let mut file = File::open(path).map_err(io_error)?;
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
