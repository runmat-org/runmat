//! Frozen driver invocation and marker files; job retention belongs to JobStore.
use super::filesystem::{atomic_json, io_error, read_json};
use super::model::{BatchDriverInvocation, BatchSubmission, DriverCompletion, DriverMarker};
use super::store::{PersistedRequest, PersistedWorkload};
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::Digest;
use std::path::{Path, PathBuf};

pub(super) fn write_completion(
    job_dir: &Path,
    completion: &DriverCompletion,
) -> NativeExecutionResult<()> {
    atomic_json(&job_dir.join("completion.json"), completion)
}

pub(super) fn write_driver_marker(
    job_dir: &Path,
    process_id: u32,
    attempt_id: Option<runmat_execution::identity::AttemptId>,
) -> NativeExecutionResult<()> {
    if process_id == 0 {
        return Err(NativeExecutionError::Protocol(
            "durable driver process id must be non-zero".into(),
        ));
    }
    atomic_json(
        &marker_path(job_dir, attempt_id),
        &DriverMarker {
            schema_version: if attempt_id.is_some() { 3 } else { 2 },
            process_id,
            attempt_id,
        },
    )
}

pub(super) fn marker_path(
    job_dir: &Path,
    attempt: Option<runmat_execution::identity::AttemptId>,
) -> PathBuf {
    match attempt {
        Some(id) => job_dir.join(format!("{id}.driver.json")),
        None => job_dir.join("driver.json"),
    }
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
            if !super::model::valid_source_name(&source_name) {
                return Err(NativeExecutionError::Protocol(
                    "frozen source name is invalid".into(),
                ));
            }
            let source_path = job_dir.join(&source_name);
            let source = runmat_filesystem::NativeFsProvider
                .read_bounded_sync(&source_path, super::model::MAX_BATCH_SOURCE_BYTES as u64)
                .map_err(io_error)?;
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
            if super::store::bind_organization(
                submission.request_digest()?,
                request.local_fea_organization,
            ) != request.request_digest
            {
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
            if super::store::bind_organization(
                submission.request_digest()?,
                request.local_fea_organization,
            ) != request.request_digest
            {
                return Err(NativeExecutionError::Protocol(
                    "durable program request no longer matches its identity".into(),
                ));
            }
            let attempt = super::program_store::load_assignment(job_dir, &submission.task)?;
            let lease = super::DurableAttemptLease::acquire(job_dir, &attempt.request)?;
            Ok(BatchDriverInvocation::Program {
                job_directory: job_dir.to_path_buf(),
                attempt: Box::new(attempt),
                submission,
                lease: Box::new(lease),
            })
        }
        PersistedWorkload::Graph { mut submission } => {
            submission.idempotency_key = None;
            submission.validate()?;
            if super::store::bind_organization(
                submission.request_digest()?,
                request.local_fea_organization,
            ) != request.request_digest
            {
                return Err(NativeExecutionError::Protocol(
                    "durable graph request no longer matches its identity".into(),
                ));
            }
            if let Some(task) = &submission.task {
                let attempt = super::program_store::load_assignment(job_dir, task)?;
                let lease = super::DurableAttemptLease::acquire(job_dir, &attempt.request)?;
                return Ok(BatchDriverInvocation::GraphTask {
                    job_directory: job_dir.to_path_buf(),
                    attempt: Box::new(attempt),
                    submission,
                    lease: Box::new(lease),
                });
            }
            Ok(BatchDriverInvocation::Graph {
                job_directory: job_dir.to_path_buf(),
                job_id: request.handle.id,
                submission,
            })
        }
    }
}
