//! Immutable original attempt assignments and provider reports; job IDs only route them.
use super::filesystem::{atomic_json, immutable_json, read_json, read_optional_json};
#[cfg(all(test, unix))]
use super::model::ProgramBatchSubmission;
use super::model::{DurableProgramAttempt, ProgramBatchExecution};
use super::store::{JobStore, PersistedRequest, PersistedWorkload};
use crate::{NativeExecutionError, NativeExecutionResult};
use runmat_execution::{identity::AttemptId, JobId};
use runmat_execution_runner::AttemptReport;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub(super) const ATTEMPT_ID_ENV: &str = "RUNMAT_EXECUTION_ASSIGNED_ATTEMPT";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Completion {
    schema_version: u16,
    response: crate::protocol::WorkerResponse,
    report: runmat_execution_runner::port::BackendReport,
    latest_progress: Option<crate::protocol::ProgramProgress>,
}

impl JobStore {
    #[cfg(all(test, unix))]
    pub(super) fn program_submission(
        &self,
        id: JobId,
    ) -> NativeExecutionResult<Option<ProgramBatchSubmission>> {
        Ok(match self.retained_submission(id)? {
            Some(super::retained::RetainedSubmission::Direct(s)) => Some(*s),
            _ => None,
        })
    }
    pub(super) fn retained_submission(
        &self,
        id: JobId,
    ) -> NativeExecutionResult<Option<super::retained::RetainedSubmission>> {
        let request: PersistedRequest = read_json(&self.job_dir(id).join("request.json"))?;
        if request.schema_version != 1 || request.handle.id != id {
            return Err(invalid("durable routing request identity is invalid"));
        }
        match request.workload {
            PersistedWorkload::Script { .. } => Ok(None),
            PersistedWorkload::Program { submission } => {
                submission.validate()?;
                if super::store::bind_organization(
                    submission.request_digest()?,
                    request.local_fea_organization,
                ) != request.request_digest
                {
                    return Err(invalid(
                        "durable program differs from its original invocation",
                    ));
                }
                Ok(Some(super::retained::RetainedSubmission::Direct(
                    submission,
                )))
            }
            PersistedWorkload::Graph { submission } => {
                submission.validate()?;
                if super::store::bind_organization(
                    submission.request_digest()?,
                    request.local_fea_organization,
                ) != request.request_digest
                {
                    return Err(invalid(
                        "durable Graph differs from its original invocation",
                    ));
                }
                if submission.task.is_some() {
                    Ok(Some(super::retained::RetainedSubmission::Graph(submission)))
                } else {
                    Ok(None)
                }
            }
        }
    }

    pub(super) fn assign_program(
        &self,
        id: JobId,
        assignment: &DurableProgramAttempt,
    ) -> NativeExecutionResult<()> {
        let submission = self
            .retained_submission(id)?
            .ok_or_else(|| invalid("workload has no original retained task assignment"))?;
        validate_assignment(assignment, submission.task())?;
        immutable_json(
            &assignment_path(&self.job_dir(id), assignment.request.id),
            assignment,
        )?;
        // Routing pointer for launch only. The detached provider opens the exact
        // per-attempt file named in its environment, never a later pointer value.
        atomic_json(&self.job_dir(id).join("assignment.json"), assignment)
    }

    pub(super) fn program_assignment(
        &self,
        id: JobId,
    ) -> NativeExecutionResult<Option<DurableProgramAttempt>> {
        let Some(submission) = self.retained_submission(id)? else {
            return Ok(None);
        };
        let path = self.job_dir(id).join("assignment.json");
        let Some(assignment) = read_optional_json(&path)? else {
            return Ok(None);
        };
        validate_assignment(&assignment, submission.task())?;
        Ok(Some(assignment))
    }

    pub(super) fn require_program_assignment(
        &self,
        id: JobId,
        request: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<()> {
        let assignment: DurableProgramAttempt =
            read_json(&assignment_path(&self.job_dir(id), request.id))?;
        let submission = self
            .retained_submission(id)?
            .ok_or_else(|| invalid("portable program control cannot route to a script"))?;
        validate_assignment(&assignment, submission.task())?;
        if assignment.request != *request {
            return Err(invalid(
                "provider control differs from its original immutable assignment",
            ));
        }
        Ok(())
    }

    pub(super) fn program_report(
        &self,
        id: JobId,
        assignment: &runmat_execution_runner::AttemptRequest,
    ) -> NativeExecutionResult<Option<ProgramBatchExecution>> {
        let path = report_path(&self.job_dir(id), assignment.id);
        let Some(completed) = read_optional_json::<Completion>(&path)? else {
            return Ok(None);
        };
        let submission = self
            .retained_submission(id)?
            .ok_or_else(|| invalid("program report routes to a script"))?;
        if completed.schema_version != 1
            || completed.report.attempt_id != assignment.id
            || completed.report.task_id != assignment.task_id
            || completed.report.worker_id != assignment.worker_id
            || completed.report.driver_fence != assignment.driver_fence
            || submission.task().request != assignment.task
        {
            return Err(invalid(
                "durable provider report differs from its original assigned task",
            ));
        }
        submission.validate_response(&completed.response)?;
        if let Some(progress) = &completed.latest_progress {
            progress.validate().map_err(invalid)?;
        }
        if let Some(resources) = &completed.report.unsuccessful_resources {
            resources.validate_against(assignment).map_err(invalid)?;
        }
        if let Some(execution) = &completed.report.execution {
            execution.validate_against(assignment).map_err(invalid)?;
        }
        if matches!(completed.report.report, AttemptReport::Started) {
            return Err(invalid("durable completion is not terminal"));
        }
        if let AttemptReport::Succeeded { result } = &completed.report.report {
            result.validate_against(assignment).map_err(invalid)?;
            if let (
                crate::protocol::WorkerResponse::GraphSuccess { graph },
                runmat_execution_runner::AttemptSuccess::Graph {
                    graph: reported, ..
                },
            ) = (&completed.response, result)
            {
                if graph != reported {
                    return Err(invalid(
                        "durable Graph report differs from its exact original response",
                    ));
                }
            } else {
                let (outputs, objects) = match &completed.response {
                    crate::protocol::WorkerResponse::Success { outputs } => (
                        outputs.as_slice(),
                        &[] as &[runmat_execution::value::ValueRef],
                    ),
                    crate::protocol::WorkerResponse::ExternalizedSuccess {
                        outputs,
                        result_objects,
                    } => (outputs.as_slice(), result_objects.as_slice()),
                    _ => {
                        return Err(invalid(
                            "successful provider report contradicts original response",
                        ))
                    }
                };
                if result.values() != Some((outputs, objects)) {
                    return Err(invalid(
                        "durable provider result differs from the original response",
                    ));
                }
            }
        } else if !matches!(
            completed.response,
            crate::protocol::WorkerResponse::Failure { .. }
                | crate::protocol::WorkerResponse::RuntimeFailure { .. }
        ) {
            return Err(invalid(
                "unsuccessful provider report carries a successful response",
            ));
        }
        Ok(Some(ProgramBatchExecution {
            response: completed.response,
            report: completed.report,
            latest_progress: completed.latest_progress,
        }))
    }
}

pub(super) fn load_assignment(
    job: &Path,
    submission: &runmat_execution_runner::TaskSubmission,
) -> NativeExecutionResult<DurableProgramAttempt> {
    let encoded = std::env::var(ATTEMPT_ID_ENV)
        .map_err(|_| invalid("durable provider lacks its original assignment identity"))?;
    let id: AttemptId = encoded
        .parse()
        .map_err(|error: runmat_execution::ContractError| invalid(error.to_string()))?;
    let assignment: DurableProgramAttempt = read_json(&assignment_path(job, id))?;
    validate_assignment(&assignment, submission)?;
    if assignment.request.id != id || assignment.job_directory != job {
        return Err(invalid(
            "durable assignment differs from its original routing directory",
        ));
    }
    Ok(assignment)
}

pub(super) fn write_program_completion(
    job: &Path,
    execution: ProgramBatchExecution,
) -> NativeExecutionResult<()> {
    let path = report_path(job, execution.report.attempt_id);
    immutable_json(
        &path,
        &Completion {
            schema_version: 1,
            response: execution.response,
            report: execution.report,
            latest_progress: execution.latest_progress,
        },
    )
}

pub(super) fn cancel_path(job: &Path, id: AttemptId) -> PathBuf {
    job.join(format!("{id}.cancel.json"))
}
fn assignment_path(job: &Path, id: AttemptId) -> PathBuf {
    job.join(format!("{id}.assignment.json"))
}
fn report_path(job: &Path, id: AttemptId) -> PathBuf {
    job.join(format!("{id}.completion.json"))
}
fn validate_assignment(
    assignment: &DurableProgramAttempt,
    submission: &runmat_execution_runner::TaskSubmission,
) -> NativeExecutionResult<()> {
    if assignment.job_directory.parent()
        != Some(assignment.config.store_root.join("jobs").as_path())
        || assignment.schema_version != 1
        || assignment.request.task_id != submission.request.id
        || assignment.request.task != submission.request
    {
        return Err(invalid(
            "durable assignment lacks the original caller-admitted task",
        ));
    }
    assignment.config.validate().map_err(invalid)
}
fn invalid(message: impl Into<String>) -> NativeExecutionError {
    NativeExecutionError::Protocol(message.into())
}
