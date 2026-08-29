use thiserror::Error;

use runmat_execution::{ProgramCallFrame, ProgramRuntimeFailure, ProgramSourceSpan};

use crate::{build_runtime_error, CallFrame, GpuGatherRetry, RuntimeError};

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ExecutionServiceError {
    #[error("execution handle belongs to a different scope")]
    ForeignScope,
    #[error("execution handle is unknown or stale")]
    UnknownHandle,
    #[error("execution was cancelled")]
    Cancelled,
    #[error("execution worker was lost: {0}")]
    WorkerLost(String),
    #[error("execution infrastructure failed: {0}")]
    Infrastructure(String),
    #[error("execution failed: {0}")]
    Failed(String),
    #[error("execution failed: {}", .0.message)]
    RuntimeFailure(Box<runmat_execution::ProgramRuntimeFailure>),
    #[error("requested output count exceeds the execution contract")]
    InvalidOutputContract,
}

impl ExecutionServiceError {
    /// Convert an execution-domain failure into its stable language-level
    /// diagnostic. This is the sole boundary between scheduler/service
    /// outcomes and user-visible runtime errors.
    pub fn into_runtime_error(self) -> RuntimeError {
        match self {
            Self::RuntimeFailure(failure) => {
                decode_runtime_failure(*failure).unwrap_or_else(|message| {
                    crate::runtime_error::semantic_error(
                        "RunMat:parallel:ExecutionProtocol",
                        message,
                    )
                })
            }
            Self::ForeignScope => crate::runtime_error::semantic_error(
                "RunMat:parallel:ForeignScope",
                self.to_string(),
            ),
            Self::UnknownHandle => crate::runtime_error::semantic_error(
                "RunMat:parallel:UnknownHandle",
                self.to_string(),
            ),
            Self::Cancelled => {
                crate::runtime_error::semantic_error("RunMat:parallel:Cancelled", self.to_string())
            }
            Self::WorkerLost(message) => crate::runtime_error::semantic_error(
                "RunMat:parallel:WorkerLost",
                format!("execution worker was lost: {message}"),
            ),
            Self::Infrastructure(message) => crate::runtime_error::semantic_error(
                "RunMat:parallel:InfrastructureFailure",
                format!("execution infrastructure failed: {message}"),
            ),
            Self::Failed(message) => crate::runtime_error::semantic_error(
                "RunMat:parallel:ExecutionFailed",
                format!("execution failed: {message}"),
            ),
            Self::InvalidOutputContract => crate::runtime_error::semantic_error(
                "RunMat:parallel:InvalidOutputContract",
                self.to_string(),
            ),
        }
    }
}

pub fn encode_runtime_failure(error: &RuntimeError) -> Result<ProgramRuntimeFailure, String> {
    let span = error
        .span
        .map(|span| encode_span(span.offset(), span.len()))
        .transpose()?;
    let call_frames = error
        .context
        .call_frames
        .iter()
        .map(|frame| {
            Ok(ProgramCallFrame {
                function: frame.function.clone(),
                source_id: frame
                    .source_id
                    .map(u64::try_from)
                    .transpose()
                    .map_err(|_| {
                        "runtime failure source identity exceeds its portable representation"
                            .to_string()
                    })?,
                span: frame
                    .span
                    .map(|(offset, length)| encode_span(offset, length))
                    .transpose()?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let failure = ProgramRuntimeFailure {
        message: error.message.clone(),
        identifier: error.identifier.clone(),
        span,
        builtin: error.context.builtin.clone(),
        task_id: error.context.task_id.clone(),
        call_frames,
        call_frames_elided: u32::try_from(error.context.call_frames_elided).map_err(|_| {
            "runtime failure elided-frame count exceeds its portable representation".to_string()
        })?,
        call_stack: error.context.call_stack.clone(),
        phase: error.context.phase.clone(),
    };
    failure.validate().map_err(|error| error.to_string())?;
    Ok(failure)
}

pub fn decode_runtime_failure(failure: ProgramRuntimeFailure) -> Result<RuntimeError, String> {
    failure.validate().map_err(|error| error.to_string())?;
    let mut builder =
        build_runtime_error(failure.message).with_gpu_gather_retry(GpuGatherRetry::Never);
    if let Some(identifier) = failure.identifier {
        builder = builder.with_identifier(identifier);
    }
    if let Some(span) = failure.span {
        let (offset, length) = decode_span_pair(span)?;
        builder = builder.with_span((offset, length).into());
    }
    if let Some(builtin) = failure.builtin {
        builder = builder.with_builtin(builtin);
    }
    if let Some(task_id) = failure.task_id {
        builder = builder.with_task_id(task_id);
    }
    if let Some(phase) = failure.phase {
        builder = builder.with_phase(phase);
    }
    builder = builder
        .with_call_frames(
            failure
                .call_frames
                .into_iter()
                .map(|frame| {
                    Ok(CallFrame {
                        function: frame.function,
                        source_id: frame.source_id.map(usize::try_from).transpose().map_err(
                            |_| "runtime failure source identity exceeds this host".to_string(),
                        )?,
                        span: frame.span.map(decode_span_pair).transpose()?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        )
        .with_call_frames_elided(failure.call_frames_elided as usize)
        .with_call_stack(failure.call_stack);
    Ok(builder.build())
}

fn encode_span(offset: usize, length: usize) -> Result<ProgramSourceSpan, String> {
    Ok(ProgramSourceSpan {
        offset: u64::try_from(offset)
            .map_err(|_| "runtime failure span offset exceeds its portable representation")?,
        length: u64::try_from(length)
            .map_err(|_| "runtime failure span length exceeds its portable representation")?,
    })
}

fn decode_span_pair(span: ProgramSourceSpan) -> Result<(usize, usize), String> {
    Ok((
        usize::try_from(span.offset)
            .map_err(|_| "runtime failure span offset exceeds this host")?,
        usize::try_from(span.length)
            .map_err(|_| "runtime failure span length exceeds this host")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_outcomes_have_distinct_stable_runtime_identifiers() {
        let cases = [
            (
                ExecutionServiceError::Cancelled,
                "RunMat:parallel:Cancelled",
            ),
            (
                ExecutionServiceError::WorkerLost("rank 2 disconnected".into()),
                "RunMat:parallel:WorkerLost",
            ),
            (
                ExecutionServiceError::Infrastructure("host spawn failed".into()),
                "RunMat:parallel:InfrastructureFailure",
            ),
            (
                ExecutionServiceError::Failed("program rejected input".into()),
                "RunMat:parallel:ExecutionFailed",
            ),
            (
                ExecutionServiceError::InvalidOutputContract,
                "RunMat:parallel:InvalidOutputContract",
            ),
        ];
        for (failure, identifier) in cases {
            assert_eq!(failure.into_runtime_error().identifier(), Some(identifier),);
        }
    }

    #[test]
    fn runtime_failure_round_trip_preserves_semantic_diagnostics() {
        let error = build_runtime_error("bad parallel value")
            .with_identifier("RunMat:ParallelValue")
            .with_builtin("sample")
            .with_task_id("task-7")
            .with_phase("execute")
            .with_span((11usize, 4usize).into())
            .with_call_frames(vec![CallFrame {
                function: "workerFunction".into(),
                source_id: Some(3),
                span: Some((7, 2)),
            }])
            .with_call_frames_elided(1)
            .with_call_stack(vec!["workerFunction at sample.m:2:4".into()])
            .build();
        let decoded = decode_runtime_failure(encode_runtime_failure(&error).unwrap()).unwrap();
        assert_eq!(decoded.message(), "bad parallel value");
        assert_eq!(decoded.identifier(), Some("RunMat:ParallelValue"));
        assert_eq!(
            decoded.span.map(|span| (span.offset(), span.len())),
            Some((11, 4))
        );
        assert_eq!(decoded.context.builtin.as_deref(), Some("sample"));
        assert_eq!(decoded.context.task_id.as_deref(), Some("task-7"));
        assert_eq!(decoded.context.phase.as_deref(), Some("execute"));
        assert_eq!(decoded.context.call_frames, error.context.call_frames);
        assert_eq!(decoded.context.call_frames_elided, 1);
        assert_eq!(decoded.context.call_stack, error.context.call_stack);
    }
}
