use thiserror::Error;

use runmat_execution::{
    ProgramCallFrame, ProgramDetachedDiagnosticSource, ProgramRuntimeFailure, ProgramSourceSpan,
};
use runmat_types::ProgramSourceId;

use crate::{
    build_runtime_error, CallFrame, CallFrameDiagnosticSource, GpuGatherRetry, RuntimeError,
};

#[derive(Clone, Debug, Error)]
pub enum ExecutionServiceError {
    #[error("future cannot be spawned: {0:?}")]
    SpawnUnsafe(runmat_types::SpawnSafetyReason),
    #[error("owned CPU control allocation failed: {0}")]
    HostAllocation(#[source] crate::context::host_allocation::HostAllocationError),
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
    /// Exact receipt-localized V8 failure. Repeated fetches share this error
    /// owner, including its selected source-frame provenance and cause chain.
    #[error("execution failed: {}", .0.message)]
    LocalizedRuntimeFailure(#[source] crate::runtime_error::RuntimeErrorOwner),
    #[error("requested output count exceeds the execution contract")]
    InvalidOutputContract,
}

impl PartialEq for ExecutionServiceError {
    fn eq(&self, other: &Self) -> bool {
        use ExecutionServiceError as Error;
        match (self, other) {
            (Error::SpawnUnsafe(a), Error::SpawnUnsafe(b)) => a == b,
            (Error::HostAllocation(a), Error::HostAllocation(b)) => a == b,
            (Error::ForeignScope, Error::ForeignScope)
            | (Error::UnknownHandle, Error::UnknownHandle)
            | (Error::Cancelled, Error::Cancelled)
            | (Error::InvalidOutputContract, Error::InvalidOutputContract) => true,
            (Error::WorkerLost(a), Error::WorkerLost(b))
            | (Error::Infrastructure(a), Error::Infrastructure(b))
            | (Error::Failed(a), Error::Failed(b)) => a == b,
            (Error::RuntimeFailure(a), Error::RuntimeFailure(b)) => a == b,
            // The localized diagnostic owns provenance and a possibly
            // downcastable cause chain; equality means the same retained
            // original, not equality of its presentation text.
            (Error::LocalizedRuntimeFailure(a), Error::LocalizedRuntimeFailure(b)) => {
                crate::runtime_error::RuntimeErrorOwner::ptr_eq(a, b)
            }
            _ => false,
        }
    }
}

impl Eq for ExecutionServiceError {}

impl From<RuntimeError> for ExecutionServiceError {
    fn from(error: RuntimeError) -> Self {
        Self::LocalizedRuntimeFailure(error.into())
    }
}

impl ExecutionServiceError {
    /// Present invocation-local service failures in their owning session's
    /// namespace instead of the parallel service API's default namespace.
    pub fn into_runtime_error_in_context(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> RuntimeError {
        if let Self::HostAllocation(cause) = self {
            return crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:Execution:CpuControl",
                "owned CPU control allocation",
                cause,
            );
        }
        let cancelled = matches!(self, Self::Cancelled);
        let mut error = self.into_runtime_error();
        if cancelled {
            let namespace = runtime.map_or_else(
                || "RunMat".to_owned(),
                crate::context::RuntimeContext::error_namespace,
            );
            error.identifier = Some(format!("{namespace}:Cancelled"));
        }
        error
    }

    /// Convert an execution-domain failure into its stable language-level
    /// diagnostic. This is the sole boundary between scheduler/service
    /// outcomes and user-visible runtime errors.
    pub fn into_runtime_error(self) -> RuntimeError {
        match self {
            Self::SpawnUnsafe(reason) => crate::runtime_error::semantic_error_with_source(
                "RunMat:parallel:SpawnUnsafe",
                Self::SpawnUnsafe(reason).to_string(),
                Self::SpawnUnsafe(reason),
            ),
            Self::HostAllocation(cause) => {
                crate::context::host_allocation::host_allocation_runtime_error(
                    None,
                    "RunMat:Execution:CpuControl",
                    "owned CPU control allocation",
                    cause,
                )
            }
            Self::LocalizedRuntimeFailure(error) => RuntimeError::from_shared_original(error),
            Self::RuntimeFailure(failure) => {
                decode_runtime_failure(*failure).unwrap_or_else(|message| {
                    crate::runtime_error::semantic_error(
                        "RunMat:parallel:ExecutionProtocol",
                        message,
                    )
                })
            }
            Self::ForeignScope => crate::runtime_error::semantic_error_with_source(
                "RunMat:parallel:ForeignScope",
                Self::ForeignScope.to_string(),
                Self::ForeignScope,
            ),
            Self::UnknownHandle => crate::runtime_error::semantic_error_with_source(
                "RunMat:parallel:UnknownHandle",
                Self::UnknownHandle.to_string(),
                Self::UnknownHandle,
            ),
            Self::Cancelled => {
                let mut error = crate::runtime_error::semantic_error_with_source(
                    "RunMat:parallel:Cancelled",
                    Self::Cancelled.to_string(),
                    Self::Cancelled,
                );
                error.failure_origin =
                    crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation;
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                error
            }
            Self::WorkerLost(message) => crate::runtime_error::semantic_error_with_source(
                "RunMat:parallel:WorkerLost",
                format!("execution worker was lost: {message}"),
                Self::WorkerLost(message),
            ),
            Self::Infrastructure(message) => crate::runtime_error::semantic_error_with_source(
                "RunMat:parallel:InfrastructureFailure",
                format!("execution infrastructure failed: {message}"),
                Self::Infrastructure(message),
            ),
            Self::Failed(message) => crate::runtime_error::semantic_error_with_source(
                "RunMat:parallel:ExecutionFailed",
                format!("execution failed: {message}"),
                Self::Failed(message),
            ),
            Self::InvalidOutputContract => crate::runtime_error::semantic_error_with_source(
                "RunMat:parallel:InvalidOutputContract",
                Self::InvalidOutputContract.to_string(),
                Self::InvalidOutputContract,
            ),
        }
    }
}

pub fn encode_runtime_failure(error: &RuntimeError) -> Result<ProgramRuntimeFailure, String> {
    encode_runtime_failure_with_sources(error, |frame| {
        if frame.source_id.is_some() {
            Err("runtime failure has a source ID without a selected portable binding".into())
        } else {
            Ok((None, None))
        }
    })
}

/// One frame-aware source admission for Direct failure and MException wires.
/// Numeric SourceIds alone cannot distinguish a retained old source table
/// from a later table reusing the same local slot.
pub(crate) fn portable_call_frame_source_in_context(
    frame: &CallFrame,
    runtime: &crate::context::RuntimeContext,
) -> Result<runmat_types::ProgramSourceId, &'static str> {
    let source_id = frame
        .source_id
        .ok_or("runtime failure frame has no source")?;
    let issued = crate::source_context::issued_source_ref_in_context(
        runtime,
        runmat_types::SourceId(source_id),
    )
    .ok_or("runtime failure refers to a missing or foreign source")?;
    let ordinal = runtime
        .portable_ordinal_for_source(issued)
        .ok_or("runtime failure source is absent from its portable binding")?;
    if let Some(actual) = &frame.source_owner {
        let expected = runtime
            .call_frame_source_owner_for_program_source(ordinal)
            .ok_or("runtime failure source has no selected issuer")?;
        if !actual.same_issuance(&expected) {
            return Err("runtime failure source belongs to a different issuance");
        }
    }
    Ok(ordinal)
}

/// Portable worker failures carry program ordinals, never worker-local source
/// integers. The selected invocation must still own the source leases here.
pub fn encode_runtime_failure_in_context(
    error: &RuntimeError,
    runtime: &crate::context::RuntimeContext,
) -> Result<ProgramRuntimeFailure, String> {
    let mut failure = encode_runtime_failure_with_sources(error, |frame| {
        frame.source_id.map_or(Ok((None, None)), |_| {
            portable_call_frame_source_in_context(frame, runtime)
                .map(|ordinal| (Some(ordinal), None))
                .map_err(str::to_owned)
        })
    })?;
    if failure.call_stack.is_empty() {
        failure.call_stack = render_source_call_stack_in_context(error, runtime);
        failure.validate().map_err(|error| error.to_string())?;
    }
    Ok(failure)
}

/// Encode a V8 worker failure against the exact ordered selected-unit source
/// projection. A frame's local SourceId is never portable authority: an
/// issuance token must match, or an untagged source must be unambiguous.
pub fn encode_runtime_failure_with_source_issuer_in_context(
    error: &RuntimeError,
    runtime: &crate::context::RuntimeContext,
    issuer: &crate::execution::value_codec::GraphSourceIssuer,
    selected_map: &runmat_execution::ExecutableSourceMapV2,
) -> Result<ProgramRuntimeFailure, String> {
    if !issuer.is_issued_in(runtime) {
        return Err("graph failure source issuer belongs to another session".into());
    }
    selected_map.validate().map_err(|error| error.to_string())?;
    let mut failure = encode_runtime_failure_with_sources(error, |frame| {
        if let Some(ordinal) = frame.source_id.and_then(|source| {
            issuer.source_ordinal(runmat_types::SourceId(source), frame.source_owner.as_ref())
        }) {
            selected_map
                .entries
                .get(usize::try_from(ordinal.0).map_err(|_| "source ordinal overflow")?)
                .filter(|record| record.source_id == ordinal)
                .ok_or_else(|| {
                    "graph failure source is absent from its selected map".to_string()
                })?;
            return Ok((Some(ordinal), None));
        }
        if let Some(diagnostic) = frame.diagnostic_source.as_ref() {
            let location = diagnostic.line_column(frame.span);
            return Ok((
                None,
                Some(ProgramDetachedDiagnosticSource {
                    display_name: diagnostic.display_name().to_owned(),
                    line: location.map(|(line, _)| line),
                    column: location.map(|(_, column)| column),
                }),
            ));
        }
        if frame.source_id.is_some() {
            Err("graph failure source has no unique selected issuer or detached diagnostic".into())
        } else {
            Ok((None, None))
        }
    })?;
    if failure.call_stack.is_empty() {
        failure.call_stack = issuer.render_call_stack(error);
        failure.validate().map_err(|error| error.to_string())?;
    }
    Ok(failure)
}

/// Render selected, authenticated source records while their invocation lease
/// is alive. This presentation is portable text; no worker SourceId escapes.
pub fn render_source_call_stack_in_context(
    error: &RuntimeError,
    runtime: &crate::context::RuntimeContext,
) -> Vec<String> {
    render_source_call_stack_with(error, |id| {
        crate::source_context::issued_source_ref_in_context(runtime, id)?;
        crate::source_context::source_info_in_context(runtime, id)
    })
}

/// Render with the exact selected graph issuer after source-unit selection.
/// Detached diagnostic presentation remains independent of live source code.
pub(crate) fn render_source_call_stack_with(
    error: &RuntimeError,
    mut source_info: impl FnMut(runmat_types::SourceId) -> Option<crate::source_context::SourceInfo>,
) -> Vec<String> {
    let mut rendered = Vec::new();
    if error.context.call_frames_elided > 0 {
        rendered.push(format!(
            "... {} frames elided ...",
            error.context.call_frames_elided
        ));
    }
    for frame in error.context.call_frames.iter().rev() {
        if let Some(diagnostic) = &frame.diagnostic_source {
            if let Some((line, column)) = diagnostic.line_column(frame.span) {
                rendered.push(format!(
                    "{} @ {}:{line}:{column}",
                    frame.function,
                    diagnostic.display_name()
                ));
                continue;
            }
            rendered.push(frame.function.clone());
            continue;
        }
        let source = frame
            .source_id
            .and_then(|id| source_info(runmat_types::SourceId(id)));
        let line = source
            .as_ref()
            .and_then(|source| {
                CallFrameDiagnosticSource::line_column_from_text(&source.text, frame.span).map(
                    |(line, column)| {
                        format!("{} @ {}:{line}:{column}", frame.function, source.name)
                    },
                )
            })
            .unwrap_or_else(|| frame.function.clone());
        rendered.push(line);
    }
    rendered
}

/// Preserve the presentation form of authenticated source frames before the
/// selected invocation releases their source leases.
pub fn preserve_source_call_stack_in_context(
    error: &mut RuntimeError,
    runtime: &crate::context::RuntimeContext,
) {
    if error.context.call_stack.is_empty() && !error.context.call_frames.is_empty() {
        error.context.call_stack = render_source_call_stack_in_context(error, runtime);
    }
}

fn encode_runtime_failure_with_sources(
    error: &RuntimeError,
    mut portable_source: impl FnMut(
        &CallFrame,
    ) -> Result<
        (
            Option<ProgramSourceId>,
            Option<ProgramDetachedDiagnosticSource>,
        ),
        String,
    >,
) -> Result<ProgramRuntimeFailure, String> {
    let span = error
        .span
        .map(|span| encode_span(span.offset(), span.len()))
        .transpose()?;
    let call_frames = error
        .context
        .call_frames
        .iter()
        .map(|frame| {
            let (program_source_id, detached_diagnostic) = portable_source(frame)?;
            Ok(ProgramCallFrame {
                function: frame.function.clone(),
                program_source_id,
                span: frame
                    .span
                    .map(|(offset, length)| encode_span(offset, length))
                    .transpose()?,
                detached_diagnostic,
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
    decode_runtime_failure_with_sources(failure, |_, detached| {
        if detached.is_some() {
            return Err("Direct runtime failure has an unsupported detached diagnostic".into());
        }
        Ok((None, None, None))
    })
}

/// Localize a V8 failure only under the receipt's exact issuing session and
/// selected unit projection. The result carries local presentation text but
/// gains no callable or source-map admission from the worker's ordinal.
pub fn decode_runtime_failure_with_source_issuer_in_context(
    failure: &ProgramRuntimeFailure,
    runtime: &crate::context::RuntimeContext,
    issuer: &crate::execution::value_codec::GraphSourceIssuer,
    selected_map: &runmat_execution::ExecutableSourceMapV2,
) -> Result<RuntimeError, String> {
    if !issuer.is_issued_in(runtime) {
        return Err("graph failure source issuer belongs to another session".into());
    }
    selected_map.validate().map_err(|error| error.to_string())?;
    decode_runtime_failure_with_sources(failure.clone(), |ordinal, detached| {
        let Some(ordinal) = ordinal else {
            let diagnostic =
                detached.map(|diagnostic| match (diagnostic.line, diagnostic.column) {
                    (Some(line), Some(column)) => CallFrameDiagnosticSource::detached(
                        std::sync::Arc::from(diagnostic.display_name.as_str()),
                        line,
                        column,
                    ),
                    (None, None) => CallFrameDiagnosticSource::detached_unknown(
                        std::sync::Arc::from(diagnostic.display_name.as_str()),
                    ),
                    _ => unreachable!("program failure validation rejects partial coordinates"),
                });
            return Ok((None, None, diagnostic));
        };
        selected_map
            .entries
            .get(usize::try_from(ordinal.0).map_err(|_| "source ordinal overflow")?)
            .filter(|record| record.source_id == ordinal)
            .ok_or_else(|| "graph failure source is absent from its selected map".to_string())?;
        let (source, basis) = issuer
            .frame_source(ordinal)
            .ok_or_else(|| "graph failure source has no selected local issuer".to_string())?;
        let diagnostic = issuer
            .frame_diagnostic_source(ordinal)
            .ok_or_else(|| "graph failure source has no selected presentation".to_string())?;
        Ok((Some(source.0), Some(basis), Some(diagnostic)))
    })
}

fn decode_runtime_failure_with_sources(
    failure: ProgramRuntimeFailure,
    mut local_source: impl FnMut(
        Option<ProgramSourceId>,
        Option<&ProgramDetachedDiagnosticSource>,
    ) -> Result<
        (
            Option<usize>,
            Option<crate::CallFrameSourceOwner>,
            Option<CallFrameDiagnosticSource>,
        ),
        String,
    >,
) -> Result<RuntimeError, String> {
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
                    let (source_id, source_owner, diagnostic_source) =
                        local_source(frame.program_source_id, frame.detached_diagnostic.as_ref())?;
                    Ok(CallFrame {
                        function: frame.function,
                        source_id,
                        span: frame.span.map(decode_span_pair).transpose()?,
                        source_owner,
                        diagnostic_source,
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
    fn localized_worker_failure_keeps_its_exact_runtime_owner_across_cached_fetches() {
        let original: crate::runtime_error::RuntimeErrorOwner =
            crate::runtime_error::semantic_error("RunMat:worker:Exact", "selected failure").into();
        let cached = ExecutionServiceError::LocalizedRuntimeFailure(original.clone());
        assert_eq!(cached, cached.clone());
        for visible in [
            cached.clone().into_runtime_error(),
            cached.into_runtime_error(),
        ] {
            assert_eq!(visible.identifier(), Some("RunMat:worker:Exact"));
            let shared = visible
                .source
                .as_deref()
                .and_then(|source| {
                    source.downcast_ref::<crate::runtime_error::SharedRuntimeErrorSource>()
                })
                .expect("each fetch retains the same original typed cause");
            assert!(std::ptr::eq(shared.original(), original.as_ref()));
        }
    }

    #[test]
    fn detached_frame_keeps_exact_presentation_after_source_retirement() {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let diagnostic = crate::CallFrameDiagnosticSource::new(
            std::sync::Arc::from("retired.m"),
            std::sync::Arc::from("first\nsecond"),
        );
        let error = build_runtime_error("failure")
            .with_call_frames(vec![CallFrame {
                function: "retired".into(),
                source_id: None,
                span: Some((6, 1)),
                source_owner: None,
                diagnostic_source: Some(diagnostic),
            }])
            .build();
        assert_eq!(
            render_source_call_stack_in_context(&error, &runtime),
            vec!["retired @ retired.m:2:1".to_owned()]
        );

        let without_location = build_runtime_error("failure")
            .with_call_frames(vec![CallFrame {
                function: "retired".into(),
                source_id: Some(7),
                span: None,
                source_owner: None,
                diagnostic_source: Some(crate::CallFrameDiagnosticSource::new(
                    std::sync::Arc::from("retired.m"),
                    std::sync::Arc::from("first\nsecond"),
                )),
            }])
            .build();
        assert_eq!(
            render_source_call_stack_in_context(&without_location, &runtime),
            vec!["retired".to_owned()],
            "captured presentation without a span cannot fall through to a later source table"
        );
    }

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
            let visible = failure.clone().into_runtime_error();
            assert_eq!(visible.identifier(), Some(identifier));
            if matches!(&failure, ExecutionServiceError::Cancelled) {
                assert_eq!(visible.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert!(crate::runtime_error::error_has_control_cancellation_origin(
                    &visible
                ));
            }
            let cause = visible
                .source
                .as_deref()
                .and_then(|source| source.downcast_ref::<ExecutionServiceError>());
            assert_eq!(cause, Some(&failure));
        }
    }

    #[test]
    fn runtime_failure_round_trip_preserves_semantic_diagnostics() {
        let base = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _unrelated = crate::source_context::lease_source_context_in_context(
            &base,
            "unrelated.m",
            None,
            "unrelated = 1;",
        )
        .expect("advance source issuer before worker map");
        let text = "error('bad parallel value')";
        let map = runmat_execution::ExecutableSourceMapV2::new(vec![
            runmat_execution::ExecutableSourceRecordV2 {
                source_id: runmat_types::ProgramSourceId(0),
                owner_identity: "worker-fixture".into(),
                relative_path: "remote_spmd.m".into(),
                display_name: "remote_spmd.m".into(),
                text: text.into(),
                content_revision: runmat_execution::Digest::sha256(text.as_bytes()),
            },
        ])
        .expect("portable source map");
        let runtime = crate::source_context::bind_portable_source_map_v2(base, &map)
            .expect("selected program source");
        let source_id = runtime
            .program_source_id(runmat_types::ProgramSourceId(0))
            .expect("issued source");
        let error = build_runtime_error("bad parallel value")
            .with_identifier("RunMat:ParallelValue")
            .with_builtin("sample")
            .with_task_id("task-7")
            .with_phase("execute")
            .with_span((11usize, 4usize).into())
            .with_call_frames(vec![CallFrame {
                function: "workerFunction".into(),
                source_id: Some(source_id.0),
                span: Some((7, 2)),
                source_owner: None,
                diagnostic_source: None,
            }])
            .with_call_frames_elided(1)
            .with_call_stack(vec!["workerFunction at sample.m:2:4".into()])
            .build();
        let portable = encode_runtime_failure_in_context(&error, &runtime).unwrap();
        assert_eq!(
            portable.call_frames[0].program_source_id,
            Some(runmat_types::ProgramSourceId(0))
        );
        assert_ne!(
            portable.call_frames[0]
                .program_source_id
                .map(|id| id.0 as usize),
            Some(source_id.0)
        );
        let selected = runtime
            .call_frame_source_owner_for_program_source(runmat_types::ProgramSourceId(0))
            .unwrap();
        for (owner, accepted) in [
            (selected, true),
            (crate::CallFrameSourceOwner::issue(), false),
        ] {
            let frame_error = build_runtime_error("worker failure")
                .with_call_frames(vec![CallFrame {
                    function: "workerFunction".into(),
                    source_id: Some(source_id.0),
                    span: Some((7, 2)),
                    source_owner: Some(owner),
                    diagnostic_source: None,
                }])
                .build();
            assert_eq!(
                encode_runtime_failure_in_context(&frame_error, &runtime).is_ok(),
                accepted,
                "portable worker failure must honor the exact source-table issuer"
            );
        }
        let decoded = decode_runtime_failure(portable).unwrap();
        assert_eq!(decoded.message(), "bad parallel value");
        assert_eq!(decoded.identifier(), Some("RunMat:ParallelValue"));
        assert_eq!(
            decoded.span.map(|span| (span.offset(), span.len())),
            Some((11, 4))
        );
        assert_eq!(decoded.context.builtin.as_deref(), Some("sample"));
        assert_eq!(decoded.context.task_id.as_deref(), Some("task-7"));
        assert_eq!(decoded.context.phase.as_deref(), Some("execute"));
        assert_eq!(decoded.context.call_frames[0].function, "workerFunction");
        assert_eq!(decoded.context.call_frames[0].source_id, None);
        assert_eq!(decoded.context.call_frames[0].span, Some((7, 2)));
        assert_eq!(decoded.context.call_frames_elided, 1);
        assert_eq!(decoded.context.call_stack, error.context.call_stack);

        let mut unrendered = error;
        unrendered.context.call_stack.clear();
        preserve_source_call_stack_in_context(&mut unrendered, &runtime);
        assert_eq!(
            unrendered.context.call_stack,
            vec![
                "... 1 frames elided ...".to_string(),
                "workerFunction @ remote_spmd.m:1:8".to_string(),
            ]
        );
        let portable = encode_runtime_failure_in_context(&unrendered, &runtime).unwrap();
        assert_eq!(
            portable.call_frames[0].program_source_id,
            Some(runmat_types::ProgramSourceId(0))
        );
        assert_eq!(
            portable.call_stack,
            vec![
                "... 1 frames elided ...".to_string(),
                "workerFunction @ remote_spmd.m:1:8".to_string(),
            ]
        );
    }
}
