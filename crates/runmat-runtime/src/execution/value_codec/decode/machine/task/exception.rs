use super::super::*;
use super::text::TextCopy;
use runmat_async::RuntimeErrorOwner;
use runmat_execution::value::{
    ExceptionFailureOrigin, ExceptionGpuGatherRetry, ExceptionRuntimeMetadata, ExceptionValue,
};

enum Stage {
    Identifier,
    Message,
    Stack(usize),
    Builtin,
    Task,
    Frame(usize),
    CallStack(usize),
    Phase,
    Complete,
}

pub(in crate::execution::value_codec::decode::machine) struct ExceptionTask<'a> {
    source: &'a ExceptionValue,
    stage: Stage,
    copy: Option<TextCopy<'a>>,
    identifier: Option<String>,
    message: Option<String>,
    stack: Vec<String>,
    builtin: Option<String>,
    task_id: Option<String>,
    frames: Vec<crate::CallFrame>,
    call_stack: Vec<String>,
    phase: Option<String>,
    path: String,
    frame_sources:
        Option<&'a dyn crate::execution::value_codec::decode::InlineFrameSourceAuthority>,
}

impl<'a> ExceptionTask<'a> {
    pub(super) fn new(
        source: &'a ExceptionValue,
        path: String,
        frame_sources: Option<
            &'a dyn crate::execution::value_codec::decode::InlineFrameSourceAuthority,
        >,
    ) -> Self {
        Self {
            source,
            stage: Stage::Identifier,
            copy: None,
            identifier: None,
            message: None,
            stack: Vec::with_capacity(source.stack.len()),
            builtin: None,
            task_id: None,
            frames: Vec::with_capacity(source.runtime.as_ref().map_or(0, |m| m.call_frames.len())),
            call_stack: Vec::with_capacity(
                source.runtime.as_ref().map_or(0, |m| m.call_stack.len()),
            ),
            phase: None,
            path,
            frame_sources,
        }
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        if !self.source.causes.is_empty() {
            return Err(ValueCodecError::unsupported(
                &self.path,
                "nested exception causes are not represented by the current runtime MException",
            ));
        }
        loop {
            let source = match self.current_source() {
                Some(source) => source,
                None if matches!(self.stage, Stage::Complete) => {
                    return Ok((1, Some(self.finish()?)));
                }
                None => {
                    self.skip_absent_stage();
                    continue;
                }
            };
            let copy = self.copy.get_or_insert_with(|| TextCopy::new(source));
            let completed = copy.advance(maximum_work);
            if copy.is_complete() {
                let text = self.copy.take().expect("active exception text").finish();
                self.accept(text)?;
            }
            return Ok((completed, None));
        }
    }

    fn metadata(&self) -> Option<&'a ExceptionRuntimeMetadata> {
        self.source.runtime.as_ref()
    }

    fn current_source(&self) -> Option<&'a str> {
        match self.stage {
            Stage::Identifier => Some(&self.source.identifier),
            Stage::Message => Some(&self.source.message),
            Stage::Stack(index) => self.source.stack.get(index).map(String::as_str),
            Stage::Builtin => self.metadata()?.builtin.as_deref(),
            Stage::Task => self.metadata()?.task_id.as_deref(),
            Stage::Frame(index) => self
                .metadata()?
                .call_frames
                .get(index)
                .map(|frame| frame.function.as_str()),
            Stage::CallStack(index) => self.metadata()?.call_stack.get(index).map(String::as_str),
            Stage::Phase => self.metadata()?.phase.as_deref(),
            Stage::Complete => None,
        }
    }

    fn skip_absent_stage(&mut self) {
        self.stage = match self.stage {
            Stage::Stack(_) => Stage::Builtin,
            Stage::Builtin => Stage::Task,
            Stage::Task => Stage::Frame(0),
            Stage::Frame(_) => Stage::CallStack(0),
            Stage::CallStack(_) => Stage::Phase,
            Stage::Phase => Stage::Complete,
            Stage::Identifier | Stage::Message | Stage::Complete => Stage::Complete,
        };
    }

    fn accept(&mut self, text: String) -> Result<(), ValueCodecError> {
        self.stage = match self.stage {
            Stage::Identifier => {
                self.identifier = Some(text);
                Stage::Message
            }
            Stage::Message => {
                self.message = Some(text);
                Stage::Stack(0)
            }
            Stage::Stack(index) => {
                self.stack.push(text);
                Stage::Stack(index + 1)
            }
            Stage::Builtin => {
                self.builtin = Some(text);
                Stage::Task
            }
            Stage::Task => {
                self.task_id = Some(text);
                Stage::Frame(0)
            }
            Stage::Frame(index) => {
                let metadata = self.metadata().expect("frame metadata");
                let frame = &metadata.call_frames[index];
                let (source_id, source_owner) =
                    match (frame.program_source_id(), self.frame_sources) {
                        (Some(ordinal), Some(sources)) => {
                            let (source, owner) =
                                sources.frame_source(ordinal).ok_or_else(|| {
                                    ValueCodecError::invalid(
                                &self.path,
                                "exception frame source is absent from selected graph units",
                            )
                                })?;
                            (Some(source.0), Some(owner))
                        }
                        _ => (None, None),
                    };
                self.frames.push(crate::CallFrame {
                    function: text,
                    source_id,
                    span: local_span(frame.span, &self.path)?,
                    source_owner,
                    diagnostic_source: match &frame.source {
                        runmat_execution::value::ExceptionFrameSource::Detached(presentation) => {
                            match (presentation.line, presentation.column) {
                                (Some(line), Some(column)) => {
                                    Some(crate::CallFrameDiagnosticSource::detached(
                                        std::sync::Arc::from(presentation.display_name.as_str()),
                                        line,
                                        column,
                                    ))
                                }
                                (None, None) => {
                                    Some(crate::CallFrameDiagnosticSource::detached_unknown(
                                        std::sync::Arc::from(presentation.display_name.as_str()),
                                    ))
                                }
                                _ => {
                                    return Err(ValueCodecError::invalid(
                                        &self.path,
                                        "detached diagnostic coordinates must be present together",
                                    ))
                                }
                            }
                        }
                        runmat_execution::value::ExceptionFrameSource::Admitted(_)
                        | runmat_execution::value::ExceptionFrameSource::Unknown => None,
                    },
                });
                Stage::Frame(index + 1)
            }
            Stage::CallStack(index) => {
                self.call_stack.push(text);
                Stage::CallStack(index + 1)
            }
            Stage::Phase => {
                self.phase = Some(text);
                Stage::Complete
            }
            Stage::Complete => Stage::Complete,
        };
        Ok(())
    }

    fn finish(&mut self) -> Result<Value, ValueCodecError> {
        let identifier = self.identifier.take().expect("decoded identifier");
        let message = self.message.take().expect("decoded message");
        let stack = std::mem::take(&mut self.stack);
        let exception = if let Some(metadata) = self.metadata() {
            let mut error = crate::RuntimeError::new(message.clone());
            error.identifier = Some(identifier.clone());
            error.span = local_span(metadata.span, &self.path)?.map(Into::into);
            error.context = crate::ErrorContext {
                builtin: self.builtin.take(),
                task_id: self.task_id.take(),
                call_frames: std::mem::take(&mut self.frames),
                call_frames_elided: local(metadata.call_frames_elided, &self.path)?,
                call_stack: std::mem::take(&mut self.call_stack),
                phase: self.phase.take(),
            };
            error.gpu_gather_retry = match metadata.gpu_gather_retry {
                ExceptionGpuGatherRetry::Legacy => crate::GpuGatherRetry::Legacy,
                ExceptionGpuGatherRetry::Never => crate::GpuGatherRetry::Never,
                ExceptionGpuGatherRetry::Requested => crate::GpuGatherRetry::Requested,
            };
            error.failure_origin = match metadata.failure_origin {
                ExceptionFailureOrigin::Unclassified => {
                    runmat_async::RuntimeFailureOrigin::Unclassified
                }
                ExceptionFailureOrigin::LanguageEvaluation => {
                    runmat_async::RuntimeFailureOrigin::LanguageEvaluation
                }
            };
            MException::from_runtime_error(
                identifier,
                message,
                stack,
                RuntimeErrorOwner::new(error),
            )
        } else {
            let mut exception = MException::new(identifier, message);
            exception.stack = stack;
            exception
        };
        Ok(Value::MException(exception))
    }

    pub(super) fn path(&self) -> &str {
        &self.path
    }
}

fn local(value: u64, path: &str) -> Result<usize, ValueCodecError> {
    usize::try_from(value)
        .map_err(|_| ValueCodecError::invalid(path, "exception metadata exceeds local width"))
}

fn local_span(
    value: Option<(u64, u64)>,
    path: &str,
) -> Result<Option<(usize, usize)>, ValueCodecError> {
    value
        .map(|(start, length)| Ok((local(start, path)?, local(length, path)?)))
        .transpose()
}
