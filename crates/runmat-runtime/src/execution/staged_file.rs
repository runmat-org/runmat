//! Shared cooperative filesystem staging and one-publication outcome owner.
//!
//! Callers own their selected provider/path, validation and diagnostics. This
//! module owns the partial-write and commit lifecycle for every builtin that
//! asks an `FsProvider` for an exclusive staged replacement.

use std::cell::{Cell, RefCell};
use std::future::{poll_fn, Future};
use std::io;

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_filesystem::{FsPublicationDisposition, FsPublicationFailure, FsStagedOutput};

#[derive(Debug)]
pub(crate) enum StagedFileError {
    Io(io::Error),
    Control(RuntimeError),
    Publication(FsPublicationFailure),
    PublicationInterrupted(RuntimeError, FsPublicationDisposition),
}

#[derive(Clone, Copy)]
pub(crate) struct StagedWriteMessages {
    pub(crate) no_progress: &'static str,
    pub(crate) excess_progress: &'static str,
}

pub(crate) trait StagedWriteCooperation {
    async fn completed(&mut self, work: usize) -> BuiltinResult<()>;
}

pub(crate) struct StagedCpuCooperation<'a> {
    runtime: Option<&'a RuntimeContext>,
    budget: Option<CpuCooperationBudget>,
    builtin: &'static str,
}

impl<'a> StagedCpuCooperation<'a> {
    pub(crate) fn new(runtime: Option<&'a RuntimeContext>, builtin: &'static str) -> Self {
        Self {
            runtime,
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            builtin,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_test_budget(
        runtime: &'a RuntimeContext,
        builtin: &'static str,
        budget: CpuCooperationBudget,
    ) -> Self {
        Self {
            runtime: Some(runtime),
            budget: Some(budget),
            builtin,
        }
    }

    pub(crate) async fn completed(&mut self, work: usize) -> BuiltinResult<()> {
        let (Some(runtime), Some(budget)) = (self.runtime, self.budget.as_mut()) else {
            return Ok(());
        };
        budget
            .checkpoint(u64::try_from(work).unwrap_or(u64::MAX))
            .await
            .map_err(|source| execution_error(runtime, self.builtin, source))
    }
}

impl StagedWriteCooperation for StagedCpuCooperation<'_> {
    async fn completed(&mut self, work: usize) -> BuiltinResult<()> {
        StagedCpuCooperation::completed(self, work).await
    }
}

pub(crate) fn execution_error(
    runtime: &RuntimeContext,
    builtin: &'static str,
    source: ExecutionServiceError,
) -> RuntimeError {
    match source {
        cancelled @ ExecutionServiceError::Cancelled => {
            build_runtime_error(format!("{builtin}: execution was cancelled"))
                .with_builtin(builtin)
                .with_identifier(format!("{}:Cancelled", runtime.error_namespace()))
                .with_source(cancelled)
                .build()
        }
        other => other.into_runtime_error(),
    }
}

pub(crate) async fn await_cancellable<T>(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    work: impl Future<Output = T>,
) -> BuiltinResult<T> {
    let Some(runtime) = runtime else {
        return Ok(work.await);
    };
    runtime
        .cpu_cooperation()
        .await_cancellable(work)
        .await
        .map_err(|source| execution_error(runtime, builtin, source))
}

pub(crate) async fn begin_stage<C, Check, Begin>(
    cooperation: &mut C,
    check: Check,
    begin: Begin,
    unsupported: &'static str,
) -> Result<Box<dyn FsStagedOutput>, StagedFileError>
where
    C: StagedWriteCooperation,
    Check: Fn() -> io::Result<()>,
    Begin: FnOnce() -> io::Result<Option<Box<dyn FsStagedOutput>>>,
{
    cooperation
        .completed(0)
        .await
        .map_err(StagedFileError::Control)?;
    check().map_err(StagedFileError::Io)?;
    let stage = begin().map_err(StagedFileError::Io)?.ok_or_else(|| {
        StagedFileError::Io(io::Error::new(io::ErrorKind::Unsupported, unsupported))
    })?;
    check().map_err(StagedFileError::Io)?;
    Ok(stage)
}

pub(crate) async fn write_stage_bytes<C, Check>(
    runtime: Option<&RuntimeContext>,
    cooperation: &mut C,
    builtin: &'static str,
    check: Check,
    stage: &mut dyn FsStagedOutput,
    bytes: &[u8],
    messages: StagedWriteMessages,
) -> Result<(), StagedFileError>
where
    C: StagedWriteCooperation,
    Check: Fn() -> io::Result<()>,
{
    for chunk in bytes.chunks(4096) {
        let mut remaining = chunk;
        while !remaining.is_empty() {
            cooperation
                .completed(remaining.len())
                .await
                .map_err(StagedFileError::Control)?;
            check().map_err(StagedFileError::Io)?;
            let written = await_cancellable(runtime, builtin, stage.write_chunk(remaining))
                .await
                .map_err(StagedFileError::Control)?;
            match written {
                Ok(0) => {
                    return Err(StagedFileError::Io(io::Error::new(
                        io::ErrorKind::WriteZero,
                        messages.no_progress,
                    )))
                }
                Ok(n) if n > remaining.len() => {
                    return Err(StagedFileError::Io(io::Error::new(
                        io::ErrorKind::InvalidData,
                        messages.excess_progress,
                    )))
                }
                Ok(n) => remaining = &remaining[n..],
                Err(source) if source.kind() == io::ErrorKind::Interrupted => continue,
                Err(source) => return Err(StagedFileError::Io(source)),
            }
        }
    }
    Ok(())
}

pub(crate) async fn finish_stage<C, Check>(
    runtime: Option<&RuntimeContext>,
    cooperation: &mut C,
    builtin: &'static str,
    check: Check,
    mut stage: Box<dyn FsStagedOutput>,
) -> Result<(), StagedFileError>
where
    C: StagedWriteCooperation,
    Check: Fn() -> io::Result<()>,
{
    cooperation
        .completed(0)
        .await
        .map_err(StagedFileError::Control)?;
    check().map_err(StagedFileError::Io)?;
    await_cancellable(runtime, builtin, stage.flush_and_sync())
        .await
        .map_err(StagedFileError::Control)?
        .map_err(StagedFileError::Io)?;
    cooperation
        .completed(0)
        .await
        .map_err(StagedFileError::Control)?;
    check().map_err(StagedFileError::Io)?;
    publish(runtime, builtin, stage).await?;
    check().map_err(|source| {
        StagedFileError::Publication(FsPublicationFailure::new(
            FsPublicationDisposition::Published,
            source,
        ))
    })
}

pub(crate) async fn publish(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    stage: Box<dyn FsStagedOutput>,
) -> Result<(), StagedFileError> {
    let polled = Cell::new(false);
    let completed = RefCell::new(None);
    let mut publication = stage.publish();
    let result = await_cancellable(
        runtime,
        builtin,
        poll_fn(|context| {
            polled.set(true);
            match publication.as_mut().poll(context) {
                std::task::Poll::Ready(outcome) => {
                    *completed.borrow_mut() = Some(outcome);
                    std::task::Poll::Ready(())
                }
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        }),
    )
    .await;
    match (result, completed.into_inner()) {
        (_, Some(Err(failure))) => Err(StagedFileError::Publication(failure)),
        (Err(error), Some(Ok(()))) => Err(StagedFileError::PublicationInterrupted(
            error,
            FsPublicationDisposition::Published,
        )),
        (Err(error), None) => Err(StagedFileError::PublicationInterrupted(
            error,
            if polled.get() {
                FsPublicationDisposition::Indeterminate
            } else {
                FsPublicationDisposition::NotPublished
            },
        )),
        (Ok(()), Some(Ok(()))) => Ok(()),
        (Ok(()), None) => unreachable!("publication future completed without an outcome"),
    }
}

/// Preserve a cancelled operation's diagnostic while retaining the typed
/// publication state for callers deciding whether replay is safe.
pub(crate) fn interruption_error(
    mut error: RuntimeError,
    disposition: FsPublicationDisposition,
) -> RuntimeError {
    let source = error.source.take();
    error.source = Some(Box::new(PublicationInterruption {
        disposition,
        source,
    }));
    error
}

pub(crate) fn error_disposition(error: &RuntimeError) -> Option<FsPublicationDisposition> {
    let mut current: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(failure) = current.downcast_ref::<FsPublicationFailure>() {
            return Some(failure.disposition());
        }
        if let Some(interruption) = current.downcast_ref::<PublicationInterruption>() {
            return Some(interruption.disposition());
        }
        current = current.source()?;
    }
}

#[derive(Debug)]
pub(crate) struct PublicationInterruption {
    disposition: FsPublicationDisposition,
    source: Option<Box<runmat_async::runtime_error::RuntimeErrorSource>>,
}

impl PublicationInterruption {
    pub(crate) fn disposition(&self) -> FsPublicationDisposition {
        self.disposition
    }
}

impl std::fmt::Display for PublicationInterruption {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "staged publication outcome is {:?}",
            self.disposition
        )
    }
}

impl std::error::Error for PublicationInterruption {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source.as_ref() as &(dyn std::error::Error + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::executor::block_on;
    use runmat_execution::cancellation::CancellationSource;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};

    struct ReadyCooperation;
    impl StagedWriteCooperation for ReadyCooperation {
        async fn completed(&mut self, _work: usize) -> BuiltinResult<()> {
            Ok(())
        }
    }

    struct ScriptedStage {
        outcomes: Arc<Mutex<Vec<io::Result<usize>>>>,
        written: Arc<Mutex<Vec<u8>>>,
    }

    #[async_trait(?Send)]
    impl FsStagedOutput for ScriptedStage {
        async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let outcome = self.outcomes.lock().unwrap().remove(0)?;
            if outcome <= bytes.len() {
                self.written
                    .lock()
                    .unwrap()
                    .extend_from_slice(&bytes[..outcome]);
            }
            Ok(outcome)
        }
        async fn flush_and_sync(&mut self) -> io::Result<()> {
            Ok(())
        }
        async fn publish(self: Box<Self>) -> Result<(), FsPublicationFailure> {
            Ok(())
        }
    }

    #[test]
    fn partial_interrupted_writes_recover_without_duplication() {
        let written = Arc::new(Mutex::new(Vec::new()));
        let mut stage = ScriptedStage {
            outcomes: Arc::new(Mutex::new(vec![
                Err(io::Error::new(io::ErrorKind::Interrupted, "signal")),
                Ok(2),
                Ok(3),
            ])),
            written: written.clone(),
        };
        block_on(write_stage_bytes(
            None,
            &mut ReadyCooperation,
            "print",
            || Ok(()),
            &mut stage,
            b"hello",
            StagedWriteMessages {
                no_progress: "zero",
                excess_progress: "excess",
            },
        ))
        .expect("interrupted partial write resumes");
        assert_eq!(&*written.lock().unwrap(), b"hello");
        assert!(stage.outcomes.lock().unwrap().is_empty());
    }

    #[test]
    fn zero_and_excess_progress_fail_before_publication() {
        for outcome in [0, 4] {
            let mut stage = ScriptedStage {
                outcomes: Arc::new(Mutex::new(vec![Ok(outcome)])),
                written: Arc::new(Mutex::new(Vec::new())),
            };
            let error = block_on(write_stage_bytes(
                None,
                &mut ReadyCooperation,
                "print",
                || Ok(()),
                &mut stage,
                b"abc",
                StagedWriteMessages {
                    no_progress: "zero",
                    excess_progress: "excess",
                },
            ))
            .expect_err("invalid progress");
            assert!(matches!(error, StagedFileError::Io(_)));
        }
    }

    #[test]
    fn pre_cancelled_stage_does_not_invoke_provider() {
        let source = CancellationSource::new();
        source.cancel(runmat_execution::CancellationReason::User);
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(super::super::RuntimeExecutionService::new()),
            source.token(),
        );
        let mut cooperation = StagedCpuCooperation::new(Some(&runtime), "print");
        let mut invoked = false;
        let result = block_on(begin_stage(
            &mut cooperation,
            || Ok(()),
            || {
                invoked = true;
                Ok(None)
            },
            "unsupported",
        ));
        assert!(matches!(result, Err(StagedFileError::Control(_))));
        assert!(!invoked);
    }
}
