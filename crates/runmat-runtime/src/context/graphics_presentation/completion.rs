//! One completion/cancellation boundary for actual graphics host operations.

use super::*;
use crate::context::RuntimeContext;

#[derive(Debug)]
pub(crate) struct GraphicsFlushEffectSource {
    disposition: GraphicsFlushDisposition,
    source: Option<Box<runmat_async::runtime_error::RuntimeErrorSource>>,
}

impl GraphicsFlushEffectSource {
    #[cfg(test)]
    pub(crate) fn disposition(&self) -> GraphicsFlushDisposition {
        self.disposition
    }
}
impl std::fmt::Display for GraphicsFlushEffectSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "graphics flush outcome is {:?}",
            self.disposition
        )
    }
}
impl std::error::Error for GraphicsFlushEffectSource {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source.as_ref() as &(dyn std::error::Error + 'static))
    }
}

pub(crate) fn attach_disposition(
    mut error: RuntimeError,
    disposition: GraphicsFlushDisposition,
) -> RuntimeError {
    let source = error.source.take();
    error.source = Some(Box::new(GraphicsFlushEffectSource {
        disposition,
        source,
    }));
    error
}

pub(crate) async fn await_completion(
    runtime: &RuntimeContext,
    mut operation: Pin<Box<dyn RuntimeGraphicsFlushOperation>>,
    cancelled: impl FnOnce(crate::execution::ExecutionServiceError) -> RuntimeError,
    missing: impl FnOnce() -> RuntimeError,
) -> Result<(GraphicsFlushReceipt, GraphicsFlushDisposition), RuntimeError> {
    let cooperation = runtime.cpu_cooperation();
    let mut terminal = None;
    let race = cooperation
        .await_cancellable(async {
            terminal = Some(operation.as_mut().await);
        })
        .await;
    let disposition = operation.as_mut().abandon();
    let result = match (terminal, race) {
        (Some(result), _) => result,
        (None, Err(source)) => Err(cancelled(source)),
        (None, Ok(())) => Err(missing()),
    };
    result
        .map(|receipt| (receipt, disposition))
        .map_err(|error| attach_disposition(error, disposition))
}
