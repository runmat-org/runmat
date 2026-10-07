//! Invocation-owned console retention and explicit noninteractive policy.
use super::{HostInteraction, RuntimeHostService, RuntimeServiceFuture};
use crate::{build_runtime_error, RuntimeError};
use runmat_execution::console::{ConsoleCapture, ConsoleLimits, ConsoleStream, ConsoleTranscript};
use std::cell::RefCell;

pub struct PortableConsoleHost {
    capture: RefCell<Option<ConsoleCapture>>,
    failure: RefCell<Option<String>>,
}

impl PortableConsoleHost {
    /// The caller admits these limits before creating the capture. This host
    /// does not reserve a task budget or invent a second resource authority.
    pub fn new(limits: ConsoleLimits) -> Result<Self, runmat_execution::ContractError> {
        Ok(Self {
            capture: RefCell::new(Some(ConsoleCapture::new(limits)?)),
            failure: RefCell::new(None),
        })
    }

    /// Seal exactly once and move the original prefix, including failed runs.
    /// Completion describes console capture, not numerical or return success.
    pub fn finish(&self, completed: bool) -> Result<ConsoleTranscript, RuntimeError> {
        self.capture
            .borrow_mut()
            .take()
            .map(|capture| capture.finish(completed))
            .ok_or_else(|| host_error("RunMat:console:Sealed", "console capture already sealed"))
    }
}

impl RuntimeHostService for PortableConsoleHost {
    fn console(&self, stream: ConsoleStream, text: String) {
        let mut capture = self.capture.borrow_mut();
        let result = capture
            .as_mut()
            .ok_or_else(|| "console capture already sealed".to_string())
            .and_then(|capture| {
                capture
                    .push(stream, text)
                    .map_err(|error| error.to_string())
            });
        if let Err(error) = result {
            self.failure.borrow_mut().get_or_insert(error);
        }
    }

    fn owns_console_retention(&self) -> bool {
        true
    }
    fn accepts_queued_interaction(&self) -> bool {
        false
    }

    fn check_execution(&self) -> Result<(), RuntimeError> {
        if let Some(message) = &*self.failure.borrow() {
            return Err(host_error("RunMat:console:CaptureRejected", message));
        }
        if self.capture.borrow().is_none() {
            return Err(host_error(
                "RunMat:console:Sealed",
                "console capture already sealed",
            ));
        }
        Ok(())
    }

    fn interact(
        &self,
        _request: HostInteraction,
    ) -> RuntimeServiceFuture<Result<crate::interaction::InteractionResponse, RuntimeError>> {
        Box::pin(async {
            Err(host_error(
                "RunMat:interaction:NoninteractiveHost",
                "portable execution does not permit interactive input",
            ))
        })
    }

    // Warning builtins own formatting and already write to the console.
    fn warning(&self, _warning: crate::warning_store::RuntimeWarning) {}
}

fn host_error(identifier: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(identifier)
        .build()
}

pub(crate) fn check_active_host() -> Result<(), RuntimeError> {
    if let Some(host) =
        super::legacy::active().and_then(|context| context.service_ports().host().cloned())
    {
        host.check_execution()?;
    }
    Ok(())
}

pub(crate) fn accepts_queued_interaction() -> bool {
    super::legacy::active()
        .and_then(|context| context.service_ports().host().cloned())
        .is_none_or(|host| host.accepts_queued_interaction())
}

#[cfg(test)]
mod tests;
