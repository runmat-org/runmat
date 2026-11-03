//! Invocation-owned console retention and explicit noninteractive policy.
use super::{HostInteraction, RuntimeHostService, RuntimeServiceFuture};
use crate::{build_runtime_error, RuntimeError};
use runmat_execution::console::{
    ConsoleCapture, ConsoleLimits, ConsoleRecord, ConsoleStream, ConsoleTranscript,
};
use std::cell::RefCell;

pub trait PortableConsoleSink {
    fn emit(&self, record: &ConsoleRecord) -> Result<(), String>;
}

impl super::RuntimeContext {
    /// Invocation admission reuses the host's exact capture owner. It neither
    /// allocates another transcript nor transfers responsibility for sealing it.
    pub fn validate_invocation_console(
        &self,
        requested: Option<ConsoleLimits>,
    ) -> Result<(), RuntimeError> {
        let Some(requested) = requested else {
            return Ok(());
        };
        let host = self.service_ports().host().ok_or_else(|| {
            host_error(
                "RunMat:console:CaptureUnavailable",
                "invocation requires its captured console host",
            )
        })?;
        if !host.owns_console_retention() || host.console_limits() != Some(requested) {
            return Err(host_error(
                "RunMat:console:ControlsMismatch",
                "invocation console host disagrees with captured limits",
            ));
        }
        host.check_execution()
    }
}

pub struct PortableConsoleHost {
    sink: Option<std::rc::Rc<dyn PortableConsoleSink>>,
    limits: ConsoleLimits,
    capture: RefCell<Option<ConsoleCapture>>,
    failure: RefCell<Option<String>>,
    admission: RefCell<Option<std::sync::Arc<dyn super::RuntimeConsoleAdmission>>>,
    admission_identity: RefCell<Option<super::RuntimeConsoleScopeIdentity>>,
    sources: std::sync::Arc<super::retained_console::RetainedConsoleSources>,
}

impl PortableConsoleHost {
    /// The caller admits these limits before creating the capture. This host
    /// does not reserve a task budget or invent a second resource authority.
    pub fn new(limits: ConsoleLimits) -> Result<Self, runmat_execution::ContractError> {
        Self::with_sink(limits, None)
    }

    pub fn with_sink(
        limits: ConsoleLimits,
        sink: Option<std::rc::Rc<dyn PortableConsoleSink>>,
    ) -> Result<Self, runmat_execution::ContractError> {
        Ok(Self {
            sink,
            limits,
            capture: RefCell::new(Some(ConsoleCapture::new(limits)?)),
            failure: RefCell::new(None),
            admission: RefCell::new(None),
            admission_identity: RefCell::new(None),
            sources: std::sync::Arc::default(),
        })
    }

    /// Install the existing authenticated root owner before nested admission.
    /// Reusing that exact handle is idempotent; replacing its authority is not.
    pub fn install_console_admission(
        &self,
        owner: std::sync::Arc<dyn super::RuntimeConsoleAdmission>,
        identity: super::RuntimeConsoleScopeIdentity,
    ) -> Result<(), RuntimeError> {
        self.check_execution()?;
        let mut saved = self.admission.borrow_mut();
        if let Some(original) = saved.as_ref() {
            if !std::sync::Arc::ptr_eq(original, &owner)
                || *self.admission_identity.borrow() != Some(identity)
            {
                return Err(host_error(
                    "RunMat:console:ControlsMismatch",
                    "child console admission owner changed",
                ));
            }
        } else {
            owner
                .reserve(identity, self.limits)
                .map_err(|message| host_error("RunMat:console:CaptureRejected", message))?;
            *saved = Some(owner);
            *self.admission_identity.borrow_mut() = Some(identity);
        }
        Ok(())
    }

    /// Clone only an owner handle. Its visitor borrows original child evidence.
    pub fn retained_console_source(
        &self,
    ) -> std::sync::Arc<dyn super::RuntimeRetainedConsoleSource> {
        self.sources.clone()
    }

    /// Seal exactly once and move the original prefix, including failed runs.
    /// Completion describes console capture, not numerical or return success.
    pub fn finish(&self, completed: bool) -> Result<ConsoleTranscript, RuntimeError> {
        let transcript = self
            .capture
            .borrow_mut()
            .take()
            .map(|capture| capture.finish(completed && self.failure.borrow().is_none()))
            .ok_or_else(|| host_error("RunMat:console:Sealed", "console capture already sealed"))?;
        if let (Some(owner), Some(identity)) = (
            self.admission.borrow().as_ref(),
            *self.admission_identity.borrow(),
        ) {
            owner
                .settle(identity, &transcript)
                .map_err(|message| host_error("RunMat:console:CaptureRejected", message))?;
        }
        Ok(transcript)
    }
}

impl RuntimeHostService for PortableConsoleHost {
    fn console(&self, stream: ConsoleStream, text: crate::console::ConsoleText) {
        if self.failure.borrow().is_some() {
            return;
        }
        let mut capture = self.capture.borrow_mut();
        let result = capture
            .as_mut()
            .ok_or_else(|| "console capture already sealed".to_string())
            .and_then(|capture| {
                capture
                    .check_borrowed_record(stream, text.as_str())
                    .map_err(|error| error.to_string())?;
                if let (Some(owner), Some(identity)) = (
                    self.admission.borrow().as_ref(),
                    *self.admission_identity.borrow(),
                ) {
                    owner.admit_record(identity, stream, text.as_str())?;
                }
                capture
                    .push_borrowed(stream, text.as_str())
                    .map_err(|error| error.to_string())
            });
        let result = result.and_then(|()| {
            if let Some(sink) = &self.sink {
                let record = capture
                    .as_ref()
                    .and_then(|capture| capture.records().last())
                    .ok_or_else(|| "console record was not retained".to_string())?;
                sink.emit(record)?;
            }
            Ok(())
        });
        if let Err(error) = result {
            self.failure.borrow_mut().get_or_insert(error);
        }
    }

    fn owns_console_retention(&self) -> bool {
        true
    }
    fn console_limits(&self) -> Option<ConsoleLimits> {
        Some(self.limits)
    }
    fn console_admission(&self) -> Option<std::sync::Arc<dyn super::RuntimeConsoleAdmission>> {
        self.admission.borrow().clone()
    }
    fn retain_console_source(
        &self,
        source: std::sync::Arc<dyn super::RuntimeRetainedConsoleSource>,
    ) -> Result<(), RuntimeError> {
        self.check_execution()?;
        let max_sources = self
            .admission
            .borrow()
            .as_ref()
            .map(|owner| owner.max_scopes())
            .ok_or_else(|| {
                host_error(
                    "RunMat:console:CaptureUnavailable",
                    "requested child capture has no authenticated root admission owner",
                )
            })?;
        self.sources.retain(source, max_sources).map_err(|message| {
            self.failure
                .borrow_mut()
                .get_or_insert_with(|| message.clone());
            host_error("RunMat:console:CaptureRejected", message)
        })
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
        noninteractive_response()
    }

    // Warning builtins own formatting and already write to the console.
    fn warning(&self, _warning: crate::warning_store::RuntimeWarning) {}
}

/// Explicitly unrequested console: do not build an unbounded runtime buffer or
/// imply a complete transcript. Other noninteractive policy is identical.
pub struct NonCapturingPortableHost;
impl RuntimeHostService for NonCapturingPortableHost {
    fn console(&self, _stream: ConsoleStream, _text: crate::console::ConsoleText) {}
    fn owns_console_retention(&self) -> bool {
        true
    }
    fn accepts_queued_interaction(&self) -> bool {
        false
    }
    fn interact(
        &self,
        _request: HostInteraction,
    ) -> RuntimeServiceFuture<Result<crate::interaction::InteractionResponse, RuntimeError>> {
        noninteractive_response()
    }
    fn warning(&self, _warning: crate::warning_store::RuntimeWarning) {}
}

/// Selected stock native console: output follows the existing admitted
/// console-text and thread-buffer path, including target-test redirection.
/// It owns no duplicate capture or graphical/interactive authority.
pub struct PortableForwardingConsoleHost;

impl RuntimeHostService for PortableForwardingConsoleHost {
    fn console(&self, stream: ConsoleStream, text: crate::console::ConsoleText) {
        crate::console::forward_host_text(stream, text);
    }

    fn accepts_queued_interaction(&self) -> bool {
        false
    }

    fn interact(
        &self,
        _request: HostInteraction,
    ) -> RuntimeServiceFuture<Result<crate::interaction::InteractionResponse, RuntimeError>> {
        noninteractive_response()
    }

    fn warning(&self, _warning: crate::warning_store::RuntimeWarning) {}
}

fn noninteractive_response(
) -> RuntimeServiceFuture<Result<crate::interaction::InteractionResponse, RuntimeError>> {
    Box::pin(async {
        Err(host_error(
            "RunMat:interaction:NoninteractiveHost",
            "portable execution does not permit interactive input",
        ))
    })
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
