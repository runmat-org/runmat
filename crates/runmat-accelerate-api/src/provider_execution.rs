//! Execution failures which may retain a provider-local allocation owner.
//!
//! Native sources retain the usual Send + Sync contract. Browser worker sources
//! remain local: forwarding only their display text would discard ownership.
use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;

#[cfg(not(target_arch = "wasm32"))]
type Source = Box<dyn Error + Send + Sync + 'static>;
#[cfg(target_arch = "wasm32")]
type Source = Box<dyn Error + 'static>;

#[derive(Debug)]
pub struct ProviderExecutionError {
    source: ExecutionSource,
    /// Only a certified pre-effect Unsupported response authorizes host replay.
    disposition: FallbackDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FallbackDisposition {
    PreEffect,
    Unproven,
    AfterEffect,
}

#[derive(Debug)]
enum ExecutionSource {
    Shared(anyhow::Error),
    Local(Source),
}

impl ExecutionSource {
    fn as_error(&self) -> &(dyn Error + 'static) {
        match self {
            Self::Shared(error) => error.as_ref(),
            Self::Local(error) => error.as_ref(),
        }
    }
}

pub type ProviderExecutionResult<T> = Result<T, ProviderExecutionError>;
pub type ProviderExecutionFuture<'a, T> =
    Pin<Box<dyn Future<Output = ProviderExecutionResult<T>> + 'a>>;

/// A failed operation whose already-created results were also retired. The
/// original execution cause and every failed-retirement owner remain available
/// independently; cleanup never changes an after-effect result into fallback.
#[derive(Debug)]
pub struct ProviderExecutionCompletionError {
    primary: Option<Box<ProviderExecutionError>>,
    cleanup: Vec<crate::allocation::GpuHandleReleaseError>,
}

impl ProviderExecutionCompletionError {
    pub fn new(
        primary: Option<ProviderExecutionError>,
        cleanup: Vec<crate::allocation::GpuHandleReleaseError>,
    ) -> Self {
        Self {
            primary: primary.map(Box::new),
            cleanup,
        }
    }

    pub fn primary(&self) -> Option<&ProviderExecutionError> {
        self.primary.as_deref()
    }

    pub fn cleanup_errors(&self) -> &[crate::allocation::GpuHandleReleaseError] {
        &self.cleanup
    }
}

impl fmt::Display for ProviderExecutionCompletionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "provider execution had {} retirement failure(s)",
            self.cleanup.len()
        )?;
        if let Some(primary) = &self.primary {
            write!(formatter, "; original execution failure: {primary}")?;
        }
        for cleanup in &self.cleanup {
            write!(formatter, "; {cleanup}")?;
        }
        Ok(())
    }
}

impl Error for ProviderExecutionCompletionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.primary
            .as_ref()
            .map(|primary| primary.as_ref() as &(dyn Error + 'static))
            .or_else(|| {
                self.cleanup
                    .first()
                    .map(|cleanup| cleanup as &(dyn Error + 'static))
            })
    }
}

impl ProviderExecutionError {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_source(error: impl Error + Send + Sync + 'static) -> Self {
        Self {
            source: ExecutionSource::Local(Box::new(error)),
            disposition: FallbackDisposition::PreEffect,
        }
        .inherit_nested_disposition()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_source(error: impl Error + 'static) -> Self {
        Self {
            source: ExecutionSource::Local(Box::new(error)),
            disposition: FallbackDisposition::PreEffect,
        }
        .inherit_nested_disposition()
    }

    /// Keep the original error chain while disabling numerical replay after
    /// any staging, submission, or output-allocation effect.
    pub fn after_effect(mut self) -> Self {
        self.disposition = FallbackDisposition::AfterEffect;
        self
    }

    /// Whether provider effects are known to have preceded this failure.
    /// Unlike a formatted diagnostic, this remains stable across platforms
    /// and retains the original error source for inspection.
    pub fn is_after_effect(&self) -> bool {
        self.disposition == FallbackDisposition::AfterEffect
    }

    /// A legacy hook did not report whether provider effects occurred. Keep
    /// its exact cause, but do not authorize numerical replay on that basis.
    /// An already-proven post-effect failure cannot be weakened to unknown.
    pub fn with_unproven_effects(mut self) -> Self {
        if self.disposition == FallbackDisposition::PreEffect {
            self.disposition = FallbackDisposition::Unproven;
        }
        self
    }

    /// Preserve a stronger retained provider disposition through a semantic
    /// RuntimeError wrapper without replacing its public diagnostic or cause.
    fn inherit_nested_disposition(mut self) -> Self {
        let mut inherited = self.disposition;
        let mut source: &(dyn Error + 'static) = self.source.as_error();
        loop {
            if let Some(provider) = source.downcast_ref::<Self>() {
                inherited = match (inherited, provider.disposition) {
                    (FallbackDisposition::AfterEffect, _)
                    | (_, FallbackDisposition::AfterEffect) => FallbackDisposition::AfterEffect,
                    (FallbackDisposition::Unproven, _) | (_, FallbackDisposition::Unproven) => {
                        FallbackDisposition::Unproven
                    }
                    _ => FallbackDisposition::PreEffect,
                };
            }
            let Some(next) = source.source() else { break };
            source = next;
        }
        self.disposition = inherited;
        self
    }

    /// Only an explicit provider unsupported marker permits numerical fallback.
    pub fn is_unsupported(&self) -> bool {
        is_unsupported_provider_error(self)
    }
}

/// Shared fallback classification for platform-local, anyhow-wrapped, and
/// RuntimeError-wrapped provider causes. Any outer disposition other than
/// proven pre-effect dominates a retained nested Unsupported cause.
pub fn is_unsupported_provider_error(mut error: &(dyn Error + 'static)) -> bool {
    loop {
        if error
            .downcast_ref::<ProviderExecutionError>()
            .is_some_and(|nested| nested.disposition != FallbackDisposition::PreEffect)
        {
            return false;
        }
        if error.is::<crate::UnsupportedProviderOperation>() {
            return true;
        }
        let Some(source) = error.source() else {
            return false;
        };
        error = source;
    }
}

impl From<anyhow::Error> for ProviderExecutionError {
    fn from(error: anyhow::Error) -> Self {
        Self {
            source: ExecutionSource::Shared(error),
            disposition: FallbackDisposition::PreEffect,
        }
        .inherit_nested_disposition()
    }
}

impl From<crate::allocation::AllocationAcquisitionError> for ProviderExecutionError {
    fn from(error: crate::allocation::AllocationAcquisitionError) -> Self {
        Self::from_source(error)
    }
}

impl From<std::io::Error> for ProviderExecutionError {
    fn from(error: std::io::Error) -> Self {
        Self::from_source(error)
    }
}

impl fmt::Display for ProviderExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.source.as_error(), formatter)
    }
}

impl Error for ProviderExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_error())
    }
}

#[cfg(test)]
mod tests;
