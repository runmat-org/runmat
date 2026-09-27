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
    source: Source,
}

pub type ProviderExecutionResult<T> = Result<T, ProviderExecutionError>;
pub type ProviderExecutionFuture<'a, T> =
    Pin<Box<dyn Future<Output = ProviderExecutionResult<T>> + 'a>>;

impl ProviderExecutionError {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_source(error: impl Error + Send + Sync + 'static) -> Self {
        Self {
            source: Box::new(error),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_source(error: impl Error + 'static) -> Self {
        Self {
            source: Box::new(error),
        }
    }

    /// Only an explicit provider unsupported marker permits numerical fallback.
    pub fn is_unsupported(&self) -> bool {
        let mut error: &(dyn Error + 'static) = self.source.as_ref();
        loop {
            if error.is::<crate::UnsupportedProviderOperation>() {
                return true;
            }
            let Some(source) = error.source() else {
                return false;
            };
            error = source;
        }
    }
}

impl From<anyhow::Error> for ProviderExecutionError {
    fn from(error: anyhow::Error) -> Self {
        Self {
            source: error.into_boxed_dyn_error(),
        }
    }
}

impl fmt::Display for ProviderExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.source, formatter)
    }
}

impl Error for ProviderExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}

#[cfg(test)]
mod tests;
