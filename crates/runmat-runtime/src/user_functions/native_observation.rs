//! Exact selected-context ownership of read-only native name observation.

use std::future::Future;

use futures::FutureExt;

use super::{InstalledNativeArtifactFact, NativeNameFact, NativeNameObserverFuture};
use crate::execution::RootedResult;
use crate::{context::RuntimeContext, RuntimeError};

impl runmat_gc::Trace for InstalledNativeArtifactFact {
    fn trace(&self, _: &mut dyn runmat_gc::Tracer) {
        // Artifact identity and the closed interop manifest are port-free
        // metadata, containing no managed language Values.
        let Self {
            identity: _,
            interop: _,
        } = self;
    }
}

impl runmat_gc::Trace for NativeNameFact {
    fn trace(&self, tracer: &mut dyn runmat_gc::Tracer) {
        let Self {
            path: _,
            origin: _,
            installed: _,
            installed_artifact,
        } = self;
        // Path/origin are owned lookup metadata, not GC payloads. Exhaustive
        // destructuring makes additions to this completion an explicit review.
        if let Some(artifact) = installed_artifact {
            runmat_gc::Trace::trace(artifact, tracer);
        }
    }
}

/// Admit the actual concrete observation body before its original single Box.
/// None is a genuine name miss; allocation/cancellation/error is terminal.
pub fn admit_native_name_observation_with<F>(
    runtime: &RuntimeContext,
    make_future: impl FnOnce() -> F,
) -> Result<NativeNameObserverFuture, RuntimeError>
where
    F: Future<Output = Result<Option<NativeNameFact>, RuntimeError>> + 'static,
{
    crate::execution::admit_rooted_result_with(runtime, move || {
        make_future().map(|result| {
            result.and_then(|fact| {
                RootedResult::new(fact).map_err(|error| error.into_runtime_error())
            })
        })
    })
}

/// Consume the same canonical completion without substituting a miss for an
/// outer admission error or inner observation failure.
pub async fn await_native_name_observation(
    future: Result<NativeNameObserverFuture, RuntimeError>,
) -> Result<Option<NativeNameFact>, RuntimeError> {
    future?.await?.with_owned(Ok)
}
