use runmat_jit::entry::PublishedEntry;
use runmat_runtime::{context::RuntimeContext, RuntimeError};

use crate::generic_native::placement::{ReadyInvocation, ReadyRouteSelection};

use super::RunMatSession;

pub(super) struct SelectedReadyRoute {
    pub published: Option<PublishedEntry>,
    pub receipt: Option<ReadyRouteSelection>,
}

impl RunMatSession {
    /// The placement service cannot pin machine code or its dependencies.
    /// Authenticate the selected publication again before native arguments
    /// move or interpreter effects begin. A stale native decision falls back
    /// to the original interpreter without publishing a false observation.
    pub(super) fn select_ready_native_route(
        &mut self,
        runtime: &RuntimeContext,
        invocation: ReadyInvocation<'_>,
        published: Option<PublishedEntry>,
        eligible: bool,
        admission: Option<crate::generic_native::placement::ReadyRouteAdmission>,
    ) -> Result<SelectedReadyRoute, RuntimeError> {
        let Some((ready, admission)) = published.as_ref().filter(|_| eligible).zip(admission)
        else {
            return Ok(SelectedReadyRoute {
                published,
                receipt: None,
            });
        };
        let receipt = crate::generic_native::placement::select_ready_route(
            admission, runtime, invocation, ready,
        )?;
        if receipt
            .as_ref()
            .is_some_and(|selection| !selection.native())
        {
            return Ok(SelectedReadyRoute {
                published: None,
                receipt,
            });
        }
        if receipt.is_some()
            && !self.generic_native_cache.ready_entry_is_current(
                invocation.unit,
                invocation.preferred_function,
                ready,
            )?
        {
            return Ok(SelectedReadyRoute {
                published: None,
                receipt: None,
            });
        }
        Ok(SelectedReadyRoute { published, receipt })
    }
}
