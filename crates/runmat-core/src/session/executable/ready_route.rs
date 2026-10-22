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
        initial: runmat_jit::entry::EntryPublications,
        admission: Option<crate::generic_native::placement::ReadyRouteAdmission>,
    ) -> Result<SelectedReadyRoute, RuntimeError> {
        let Some(admission) = admission.filter(|_| eligible) else {
            return Ok(SelectedReadyRoute {
                published,
                receipt: None,
            });
        };
        let ready = self.generic_native_cache.ready_routes(
            invocation.unit,
            invocation.preferred_function,
            invocation.profile,
            initial,
        )?;
        if ready.iter().all(Option::is_none) {
            return Ok(SelectedReadyRoute {
                published,
                receipt: None,
            });
        }
        let receipt = crate::generic_native::placement::select_ready_route(
            admission, runtime, invocation, &ready,
        )?;
        let Some(selection) = receipt.as_ref() else {
            return Ok(SelectedReadyRoute { published, receipt });
        };
        let Some(index) = selection.native() else {
            return Ok(SelectedReadyRoute {
                published: None,
                receipt,
            });
        };
        let selected = ready
            .into_iter()
            .nth(index)
            .flatten()
            .expect("selection authenticated an admitted ready publication");
        if !self.generic_native_cache.ready_entry_is_current(
            invocation.unit,
            invocation.preferred_function,
            &selected,
        )? {
            return Ok(SelectedReadyRoute {
                published: None,
                receipt: None,
            });
        }
        Ok(SelectedReadyRoute {
            published: Some(selected),
            receipt,
        })
    }
}
