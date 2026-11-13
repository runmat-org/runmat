//! The exact lifecycle lease of a figure created by a worker invocation.
//!
//! This is neither a scene snapshot nor a numeric figure-ID inventory. Scene
//! successors retain this lease; closing and recreating a figure never does.

use super::{close_figure_selected, FigureHandle, FigureState};
use std::rc::Rc;

pub(super) fn claim(state: &mut FigureState, handle: FigureHandle) {
    let Some(runtime) = crate::context::legacy::active() else {
        return;
    };
    if !runtime.state().worker_invocation_resources.get() {
        return;
    }
    let owner = Rc::new(());
    state.invocation_owner = Some(owner.clone());
    runtime.register_session_resource_cleanup(move || {
        // The predicate and removal share the existing registry lock. A
        // replacement with the same displayed ID cannot be retired by us.
        let _ = close_figure_selected(Some(handle), |state| {
            state
                .invocation_owner
                .as_ref()
                .is_some_and(|current| Rc::ptr_eq(current, &owner))
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RuntimeContext, RuntimeContextGuard};
    use crate::execution::RuntimeExecutionService;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn worker_cleanup_preserves_other_figures_and_fences_numeric_handle_reuse() {
        use super::super::{
            clear_figure, clone_figure, close_figure, new_figure_handle, select_figure,
        };

        let ordinary = new_figure_handle();
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        runtime.enable_worker_invocation_resources();
        let probe = runtime.worker_invocation_retirement_check();
        let (owned, reused) = {
            let _selected = RuntimeContextGuard::enter(runtime.clone());
            // Clearing an ordinary UI figure cannot claim it for this worker.
            clear_figure(Some(ordinary)).expect("ordinary clear");
            let owned = new_figure_handle();
            clear_figure(Some(owned)).expect("owned successor");
            let reused = new_figure_handle();
            close_figure(Some(reused)).expect("close owned generation");
            (owned, reused)
        };
        select_figure(reused); // another lifecycle, same numeric handle
        runtime.retire_worker_invocation_resources();
        assert!(clone_figure(owned).is_none());
        assert!(clone_figure(ordinary).is_some());
        assert!(clone_figure(reused).is_some());
        assert!(!probe(), "live runtime is not a retirement proof");
        let retained = runtime.clone();
        drop(runtime);
        assert!(!probe(), "an escaped context must remain unverified");
        drop(retained);
        RuntimeContext::collect_retired_worker_values().expect("collect retired roots");
        assert!(probe());
        close_figure(Some(ordinary)).expect("close ordinary test figure");
        close_figure(Some(reused)).expect("close replacement test figure");
    }
}
