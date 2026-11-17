//! Live meshing counts from this host to the page.
//!
//! Meshing kernels report work counts through `runmat_meshing_core::work_count`. While a
//! meshing request runs, [`MeshingProgressBridge::install`] installs the one observer for this
//! module instance. The bridge turns each count into a `MeshPrep` [`FeaProgressEvent`] with the
//! shared [`MeshingProgressMap`], so the units, messages and model-wide face totals match the
//! native host. It sends the events through a [`ProgressRelay`], so the page receives the rate
//! limits of the native host. The browser host draws heartbeats from its own timer, because
//! this module runs synchronously and has no timer of its own.
//!
//! The serial browser executor does not name the surface partition of a count, so the bridge
//! passes `None` and the map infers partitions from restarting counts.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use runmat_analysis_fea::progress::meshing::MeshingProgressMap;
use runmat_analysis_fea::progress::rate::MINIMUM_EVENT_INTERVAL;
use runmat_analysis_fea::progress::{FeaProgressHandler, ProgressRelay};
#[cfg(target_os = "emscripten")]
use runmat_analysis_fea::FeaProgressEvent;
use runmat_meshing_core::work_count::{
    install_work_observer, MeshingWorkCount, MeshingWorkUnit, WorkObserverGuard,
};

struct BridgeState {
    map: MeshingProgressMap,
    sent_at: BTreeMap<MeshingWorkUnit, Instant>,
}

/// Forwards this module's meshing work counts while it lives.
pub struct MeshingProgressBridge {
    _observer: WorkObserverGuard,
}

impl MeshingProgressBridge {
    /// Sends work counts to `page` until the bridge drops.
    pub fn install(page: FeaProgressHandler) -> Self {
        let relay = ProgressRelay::new(page);
        let state = Mutex::new(BridgeState {
            map: MeshingProgressMap::new(),
            sent_at: BTreeMap::new(),
        });
        let observer = Arc::new(move |count: MeshingWorkCount| {
            let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
            // The map sees every count, so its face totals stay whole when an event is skipped.
            let Some(event) = state.map.work_count(None, None, &count) else {
                return;
            };
            // Every partition ends on a full count, which the relay always passes. Space
            // counts here so a model with many small partitions stays within the relay's rate.
            let now = Instant::now();
            if state
                .sent_at
                .get(&count.unit)
                .is_some_and(|at| now.saturating_duration_since(*at) < MINIMUM_EVENT_INTERVAL)
            {
                return;
            }
            state.sent_at.insert(count.unit, now);
            relay.emit(event);
        });
        Self {
            _observer: install_work_observer(observer),
        }
    }
}

/// The page's progress callback, `globalThis.__runmatFeaProgress`, when this module runs in a
/// browser worker. Native builds have no page.
pub(crate) fn page_handler() -> Option<FeaProgressHandler> {
    #[cfg(target_os = "emscripten")]
    {
        Some(Arc::new(|event: FeaProgressEvent| {
            unsafe extern "C" {
                fn emscripten_run_script(script: *const std::ffi::c_char);
            }
            let Ok(payload) = serde_json::to_string(&event) else {
                return;
            };
            // A JSON string literal is a valid JavaScript string literal.
            let Ok(literal) = serde_json::to_string(&payload) else {
                return;
            };
            if let Ok(script) =
                std::ffi::CString::new(format!("globalThis.__runmatFeaProgress?.({literal})"))
            {
                unsafe { emscripten_run_script(script.as_ptr()) };
            }
        }))
    }
    #[cfg(not(target_os = "emscripten"))]
    {
        None
    }
}
