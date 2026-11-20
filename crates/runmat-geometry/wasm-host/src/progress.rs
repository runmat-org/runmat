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
use runmat_analysis_fea::FeaProgressEvent;
use runmat_meshing_core::work_count::{
    install_work_observer, MeshingWorkCount, MeshingWorkUnit, WorkObserverGuard,
};

struct BridgeState {
    map: MeshingProgressMap,
    sent_at: BTreeMap<MeshingWorkUnit, Instant>,
    /// Each unit's latest count that the spacing held back. The bridge sends it when the
    /// request ends, so the page's last count for a unit is the true final one.
    held: BTreeMap<MeshingWorkUnit, FeaProgressEvent>,
}

/// Forwards this module's meshing work counts while it lives.
pub struct MeshingProgressBridge {
    observer: Option<WorkObserverGuard>,
    state: Arc<Mutex<BridgeState>>,
    page: FeaProgressHandler,
}

impl Drop for MeshingProgressBridge {
    fn drop(&mut self) {
        // The observer goes first, so no count arrives after the held ones are sent.
        drop(self.observer.take());
        let held = std::mem::take(
            &mut self
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .held,
        );
        // The stage has ended: its last counts go to the page directly, past the relay's rate
        // limits, which would drop a count that arrives within the minimum interval.
        for event in held.into_values() {
            (self.page)(event);
        }
    }
}

impl MeshingProgressBridge {
    /// Sends work counts to `page` until the bridge drops.
    pub fn install(page: FeaProgressHandler) -> Self {
        let relay = ProgressRelay::new(Arc::clone(&page));
        let state = Arc::new(Mutex::new(BridgeState {
            map: MeshingProgressMap::new(),
            sent_at: BTreeMap::new(),
            held: BTreeMap::new(),
        }));
        let shared = Arc::clone(&state);
        let observer = Arc::new(move |count: MeshingWorkCount| {
            let mut state = shared.lock().unwrap_or_else(|error| error.into_inner());
            // The map sees every count, so its face totals stay whole when an event is skipped.
            let Some(event) = state.map.work_count(None, None, &count) else {
                return;
            };
            // Space counts here so a model with many small partitions stays within the relay's
            // rate. A full count (a partition's last face, the last edge) always goes: the relay
            // passes it, and skipping it would leave the page short of the true total.
            let now = Instant::now();
            let full = event.current.is_some() && event.current == event.total;
            if !full
                && state
                    .sent_at
                    .get(&count.unit)
                    .is_some_and(|at| now.saturating_duration_since(*at) < MINIMUM_EVENT_INTERVAL)
            {
                state.held.insert(count.unit, event);
                return;
            }
            state.held.remove(&count.unit);
            state.sent_at.insert(count.unit, now);
            relay.emit(event);
        });
        Self {
            observer: Some(install_work_observer(observer)),
            state,
            page,
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::work_count::{add_volume_points, report_work};

    /// Counts that arrive faster than the spacing still end on the true final counts: each
    /// partition's full face count, and the last volume count when the request ends.
    #[test]
    fn fast_counts_end_on_the_final_ones() {
        let events = Arc::new(Mutex::new(Vec::<FeaProgressEvent>::new()));
        let sink = Arc::clone(&events);
        let bridge =
            MeshingProgressBridge::install(Arc::new(move |event| sink.lock().unwrap().push(event)));
        for face in 1..=8 {
            report_work(MeshingWorkUnit::SurfaceFaces, face, Some(8));
        }
        for face in 1..=6 {
            report_work(MeshingWorkUnit::SurfaceFaces, face, Some(6));
        }
        for _ in 0..3 {
            add_volume_points(10);
        }
        drop(bridge);
        let events = events.lock().unwrap().clone();
        let last = |unit: &str| {
            events
                .iter()
                .rev()
                .find(|e| e.metadata.get("unit").map(String::as_str) == Some(unit))
                .map(|e| (e.current, e.total))
        };
        assert_eq!(
            last("surface_faces"),
            Some((Some(14), Some(14))),
            "{events:?}"
        );
        assert_eq!(last("volume_points"), Some((Some(30), None)), "{events:?}");
        // Spacing still applies in between: not every count reached the page.
        assert!(events.len() < 17, "{events:?}");
    }
}
