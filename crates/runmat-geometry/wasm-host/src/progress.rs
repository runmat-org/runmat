//! Live meshing counts from this host to the page.
//!
//! Meshing kernels report work counts through `runmat_meshing_core::work_count`. While a
//! meshing request runs, [`MeshingProgressBridge::install`] installs the one observer for this
//! module instance. The bridge turns each count into a `MeshPrep` [`FeaProgressEvent`] with the
//! same units and messages as the native host, and sends it through a [`ProgressRelay`], so
//! the page receives the rate limits of the native host. The browser host draws heartbeats
//! from its own timer, because this module runs synchronously and has no timer of its own.
//!
//! The serial browser executor meshes faces in partitions of a few faces each, and each
//! partition counts from its own start. The bridge adds the partitions up: `current` is the
//! faces meshed so far and `total` the faces of the partitions started so far, which is what
//! the native host reports for its partitions.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use runmat_analysis_fea::progress::rate::{MINIMUM_EVENT_INTERVAL, UNIT_METADATA};
use runmat_analysis_fea::progress::{FeaProgressHandler, ProgressRelay};
use runmat_analysis_fea::{FeaProgressEvent, FeaProgressPhase, FeaProgressStatus};
use runmat_meshing_core::work_count::{
    install_work_observer, MeshingWorkCount, MeshingWorkUnit, WorkObserverGuard,
};
use std::time::Instant;

const MESHING_OPERATION: &str = "fea.meshing";

#[derive(Default)]
struct SurfaceCount {
    /// Faces of partitions that finished, and of all partitions started so far.
    finished: u64,
    started_total: u64,
    /// The open partition's last count.
    open: Option<(u64, u64)>,
}

impl SurfaceCount {
    fn record(&mut self, current: u64, total: u64) -> (u64, u64) {
        if let Some((open_current, open_total)) = self.open {
            let restarted = current < open_current
                || (open_current >= open_total && (current, total) != (open_current, open_total));
            if restarted {
                self.finished += open_total;
            } else {
                self.started_total -= open_total;
            }
        }
        self.started_total += total;
        self.open = Some((current, total));
        (self.finished + current, self.started_total)
    }
}

struct BridgeState {
    surface: SurfaceCount,
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
            surface: SurfaceCount::default(),
            sent_at: BTreeMap::new(),
        });
        let observer = Arc::new(move |count: MeshingWorkCount| {
            let mut state = state.lock().unwrap_or_else(|error| error.into_inner());
            let event = match count.unit {
                MeshingWorkUnit::SurfaceFaces => {
                    let total = count.total.unwrap_or(count.current).max(count.current);
                    let (done, total) = state.surface.record(count.current, total);
                    work_event(
                        "surface_faces",
                        format!("Meshed {done} of {total} faces"),
                        Some(done),
                        Some(total),
                    )
                }
                MeshingWorkUnit::VolumePoints => work_event(
                    "volume_points",
                    format!("Inserted {} volume mesh points", count.current),
                    Some(count.current),
                    None,
                ),
                MeshingWorkUnit::QuadraticEdges => {
                    let total = count.total.unwrap_or(count.current);
                    work_event(
                        "quadratic_edges",
                        format!("Placed midside nodes on {} of {total} edges", count.current),
                        Some(count.current),
                        Some(total),
                    )
                }
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

fn work_event(
    unit: &str,
    message: String,
    current: Option<u64>,
    total: Option<u64>,
) -> FeaProgressEvent {
    let fraction = match (current, total) {
        (Some(current), Some(total)) if total > 0 => Some((current as f64 / total as f64).min(1.0)),
        _ => None,
    };
    FeaProgressEvent {
        operation: MESHING_OPERATION.into(),
        phase: FeaProgressPhase::MeshPrep,
        status: FeaProgressStatus::Advanced,
        message,
        current,
        total,
        fraction,
        metadata: BTreeMap::from([(UNIT_METADATA.to_string(), unit.to_string())]),
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
    use super::SurfaceCount;

    #[test]
    fn surface_counts_add_up_across_partitions() {
        let mut count = SurfaceCount::default();
        assert_eq!(count.record(1, 8), (1, 8));
        assert_eq!(count.record(5, 8), (5, 8));
        assert_eq!(count.record(8, 8), (8, 8));
        // The next partition counts from its own start.
        assert_eq!(count.record(1, 3), (9, 11));
        assert_eq!(count.record(0, 4).0, 11);
        assert_eq!(count.record(0, 4), (11, 15));
        assert_eq!(count.record(4, 4), (15, 15));
    }
}
