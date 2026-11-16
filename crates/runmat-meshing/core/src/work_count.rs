//! Live work counts for hosts that show meshing progress.
//!
//! A meshing worker process installs one observer with [`install_work_observer`]. Kernels then
//! report counts at natural batch points: surface meshing per face, volume refinement per
//! accepted batch of points, quadratic elevation every [`QUADRATIC_EDGE_REPORT_INTERVAL`]
//! edges. Without an observer a report costs one relaxed atomic load. Counts never feed back
//! into meshing, so they cannot change a mesh, its evidence or its digest.
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

/// The quantity a [`MeshingWorkCount`] counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingWorkUnit {
    /// Faces of the current surface partition that have a surface mesh.
    SurfaceFaces,
    /// Points inserted into the volume mesh by this process; there is no fixed total.
    VolumePoints,
    /// Mesh edges that received a quadratic (tet10) midside node.
    QuadraticEdges,
}

/// One absolute count. `total` is absent when the work has no known end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshingWorkCount {
    pub unit: MeshingWorkUnit,
    pub current: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
}

pub type MeshingWorkObserver = Arc<dyn Fn(MeshingWorkCount) + Send + Sync + 'static>;

/// Quadratic elevation reports after this many edges.
pub const QUADRATIC_EDGE_REPORT_INTERVAL: u64 = 1024;

static INSTALLED: AtomicBool = AtomicBool::new(false);
static OBSERVER: RwLock<Option<MeshingWorkObserver>> = RwLock::new(None);
static VOLUME_POINTS: AtomicU64 = AtomicU64::new(0);

/// Removes the observer when dropped.
pub struct WorkObserverGuard(());

impl Drop for WorkObserverGuard {
    fn drop(&mut self) {
        INSTALLED.store(false, Ordering::Release);
        if let Ok(mut slot) = OBSERVER.write() {
            *slot = None;
        }
    }
}

/// Sends this process's work counts to `observer` until the guard drops.
pub fn install_work_observer(observer: MeshingWorkObserver) -> WorkObserverGuard {
    if let Ok(mut slot) = OBSERVER.write() {
        *slot = Some(observer);
    }
    VOLUME_POINTS.store(0, Ordering::Relaxed);
    INSTALLED.store(true, Ordering::Release);
    WorkObserverGuard(())
}

pub fn report_work(unit: MeshingWorkUnit, current: u64, total: Option<u64>) {
    if !INSTALLED.load(Ordering::Relaxed) {
        return;
    }
    let observer = OBSERVER.read().ok().and_then(|slot| slot.clone());
    if let Some(observer) = observer {
        observer(MeshingWorkCount {
            unit,
            current,
            total,
        });
    }
}

/// Adds `count` inserted volume points to this process's running total and reports it.
pub fn add_volume_points(count: u64) {
    if !INSTALLED.load(Ordering::Relaxed) || count == 0 {
        return;
    }
    let total = VOLUME_POINTS.fetch_add(count, Ordering::Relaxed) + count;
    report_work(MeshingWorkUnit::VolumePoints, total, None);
}

#[cfg(test)]
#[path = "work_count_tests.rs"]
mod tests;
