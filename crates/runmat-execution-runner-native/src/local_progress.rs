//! Live progress of a local FEA run, as seen by the host.
//!
//! The host's observer travels with [`crate::NativeExecutionConfig`] in a
//! [`ProgressObserverSlot`]. A local study wraps it in a [`ProgressRelay`] (rate limits and
//! heartbeats). Worker processes send their own events as progress frames; the parent decodes
//! them here and forwards them to the same relay:
//!
//! - analysis workers send FEA events (assembly counts, solver iterations, factorization);
//! - meshing workers send work counts, which [`MeshingProgressTracker`] turns into `MeshPrep`
//!   events with model-wide totals.
use std::collections::BTreeMap;

use runmat_analysis_fea::progress::rate::UNIT_METADATA;
use runmat_analysis_fea::progress::ProgressRelay;
use runmat_analysis_fea::{FeaProgressEvent, FeaProgressPhase, FeaProgressStatus};
use runmat_meshing_core::work_count::MeshingWorkUnit;
use runmat_meshing_core::MeshingStageKind;

use crate::protocol::ProgramProgress;
use crate::LocalFeaProgressObserver;

/// The host's progress observer, carried by the execution configuration. It never takes part in
/// configuration equality or serialization.
#[derive(Clone, Default)]
pub struct ProgressObserverSlot(Option<LocalFeaProgressObserver>);

impl ProgressObserverSlot {
    pub fn new(observer: Option<LocalFeaProgressObserver>) -> Self {
        Self(observer)
    }

    pub fn get(&self) -> Option<&LocalFeaProgressObserver> {
        self.0.as_ref()
    }

    pub(crate) fn emit(&self, event: FeaProgressEvent) {
        if let Some(observer) = &self.0 {
            observer(event);
        }
    }
}

impl std::fmt::Debug for ProgressObserverSlot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("ProgressObserverSlot")
            .field(&self.0.is_some())
            .finish()
    }
}

impl PartialEq for ProgressObserverSlot {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for ProgressObserverSlot {}

/// Wraps `observer` in a rate-limiting relay. Native hosts also get heartbeats while the
/// returned guard lives.
pub(crate) fn relay_observer(
    observer: Option<LocalFeaProgressObserver>,
) -> (
    Option<LocalFeaProgressObserver>,
    Option<runmat_analysis_fea::progress::HeartbeatGuard>,
) {
    match observer {
        Some(observer) => {
            let relay = ProgressRelay::new(observer);
            let heartbeat = relay.spawn_heartbeat();
            (Some(relay.handler()), Some(heartbeat))
        }
        None => (None, None),
    }
}

/// Forwards the live counts among an analysis worker's progress frames. The parent owns the
/// run's phase boundaries, so a worker's own `Started` and `Completed` events stay in its log.
pub(crate) fn forward_analysis_progress(slot: &ProgressObserverSlot, records: Vec<ProgramProgress>) {
    if slot.get().is_none() {
        return;
    }
    for event in records
        .iter()
        .filter_map(crate::analysis_progress::decode)
        .filter(|event| event.status == FeaProgressStatus::Advanced)
    {
        slot.emit(event);
    }
}

const MESHING_OPERATION: &str = "fea.meshing";

/// Turns meshing stages and worker work counts into `MeshPrep` events.
#[derive(Default)]
pub(crate) struct MeshingProgressTracker {
    slot: ProgressObserverSlot,
    /// Per surface partition: faces meshed and faces admitted.
    surface: BTreeMap<u32, (u64, u64)>,
    stages_started: u64,
}

impl MeshingProgressTracker {
    pub(crate) fn new(slot: ProgressObserverSlot) -> Self {
        Self {
            slot,
            ..Self::default()
        }
    }

    /// Registers a stage before it runs and reports what it does.
    pub(crate) fn stage_started(&mut self, stage: MeshingStageKind, partition: PartitionFaces) {
        if self.slot.get().is_none() {
            return;
        }
        if stage == MeshingStageKind::SurfaceMesh {
            if let Some(faces) = partition.faces {
                self.surface.entry(partition.index).or_insert((0, faces));
            }
        }
        self.stages_started += 1;
        let mut metadata = stage_metadata(stage);
        metadata.insert(UNIT_METADATA.into(), "stages".into());
        self.slot.emit(event(
            stage_message(stage).into(),
            Some(self.stages_started),
            None,
            metadata,
        ));
    }

    /// Marks a finished stage; a finished surface partition counts all of its faces.
    pub(crate) fn stage_completed(&mut self, stage: MeshingStageKind, partition: PartitionFaces) {
        if stage != MeshingStageKind::SurfaceMesh || partition.faces.is_none() {
            return;
        }
        if let Some((done, total)) = self.surface.get_mut(&partition.index) {
            *done = *total;
        }
        self.emit_surface();
    }

    pub(crate) fn record(
        &mut self,
        stage: MeshingStageKind,
        partition: PartitionFaces,
        records: &[ProgramProgress],
    ) {
        if self.slot.get().is_none() {
            return;
        }
        for count in records.iter().filter_map(crate::meshing_host::decode_work_count) {
            match count.unit {
                MeshingWorkUnit::SurfaceFaces => {
                    let entry = self
                        .surface
                        .entry(partition.index)
                        .or_insert((0, count.total.unwrap_or(count.current)));
                    entry.0 = entry.0.max(count.current);
                    entry.1 = entry.1.max(count.total.unwrap_or(entry.0));
                    self.emit_surface();
                }
                MeshingWorkUnit::VolumePoints => {
                    let mut metadata = stage_metadata(stage);
                    metadata.insert(UNIT_METADATA.into(), "volume_points".into());
                    self.slot.emit(event(
                        format!("Inserted {} volume mesh points", count.current),
                        Some(count.current),
                        None,
                        metadata,
                    ));
                }
                MeshingWorkUnit::QuadraticEdges => {
                    let mut metadata = stage_metadata(stage);
                    metadata.insert(UNIT_METADATA.into(), "quadratic_edges".into());
                    let total = count.total.unwrap_or(count.current);
                    self.slot.emit(event(
                        format!("Placed midside nodes on {} of {total} edges", count.current),
                        Some(count.current),
                        Some(total),
                        metadata,
                    ));
                }
            }
        }
    }

    fn emit_surface(&self) {
        let (done, total) = self
            .surface
            .values()
            .fold((0, 0), |(done, total), (d, t)| (done + d, total + t));
        if total == 0 {
            return;
        }
        let mut metadata = stage_metadata(MeshingStageKind::SurfaceMesh);
        metadata.insert(UNIT_METADATA.into(), "surface_faces".into());
        self.slot.emit(event(
            format!("Meshed {done} of {total} faces"),
            Some(done),
            Some(total),
            metadata,
        ));
    }
}

/// A partition's index and, for a face partition, the faces it admits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PartitionFaces {
    pub index: u32,
    pub faces: Option<u64>,
}

impl PartitionFaces {
    pub(crate) fn of(host: &runmat_meshing_execution::MeshingHostWorkload) -> Self {
        let partition = &host.workload.partition;
        let batch = partition.kind == runmat_meshing_core::MeshingPartitionKind::CanonicalEntityBatch;
        Self {
            index: partition.partition_index,
            faces: partition
                .entity_range
                .as_ref()
                .filter(|_| batch)
                .map(|range| range.entity_count),
        }
    }
}

fn stage_metadata(stage: MeshingStageKind) -> BTreeMap<String, String> {
    let name = serde_json::to_value(stage)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{stage:?}"));
    BTreeMap::from([("stage".to_string(), name)])
}

fn stage_message(stage: MeshingStageKind) -> &'static str {
    match stage {
        MeshingStageKind::GeometryAdmission | MeshingStageKind::Healing => {
            "Checking the meshing geometry"
        }
        MeshingStageKind::Sizing => "Computing the mesh size field",
        MeshingStageKind::CurveMesh => "Meshing edges",
        MeshingStageKind::SurfaceMesh => "Meshing faces",
        MeshingStageKind::ProtectedBoundaryComplex
        | MeshingStageKind::Tetrahedralization
        | MeshingStageKind::ConstraintRecovery
        | MeshingStageKind::Refinement
        | MeshingStageKind::Optimization => "Meshing the volume",
        MeshingStageKind::OrderElevation => "Placing quadratic midside nodes",
        MeshingStageKind::Validation => "Validating the mesh",
        MeshingStageKind::Serialization | MeshingStageKind::Publication => {
            "Publishing the mesh"
        }
    }
}

fn event(
    message: String,
    current: Option<u64>,
    total: Option<u64>,
    metadata: BTreeMap<String, String>,
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
        metadata,
    }
}

#[cfg(test)]
#[path = "local_progress_tests.rs"]
mod tests;
