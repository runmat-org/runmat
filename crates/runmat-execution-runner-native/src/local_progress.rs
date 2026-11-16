//! Live progress of a local FEA run, as seen by the host.
//!
//! The host's observer travels with [`crate::NativeExecutionConfig`] in a
//! [`ProgressObserverSlot`]. A local study wraps it in a [`ProgressRelay`] (rate limits and
//! heartbeats). Worker processes send their own events as progress frames; the parent decodes
//! them here and forwards them to the same relay:
//!
//! - analysis workers send FEA events (assembly counts, solver iterations, factorization);
//! - meshing workers send work counts, which [`MeshingProgressTracker`] turns into `MeshPrep`
//!   events with model-wide totals through the shared [`MeshingProgressMap`].
use runmat_analysis_fea::progress::meshing::{MeshingPartitionFaces, MeshingProgressMap};
use runmat_analysis_fea::progress::ProgressRelay;
use runmat_analysis_fea::{FeaProgressEvent, FeaProgressStatus};
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

/// Sends `MeshPrep` events for meshing stages and worker work counts. The mapping itself is
/// [`MeshingProgressMap`], which every meshing host shares.
#[derive(Default)]
pub(crate) struct MeshingProgressTracker {
    slot: ProgressObserverSlot,
    map: MeshingProgressMap,
}

impl MeshingProgressTracker {
    pub(crate) fn new(slot: ProgressObserverSlot) -> Self {
        Self {
            slot,
            map: MeshingProgressMap::new(),
        }
    }

    /// Registers a stage before it runs and reports what it does.
    pub(crate) fn stage_started(&mut self, stage: MeshingStageKind, partition: MeshingPartitionFaces) {
        if self.slot.get().is_none() {
            return;
        }
        self.slot.emit(self.map.stage_started(stage, partition));
    }

    /// Marks a finished stage; a finished surface partition counts all of its faces.
    pub(crate) fn stage_completed(
        &mut self,
        stage: MeshingStageKind,
        partition: MeshingPartitionFaces,
    ) {
        if let Some(event) = self.map.stage_completed(stage, partition) {
            self.slot.emit(event);
        }
    }

    pub(crate) fn record(
        &mut self,
        stage: MeshingStageKind,
        partition: MeshingPartitionFaces,
        records: &[ProgramProgress],
    ) {
        if self.slot.get().is_none() {
            return;
        }
        for count in records.iter().filter_map(crate::meshing_host::decode_work_count) {
            if let Some(event) = self.map.work_count(Some(stage), Some(partition.index), &count) {
                self.slot.emit(event);
            }
        }
    }
}

/// A host workload's partition index and, for a face partition, the faces it admits.
pub(crate) fn partition_faces(
    host: &runmat_meshing_execution::MeshingHostWorkload,
) -> MeshingPartitionFaces {
    let partition = &host.workload.partition;
    let batch = partition.kind == runmat_meshing_core::MeshingPartitionKind::CanonicalEntityBatch;
    MeshingPartitionFaces {
        index: partition.partition_index,
        faces: partition
            .entity_range
            .as_ref()
            .filter(|_| batch)
            .map(|range| range.entity_count),
    }
}

#[cfg(test)]
#[path = "local_progress_tests.rs"]
mod tests;
