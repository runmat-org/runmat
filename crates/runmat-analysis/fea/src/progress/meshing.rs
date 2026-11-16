//! Meshing progress as FEA progress events.
//!
//! Meshing kernels report work counts (`runmat_meshing_core::work_count`) and hosts see the
//! stages they run. [`MeshingProgressMap`] turns both into `MeshPrep` [`FeaProgressEvent`]s
//! with model-wide totals. Every host that meshes uses it, so the units, messages and metadata
//! of a meshing run do not depend on where it runs:
//!
//! | count | `unit` | message |
//! |---|---|---|
//! | stage started | `stages` | what the stage does, such as "Meshing faces" |
//! | surface faces | `surface_faces` | "Meshed 12 of 40 faces" (all partitions added up) |
//! | volume points | `volume_points` | "Inserted 4096 volume mesh points" |
//! | quadratic edges | `quadratic_edges` | "Placed midside nodes on 2048 of 8192 edges" |
//!
//! Surface faces are meshed in partitions that each count from their own start. A host that
//! knows the partition of a count passes its index; a host that does not (the serial browser
//! executor) passes `None`, and the map starts a new partition when a count restarts or a
//! finished partition's count changes. A finished partition counts all of its faces.
//!
//! The map only builds events. The host sends them, usually through a
//! [`super::ProgressRelay`], which applies the rate limits.
use std::collections::BTreeMap;

use runmat_meshing_core::work_count::{MeshingWorkCount, MeshingWorkUnit};
use runmat_meshing_core::MeshingStageKind;

use super::rate::UNIT_METADATA;
use super::{FeaProgressEvent, FeaProgressPhase, FeaProgressStatus};

/// The `operation` of every meshing progress event.
pub const MESHING_OPERATION: &str = "fea.meshing";
/// Metadata key with the meshing stage of an event, in its serialized form.
pub const STAGE_METADATA: &str = "stage";

/// A partition's index and, for a face partition, the faces it admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshingPartitionFaces {
    pub index: u32,
    pub faces: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SurfacePartition {
    /// A partition the host named.
    Known(u32),
    /// A partition the map inferred from restarting counts.
    Inferred(u32),
}

/// Builds `MeshPrep` events from meshing stages and work counts (see the module documentation).
#[derive(Debug, Default)]
pub struct MeshingProgressMap {
    /// Per surface partition: faces meshed and faces admitted.
    surface: BTreeMap<SurfacePartition, (u64, u64)>,
    /// The inferred partition that received the last count without a known partition.
    open_inferred: Option<u32>,
    stages_started: u64,
}

impl MeshingProgressMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a stage before it runs: the event says what the stage does and counts the
    /// stages started so far.
    pub fn stage_started(
        &mut self,
        stage: MeshingStageKind,
        partition: MeshingPartitionFaces,
    ) -> FeaProgressEvent {
        if stage == MeshingStageKind::SurfaceMesh {
            if let Some(faces) = partition.faces {
                self.surface
                    .entry(SurfacePartition::Known(partition.index))
                    .or_insert((0, faces));
            }
        }
        self.stages_started += 1;
        event(
            Some(stage),
            "stages",
            stage_message(stage).into(),
            Some(self.stages_started),
            None,
        )
    }

    /// Marks a finished stage. A finished surface partition counts all of its faces; other
    /// stages report nothing.
    pub fn stage_completed(
        &mut self,
        stage: MeshingStageKind,
        partition: MeshingPartitionFaces,
    ) -> Option<FeaProgressEvent> {
        if stage != MeshingStageKind::SurfaceMesh || partition.faces.is_none() {
            return None;
        }
        if let Some((done, total)) = self
            .surface
            .get_mut(&SurfacePartition::Known(partition.index))
        {
            *done = *total;
        }
        self.surface_event()
    }

    /// One work count. `stage` is the running stage when the host knows it; `partition` is the
    /// surface partition the count belongs to, or `None` to infer it from the counts.
    pub fn work_count(
        &mut self,
        stage: Option<MeshingStageKind>,
        partition: Option<u32>,
        count: &MeshingWorkCount,
    ) -> Option<FeaProgressEvent> {
        match count.unit {
            MeshingWorkUnit::SurfaceFaces => {
                let total = count.total.unwrap_or(count.current).max(count.current);
                let key = match partition {
                    Some(index) => SurfacePartition::Known(index),
                    None => self.inferred_partition(count.current, total),
                };
                let entry = self.surface.entry(key).or_insert((0, total));
                entry.0 = entry.0.max(count.current);
                entry.1 = entry.1.max(total);
                self.surface_event()
            }
            MeshingWorkUnit::VolumePoints => Some(event(
                stage,
                "volume_points",
                format!("Inserted {} volume mesh points", count.current),
                Some(count.current),
                None,
            )),
            MeshingWorkUnit::QuadraticEdges => {
                let total = count.total.unwrap_or(count.current);
                Some(event(
                    stage,
                    "quadratic_edges",
                    format!("Placed midside nodes on {} of {total} edges", count.current),
                    Some(count.current),
                    Some(total),
                ))
            }
        }
    }

    /// The partition of a count without a known partition: the open one, unless the count
    /// restarted or the open partition already finished with another count.
    fn inferred_partition(&mut self, current: u64, total: u64) -> SurfacePartition {
        if let Some(open) = self.open_inferred {
            let key = SurfacePartition::Inferred(open);
            let (done, admitted) = self.surface.get(&key).copied().unwrap_or_default();
            let restarted =
                current < done || (done >= admitted && (current, total) != (done, admitted));
            if !restarted {
                return key;
            }
            if let Some(finished) = self.surface.get_mut(&key) {
                finished.0 = finished.1;
            }
        }
        let next = self.open_inferred.map_or(0, |open| open + 1);
        self.open_inferred = Some(next);
        SurfacePartition::Inferred(next)
    }

    fn surface_event(&self) -> Option<FeaProgressEvent> {
        let (done, total) = self
            .surface
            .values()
            .fold((0, 0), |(done, total), (d, t)| (done + d, total + t));
        (total > 0).then(|| {
            event(
                Some(MeshingStageKind::SurfaceMesh),
                "surface_faces",
                format!("Meshed {done} of {total} faces"),
                Some(done),
                Some(total),
            )
        })
    }
}

/// What a meshing stage does, as a progress message.
pub fn stage_message(stage: MeshingStageKind) -> &'static str {
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
        MeshingStageKind::Serialization | MeshingStageKind::Publication => "Publishing the mesh",
    }
}

fn stage_name(stage: MeshingStageKind) -> String {
    serde_json::to_value(stage)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{stage:?}"))
}

fn event(
    stage: Option<MeshingStageKind>,
    unit: &str,
    message: String,
    current: Option<u64>,
    total: Option<u64>,
) -> FeaProgressEvent {
    let fraction = match (current, total) {
        (Some(current), Some(total)) if total > 0 => Some((current as f64 / total as f64).min(1.0)),
        _ => None,
    };
    let mut metadata = BTreeMap::from([(UNIT_METADATA.to_string(), unit.to_string())]);
    if let Some(stage) = stage {
        metadata.insert(STAGE_METADATA.into(), stage_name(stage));
    }
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
#[path = "meshing_tests.rs"]
mod tests;
