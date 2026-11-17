use super::*;
use runmat_analysis_fea::progress::rate::UNIT_METADATA;
use runmat_analysis_fea::FeaProgressPhase;
use runmat_meshing_core::work_count::{MeshingWorkCount, MeshingWorkUnit};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn recording() -> (ProgressObserverSlot, Arc<Mutex<Vec<FeaProgressEvent>>>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    (
        ProgressObserverSlot::new(Some(Arc::new(move |event| sink.lock().unwrap().push(event)))),
        events,
    )
}

fn work_frame(sequence: u64, count: MeshingWorkCount) -> ProgramProgress {
    ProgramProgress {
        schema_version: crate::protocol::NATIVE_WORKER_MESSAGE_SCHEMA_V2,
        sequence,
        media_type: crate::meshing_host::progress::MESHING_WORK_MEDIA_TYPE.into(),
        value_schema: crate::meshing_host::progress::MESHING_WORK_VALUE_SCHEMA.into(),
        payload: serde_json::to_vec(&count).unwrap(),
    }
}

fn faces(index: u32, faces: u64) -> MeshingPartitionFaces {
    MeshingPartitionFaces {
        index,
        faces: Some(faces),
    }
}

#[test]
fn surface_partitions_sum_to_a_model_wide_face_count_that_ends_at_its_total() {
    let (slot, events) = recording();
    let mut tracker = MeshingProgressTracker::new(slot);
    let stage = MeshingStageKind::SurfaceMesh;
    tracker.stage_started(stage, faces(0, 6));
    tracker.stage_started(stage, faces(1, 4));
    for current in 1..=6 {
        tracker.record(
            stage,
            faces(0, 6),
            &[work_frame(
                current,
                MeshingWorkCount {
                    unit: MeshingWorkUnit::SurfaceFaces,
                    current,
                    total: Some(6),
                },
            )],
        );
    }
    tracker.stage_completed(stage, faces(0, 6));
    // The second partition was restored from cache and reported nothing itself.
    tracker.stage_completed(stage, faces(1, 4));

    let events = events.lock().unwrap();
    let counts = events
        .iter()
        .filter(|event| event.metadata.get(UNIT_METADATA).map(String::as_str) == Some("surface_faces"))
        .map(|event| (event.current.unwrap(), event.total.unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(counts.first(), Some(&(1, 10)));
    assert_eq!(counts.last(), Some(&(10, 10)));
    assert!(counts.windows(2).all(|pair| pair[0].0 <= pair[1].0), "{counts:?}");
    assert!(events.iter().all(|event| event.phase == FeaProgressPhase::MeshPrep
        && event.status == FeaProgressStatus::Advanced
        && event.metadata.contains_key("stage")));
}

#[test]
fn volume_points_and_quadratic_edges_become_mesh_events() {
    let (slot, events) = recording();
    let mut tracker = MeshingProgressTracker::new(slot);
    let whole = MeshingPartitionFaces {
        index: 0,
        faces: None,
    };
    tracker.record(
        MeshingStageKind::Tetrahedralization,
        whole,
        &[work_frame(
            1,
            MeshingWorkCount {
                unit: MeshingWorkUnit::VolumePoints,
                current: 4096,
                total: None,
            },
        )],
    );
    tracker.record(
        MeshingStageKind::OrderElevation,
        whole,
        &[work_frame(
            2,
            MeshingWorkCount {
                unit: MeshingWorkUnit::QuadraticEdges,
                current: 2048,
                total: Some(8192),
            },
        )],
    );
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!((events[0].current, events[0].total), (Some(4096), None));
    assert_eq!(events[0].metadata[UNIT_METADATA], "volume_points");
    assert_eq!(events[0].metadata["stage"], "tetrahedralization");
    assert_eq!(events[1].fraction, Some(0.25));
    assert_eq!(events[1].metadata["stage"], "order_elevation");
}

#[test]
fn analysis_frames_decode_back_to_fea_events() {
    let (slot, events) = recording();
    let event = FeaProgressEvent {
        operation: "fea.assembly".into(),
        phase: FeaProgressPhase::ModelAssembly,
        status: FeaProgressStatus::Advanced,
        message: "10 of 20 elements".into(),
        current: Some(10),
        total: Some(20),
        fraction: Some(0.5),
        metadata: BTreeMap::new(),
    };
    let frame = ProgramProgress {
        schema_version: crate::protocol::NATIVE_WORKER_MESSAGE_SCHEMA_V2,
        sequence: 1,
        media_type: crate::analysis_progress::FEA_PROGRESS_MEDIA_TYPE.into(),
        value_schema: crate::analysis_progress::FEA_PROGRESS_VALUE_SCHEMA.into(),
        payload: serde_json::to_vec(&event).unwrap(),
    };
    frame.validate().unwrap();
    let mut boundary = event.clone();
    boundary.status = FeaProgressStatus::Started;
    let boundary = ProgramProgress {
        sequence: 3,
        payload: serde_json::to_vec(&boundary).unwrap(),
        ..frame.clone()
    };
    forward_analysis_progress(&slot, vec![frame, boundary, work_frame(2, MeshingWorkCount {
        unit: MeshingWorkUnit::VolumePoints,
        current: 1,
        total: None,
    })]);
    assert_eq!(*events.lock().unwrap(), vec![event]);
}
