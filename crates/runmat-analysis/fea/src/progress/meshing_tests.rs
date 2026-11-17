use super::*;

fn faces(current: u64, total: u64) -> MeshingWorkCount {
    MeshingWorkCount {
        unit: MeshingWorkUnit::SurfaceFaces,
        current,
        total: Some(total),
    }
}

fn counts(event: Option<FeaProgressEvent>) -> (u64, u64) {
    let event = event.expect("a surface event");
    (event.current.unwrap(), event.total.unwrap())
}

#[test]
fn known_surface_partitions_sum_to_a_model_wide_count() {
    let mut map = MeshingProgressMap::new();
    let stage = MeshingStageKind::SurfaceMesh;
    let first = MeshingPartitionFaces {
        index: 0,
        faces: Some(6),
    };
    let second = MeshingPartitionFaces {
        index: 1,
        faces: Some(4),
    };
    let started = map.stage_started(stage, first);
    assert_eq!(started.message, "Meshing faces");
    assert_eq!(started.metadata[UNIT_METADATA], "stages");
    assert_eq!(started.metadata[STAGE_METADATA], "surface_mesh");
    map.stage_started(stage, second);
    assert_eq!(counts(map.work_count(Some(stage), Some(0), &faces(1, 6))), (1, 10));
    assert_eq!(counts(map.work_count(Some(stage), Some(0), &faces(6, 6))), (6, 10));
    // A partition restored from cache reports nothing until it finishes.
    let finished = map.stage_completed(stage, second);
    assert_eq!(counts(finished.clone()), (10, 10));
    let finished = finished.unwrap();
    assert_eq!(finished.message, "Meshed 10 of 10 faces");
    assert_eq!(finished.metadata[UNIT_METADATA], "surface_faces");
    assert_eq!(finished.operation, MESHING_OPERATION);
    assert_eq!(finished.phase, FeaProgressPhase::MeshPrep);
}

#[test]
fn unknown_surface_partitions_are_inferred_from_restarting_counts() {
    let mut map = MeshingProgressMap::new();
    let mut record = |current, total| counts(map.work_count(None, None, &faces(current, total)));
    assert_eq!(record(1, 8), (1, 8));
    assert_eq!(record(5, 8), (5, 8));
    assert_eq!(record(8, 8), (8, 8));
    // The next partition counts from its own start.
    assert_eq!(record(1, 3), (9, 11));
    // A restart finishes the open partition with all of its faces.
    assert_eq!(record(0, 4).0, 11);
    assert_eq!(record(0, 4), (11, 15));
    assert_eq!(record(4, 4), (15, 15));
}

#[test]
fn volume_points_and_quadratic_edges_carry_their_stage_when_known() {
    let mut map = MeshingProgressMap::new();
    let points = map
        .work_count(
            Some(MeshingStageKind::Tetrahedralization),
            None,
            &MeshingWorkCount {
                unit: MeshingWorkUnit::VolumePoints,
                current: 4096,
                total: None,
            },
        )
        .unwrap();
    assert_eq!((points.current, points.total), (Some(4096), None));
    assert_eq!(points.message, "Inserted 4096 volume mesh points");
    assert_eq!(points.metadata[UNIT_METADATA], "volume_points");
    assert_eq!(points.metadata[STAGE_METADATA], "tetrahedralization");
    let edges = map
        .work_count(
            None,
            None,
            &MeshingWorkCount {
                unit: MeshingWorkUnit::QuadraticEdges,
                current: 2048,
                total: Some(8192),
            },
        )
        .unwrap();
    assert_eq!(edges.fraction, Some(0.25));
    assert_eq!(edges.message, "Placed midside nodes on 2048 of 8192 edges");
    assert!(!edges.metadata.contains_key(STAGE_METADATA));
}
