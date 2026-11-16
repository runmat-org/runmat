use super::*;
use std::sync::Mutex;

#[test]
fn installed_observer_receives_counts_and_volume_points_accumulate() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    {
        let _guard = install_work_observer(Arc::new(move |count| sink.lock().unwrap().push(count)));
        report_work(MeshingWorkUnit::SurfaceFaces, 3, Some(10));
        add_volume_points(32);
        add_volume_points(0);
        add_volume_points(8);
    }
    report_work(MeshingWorkUnit::SurfaceFaces, 4, Some(10));
    let seen = seen.lock().unwrap();
    assert_eq!(
        *seen,
        vec![
            MeshingWorkCount {
                unit: MeshingWorkUnit::SurfaceFaces,
                current: 3,
                total: Some(10),
            },
            MeshingWorkCount {
                unit: MeshingWorkUnit::VolumePoints,
                current: 32,
                total: None,
            },
            MeshingWorkCount {
                unit: MeshingWorkUnit::VolumePoints,
                current: 40,
                total: None,
            },
        ]
    );
    let wire = serde_json::to_string(&seen[1]).unwrap();
    assert_eq!(wire, r#"{"unit":"volume_points","current":32}"#);
}
