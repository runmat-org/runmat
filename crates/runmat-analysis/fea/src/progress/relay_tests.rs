use super::*;
use crate::progress::FeaProgressPhase;
use std::time::Duration;

fn recording() -> (Arc<ProgressRelay>, Arc<Mutex<Vec<FeaProgressEvent>>>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let relay = ProgressRelay::new(Arc::new(move |event| sink.lock().unwrap().push(event)));
    (relay, events)
}

fn event(status: FeaProgressStatus, current: Option<u64>) -> FeaProgressEvent {
    FeaProgressEvent {
        operation: "test".into(),
        phase: FeaProgressPhase::MeshPrep,
        status,
        message: "Meshing faces".into(),
        current,
        total: current.map(|_| 100),
        fraction: current.map(|current| current as f64 / 100.0),
        metadata: BTreeMap::new(),
    }
}

#[test]
fn heartbeats_repeat_the_last_count_every_two_seconds_while_a_phase_is_open() {
    let (relay, events) = recording();
    let origin = Instant::now();
    let after = |millis| origin + Duration::from_millis(millis);
    assert!(!relay.heartbeat_at(after(5_000)), "no phase is open yet");
    relay.emit_at(event(FeaProgressStatus::Started, None), after(0));
    relay.emit_at(event(FeaProgressStatus::Advanced, Some(40)), after(500));
    assert!(!relay.heartbeat_at(after(2_000)));
    assert!(relay.heartbeat_at(after(2_500)));
    assert!(!relay.heartbeat_at(after(3_000)));
    assert!(relay.heartbeat_at(after(4_500)));
    relay.emit_at(event(FeaProgressStatus::Completed, None), after(4_600));
    assert!(!relay.heartbeat_at(after(9_000)), "a closed phase sends no heartbeats");

    let events = events.lock().unwrap();
    let heartbeats = events
        .iter()
        .filter(|event| event.metadata.contains_key(HEARTBEAT_METADATA))
        .collect::<Vec<_>>();
    assert_eq!(heartbeats.len(), 2);
    for (heartbeat, elapsed) in heartbeats.iter().zip(["2500", "4500"]) {
        assert_eq!(heartbeat.status, FeaProgressStatus::Advanced);
        assert_eq!(heartbeat.current, Some(40));
        assert_eq!(heartbeat.message, "Meshing faces");
        assert_eq!(heartbeat.metadata[ELAPSED_METADATA], elapsed);
    }
    assert_eq!(events.len(), 5);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn the_heartbeat_thread_stops_with_its_guard() {
    let (relay, events) = recording();
    relay.emit(event(FeaProgressStatus::Started, None));
    let guard = relay.spawn_heartbeat();
    drop(guard);
    assert_eq!(events.lock().unwrap().len(), 1);
}
