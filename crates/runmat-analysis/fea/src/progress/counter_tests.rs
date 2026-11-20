use super::*;
use std::sync::Arc;

#[test]
fn a_short_count_stays_silent_and_a_long_one_ends_at_its_total() {
    let events = Arc::new(Mutex::new(Vec::<FeaProgressEvent>::new()));
    let sink = Arc::clone(&events);
    let handler: FeaProgressHandler = Arc::new(move |event| sink.lock().unwrap().push(event));

    let quick = CountReporter::with_handler(
        Some(Arc::clone(&handler)),
        "test",
        FeaProgressPhase::ModelAssembly,
        "elements",
        "elements",
        100,
    );
    quick.add(100);
    quick.finish();
    assert!(events.lock().unwrap().is_empty());

    let slow = CountReporter::with_handler(
        Some(handler),
        "test",
        FeaProgressPhase::ModelAssembly,
        "elements",
        "elements",
        1_000,
    );
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..250 {
                    slow.add(1);
                    std::thread::sleep(Duration::from_millis(3));
                }
            });
        }
    });
    slow.finish();
    let events = events.lock().unwrap();
    assert!(events.len() >= 2, "{}", events.len());
    let counts = events
        .iter()
        .map(|event| event.current.unwrap())
        .collect::<Vec<_>>();
    assert!(
        counts.windows(2).all(|pair| pair[0] <= pair[1]),
        "{counts:?}"
    );
    let last = events.last().unwrap();
    assert_eq!(
        (last.current, last.total, last.fraction),
        (Some(1_000), Some(1_000), Some(1.0))
    );
    assert_eq!(last.metadata[UNIT_METADATA], "elements");
    assert!(events
        .iter()
        .all(|event| event.status == FeaProgressStatus::Advanced));
}
