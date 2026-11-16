use super::*;
use std::sync::{Arc, Mutex};

fn report(phase: ProgressPhase, iteration: usize, residual: f64) {
    report_solver_progress(Progress {
        phase,
        iteration,
        residual,
        converged: 0,
    });
}

#[test]
fn krylov_iterations_and_factorization_reach_the_installed_handler() {
    let events = Arc::new(Mutex::new(Vec::<FeaProgressEvent>::new()));
    let sink = Arc::clone(&events);
    let _context = super::super::replace_fea_progress_context(super::super::FeaProgressContext {
        handler: Some(Arc::new(move |event| sink.lock().unwrap().push(event))),
        ..Default::default()
    });
    report(ProgressPhase::NumericFactorization, 0, 0.0);
    report(ProgressPhase::FactorizationComplete, 0, 0.0);
    // Within the event interval of the factorization events.
    report(ProgressPhase::Krylov, 16, 1.0e-3);
    std::thread::sleep(MINIMUM_EVENT_INTERVAL);
    report(ProgressPhase::Krylov, 32, 2.5e-6);
    // A finish without a reported start is not sent.
    report(ProgressPhase::FactorizationComplete, 0, 0.0);

    let events = events.lock().unwrap();
    assert_eq!(events.len(), 3, "{events:?}");
    assert!(events.iter().all(|event| event.phase == FeaProgressPhase::Solve
        && event.status == FeaProgressStatus::Advanced));
    assert_eq!(events[0].metadata[UNIT_METADATA], "factorization");
    assert_eq!((events[0].current, events[0].total), (Some(0), Some(1)));
    assert_eq!((events[1].current, events[1].total), (Some(1), Some(1)));
    assert!(events[1].metadata.contains_key("factorizationMs"));
    assert_eq!(events[2].current, Some(32));
    assert_eq!(events[2].metadata["iteration"], "32");
    assert_eq!(events[2].metadata["residual"], "2.5e-6");
    assert_eq!(events[2].message, "Iteration 32, relative residual 2.50e-6");
}
