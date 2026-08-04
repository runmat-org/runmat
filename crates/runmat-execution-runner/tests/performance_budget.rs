mod common;

use std::time::{Duration, Instant};

use runmat_execution::task::RetryPolicy;
use runmat_execution_runner::driver::DriverCommand;
use runmat_execution_runner::port::BackendReport;

// Keep this large enough to expose accidental superlinear scheduler work while
// remaining a stable warnings-denied CI gate on unoptimized builders.
const TASK_COUNT: usize = 1_000;
const SCHEDULER_BUDGET: Duration = Duration::from_secs(10);
const CHECKPOINT_BUDGET: Duration = Duration::from_secs(2);

#[test]
fn scheduler_and_checkpoint_have_explicit_debug_build_budgets() {
    let mut fixture = common::fixture(1, 1);
    let scheduler_started = Instant::now();
    for ordinal in 0..TASK_COUNT {
        let submission = common::task(
            &format!("performance-{ordinal}"),
            fixture.scope,
            fixture.pool,
            RetryPolicy::Never,
        );
        let request = common::submit(&mut fixture.driver, submission);
        fixture
            .driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &request,
                common::success(),
            )))
            .unwrap();
    }
    let scheduler_elapsed = scheduler_started.elapsed();
    assert!(
        scheduler_elapsed <= SCHEDULER_BUDGET,
        "{TASK_COUNT} serial scheduler commits took {scheduler_elapsed:?}, exceeding the debug-build budget {SCHEDULER_BUDGET:?}"
    );

    let checkpoint_started = Instant::now();
    let checkpoint = serde_json::to_vec(&fixture.driver.snapshot()).unwrap();
    let checkpoint_elapsed = checkpoint_started.elapsed();
    assert!(
        checkpoint_elapsed <= CHECKPOINT_BUDGET,
        "checkpoint encoding took {checkpoint_elapsed:?}, exceeding the debug-build budget {CHECKPOINT_BUDGET:?}"
    );
    assert!(
        checkpoint.len() < 32 * 1024 * 1024,
        "checkpoint exceeded the bounded 32 MiB smoke budget"
    );

    eprintln!(
        "execution performance: tasks={TASK_COUNT} scheduler_ms={} checkpoint_ms={} checkpoint_bytes={}",
        scheduler_elapsed.as_millis(),
        checkpoint_elapsed.as_millis(),
        checkpoint.len()
    );
}
