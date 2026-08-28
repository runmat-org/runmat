use crate::identity::TestId;

use super::{AttemptResult, ResultState, TerminalDisposition, TestResult};

pub fn merge_attempts(test_id: TestId, mut attempts: Vec<AttemptResult>) -> Option<TestResult> {
    if attempts.is_empty() || attempts.iter().any(|attempt| attempt.test_id != test_id) {
        return None;
    }
    attempts.sort_by_key(|attempt| attempt.attempt);
    if attempts
        .windows(2)
        .any(|pair| pair[0].attempt == pair[1].attempt)
    {
        return None;
    }
    let final_attempt = attempts.last()?;
    let flaky = final_attempt.state.is_success()
        && attempts
            .iter()
            .take(attempts.len() - 1)
            .any(|attempt| !attempt.state.is_success());
    Some(TestResult {
        test_id,
        state: final_attempt.state,
        attempts,
        flaky,
    })
}

pub fn aggregate_run_state<'a>(states: impl IntoIterator<Item = &'a ResultState>) -> ResultState {
    let mut aggregate = ResultState::PASSED;
    for state in states {
        aggregate.failed |= state.failed;
        aggregate.incomplete |= state.incomplete;
        aggregate.disposition = stronger_disposition(aggregate.disposition, state.disposition);
    }
    aggregate
}

fn stronger_disposition(
    left: TerminalDisposition,
    right: TerminalDisposition,
) -> TerminalDisposition {
    match (rank(left), rank(right)) {
        (left_rank, right_rank) if right_rank > left_rank => right,
        _ => left,
    }
}

fn rank(value: TerminalDisposition) -> u8 {
    use TerminalDisposition::*;
    match value {
        Passed => 0,
        Filtered => 1,
        Failed => 2,
        Cancelled => 3,
        TimedOut => 4,
        Crashed => 5,
    }
}
