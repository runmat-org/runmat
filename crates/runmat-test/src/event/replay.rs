use crate::error::TestDomainError;
use crate::identity::RunId;
use crate::result::{AttemptResult, RunResult};

use super::{TestEvent, TestEventPayload};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayedEvents {
    pub run_id: RunId,
    pub attempts: Vec<AttemptResult>,
    pub result: RunResult,
}

pub fn replay(events: &[TestEvent]) -> Result<ReplayedEvents, TestDomainError> {
    let first = events
        .first()
        .ok_or(TestDomainError::IncompleteEventStream)?;
    if !matches!(first.payload, TestEventPayload::RunStarted)
        || !matches!(
            events.last().map(|event| &event.payload),
            Some(TestEventPayload::RunFinished { .. })
        )
    {
        return Err(TestDomainError::IncompleteEventStream);
    }
    let run_id = first.run_id.clone();
    let mut attempts = Vec::new();
    let mut result = None;
    for (expected, event) in events.iter().enumerate() {
        if event.sequence != expected as u64 {
            return Err(TestDomainError::EventSequence {
                expected: expected as u64,
                actual: event.sequence,
            });
        }
        if event.run_id != run_id {
            return Err(TestDomainError::EventRunMismatch);
        }
        match &event.payload {
            TestEventPayload::TestFinished { result } => attempts.push(result.clone()),
            TestEventPayload::RunFinished { result: run_result } => {
                if result.is_some() || event.sequence + 1 != events.len() as u64 {
                    return Err(TestDomainError::IncompleteEventStream);
                }
                result = Some(run_result.clone());
            }
            _ => {}
        }
    }
    let result = result.ok_or(TestDomainError::IncompleteEventStream)?;
    Ok(ReplayedEvents {
        run_id,
        attempts,
        result,
    })
}
