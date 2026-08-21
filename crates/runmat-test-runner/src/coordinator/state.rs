use runmat_test::event::{TestEvent, TestEventPayload};
use runmat_test::identity::RunId;

use crate::reporter::ReporterFanout;
use crate::RunnerResult;

pub(super) struct EventState<'a> {
    run_id: RunId,
    sequence: u64,
    events: Vec<TestEvent>,
    reporters: &'a mut ReporterFanout,
}

impl<'a> EventState<'a> {
    pub fn new(run_id: RunId, reporters: &'a mut ReporterFanout) -> Self {
        Self {
            run_id,
            sequence: 0,
            events: Vec::new(),
            reporters,
        }
    }

    pub fn emit(&mut self, payload: TestEventPayload) -> RunnerResult<()> {
        let event = TestEvent {
            sequence: self.sequence,
            run_id: self.run_id.clone(),
            payload,
        };
        self.sequence += 1;
        self.reporters.event(&event)?;
        self.events.push(event);
        Ok(())
    }

    pub fn forward(&mut self, event: TestEvent) -> RunnerResult<()> {
        if event.run_id != self.run_id {
            return Err(crate::RunnerError::Protocol(
                "worker event changed run identity".into(),
            ));
        }
        if matches!(
            event.payload,
            TestEventPayload::RunStarted
                | TestEventPayload::RunFinished { .. }
                | TestEventPayload::TestStarted { .. }
                | TestEventPayload::TestFinished { .. }
        ) {
            return Ok(());
        }
        self.emit(event.payload)
    }

    pub fn finish(self) -> Vec<TestEvent> {
        self.events
    }
}
