use crate::identity::RunId;

use super::{EventSink, TestEvent, TestEventPayload};

pub struct SequencedEventSink<'a, S> {
    run_id: RunId,
    next_sequence: u64,
    sink: &'a mut S,
}

impl<'a, S: EventSink> SequencedEventSink<'a, S> {
    pub fn new(run_id: RunId, sink: &'a mut S) -> Self {
        Self {
            run_id,
            next_sequence: 0,
            sink,
        }
    }

    pub fn emit(&mut self, payload: TestEventPayload) {
        let event = TestEvent {
            sequence: self.next_sequence,
            run_id: self.run_id.clone(),
            payload,
        };
        self.next_sequence += 1;
        self.sink.emit(event);
    }

    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }
}
