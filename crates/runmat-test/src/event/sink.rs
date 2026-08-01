use super::TestEvent;

pub trait EventSink {
    fn emit(&mut self, event: TestEvent);
}

impl EventSink for Vec<TestEvent> {
    fn emit(&mut self, event: TestEvent) {
        self.push(event);
    }
}
