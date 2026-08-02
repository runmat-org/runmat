use runmat_test::event::TestEvent;
use runmat_test::result::RunResult;

use super::{EventObserver, RenderedReport, Reporter};
use crate::RunnerResult;

#[derive(Default)]
pub struct ReporterFanout {
    reporters: Vec<Box<dyn Reporter>>,
    observers: Vec<Box<dyn EventObserver>>,
}

impl ReporterFanout {
    pub fn push(&mut self, reporter: impl Reporter + 'static) {
        self.reporters.push(Box::new(reporter));
    }

    pub fn push_observer(&mut self, observer: impl EventObserver + 'static) {
        self.observers.push(Box::new(observer));
    }

    pub fn event(&mut self, event: &TestEvent) -> RunnerResult<()> {
        for observer in &mut self.observers {
            observer.event(event)?;
        }
        for reporter in &mut self.reporters {
            reporter.event(event)?;
        }
        Ok(())
    }

    pub fn finish(&mut self, result: &RunResult) -> RunnerResult<Vec<RenderedReport>> {
        self.reporters
            .iter_mut()
            .map(|reporter| reporter.finish(result))
            .collect()
    }
}
