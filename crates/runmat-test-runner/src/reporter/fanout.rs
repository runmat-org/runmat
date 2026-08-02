use runmat_test::event::TestEvent;
use runmat_test::result::RunResult;

use super::{RenderedReport, Reporter};
use crate::RunnerResult;

#[derive(Default)]
pub struct ReporterFanout {
    reporters: Vec<Box<dyn Reporter>>,
}

impl ReporterFanout {
    pub fn push(&mut self, reporter: impl Reporter + 'static) {
        self.reporters.push(Box::new(reporter));
    }

    pub fn event(&mut self, event: &TestEvent) -> RunnerResult<()> {
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
