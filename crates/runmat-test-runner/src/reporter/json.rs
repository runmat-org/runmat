use runmat_test::event::TestEvent;
use runmat_test::result::RunResult;

use super::{RenderedReport, Reporter};
use crate::{RunnerError, RunnerResult};

#[derive(Default)]
pub struct JsonReporter {
    events: Vec<TestEvent>,
}

impl Reporter for JsonReporter {
    fn event(&mut self, event: &TestEvent) -> RunnerResult<()> {
        self.events.push(event.clone());
        Ok(())
    }

    fn finish(&mut self, result: &RunResult) -> RunnerResult<RenderedReport> {
        let bytes = serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "result": result,
            "events": self.events,
        }))
        .map_err(|error| RunnerError::Reporter(error.to_string()))?;
        Ok(RenderedReport {
            name: "test-results.json".into(),
            media_type: "application/json".into(),
            bytes,
        })
    }
}
