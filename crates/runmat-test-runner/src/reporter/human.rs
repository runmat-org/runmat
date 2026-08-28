use runmat_test::event::{TestEvent, TestEventPayload};
use runmat_test::result::RunResult;

use super::location::{attempt_diagnostic, source_label};
use super::{RenderedReport, ReportSummary, Reporter};
use crate::RunnerResult;

#[derive(Default)]
pub struct HumanReporter {
    lines: Vec<String>,
}

impl Reporter for HumanReporter {
    fn event(&mut self, event: &TestEvent) -> RunnerResult<()> {
        if let TestEventPayload::TestFinished { result } = &event.payload {
            let location = attempt_diagnostic(result)
                .and_then(source_label)
                .map(|location| format!(" ({location})"))
                .unwrap_or_default();
            self.lines.push(format!(
                "{} ... {:?}{}",
                result.test_id.as_str(),
                result.state.disposition,
                location
            ));
        }
        Ok(())
    }

    fn finish(&mut self, result: &RunResult) -> RunnerResult<RenderedReport> {
        let summary = ReportSummary::from_result(result);
        self.lines.push(format!(
            "{} tests: {} passed, {} failed, {} incomplete",
            summary.total, summary.passed, summary.failed, summary.incomplete
        ));
        Ok(RenderedReport {
            name: "test-results.txt".into(),
            media_type: "text/plain; charset=utf-8".into(),
            bytes: format!("{}\n", self.lines.join("\n")).into_bytes(),
        })
    }
}
