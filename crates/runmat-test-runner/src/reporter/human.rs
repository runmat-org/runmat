use runmat_test::event::TestEvent;
use runmat_test::result::RunResult;

use super::location::{attempt_diagnostic, source_label};
use super::{RenderedReport, ReportSummary, Reporter};
use crate::RunnerResult;

#[derive(Default)]
pub struct HumanReporter {
    process_output: super::process_output::ProcessOutputReports,
}

impl Reporter for HumanReporter {
    fn event(&mut self, event: &TestEvent) -> RunnerResult<()> {
        self.process_output.observe(event);
        Ok(())
    }

    fn finish(&mut self, result: &RunResult) -> RunnerResult<RenderedReport> {
        let summary = ReportSummary::from_result(result);
        let mut lines = result
            .tests
            .iter()
            .map(|test| {
                let label = test
                    .report_identity
                    .as_ref()
                    .map(runmat_test::result::TestReportIdentity::human_label)
                    .unwrap_or_else(|| test.test_id.as_str().to_owned());
                let location = test
                    .attempts
                    .last()
                    .and_then(attempt_diagnostic)
                    .and_then(source_label)
                    .map(|location| format!(" ({location})"))
                    .unwrap_or_default();
                format!("{label} ... {:?}{location}", test.state.disposition)
            })
            .collect::<Vec<_>>();
        lines.push(format!(
            "{} tests: {} passed, {} failed, {} incomplete",
            summary.total, summary.passed, summary.failed, summary.incomplete
        ));
        lines.push(format!(
            "Run: {:?} (incomplete={})",
            result.state.disposition, result.state.incomplete
        ));
        let output = self.process_output.text();
        if !output.is_empty() {
            lines.push(output);
        }
        Ok(RenderedReport {
            name: "test-results.txt".into(),
            media_type: "text/plain; charset=utf-8".into(),
            bytes: format!("{}\n", lines.join("\n")).into_bytes(),
        })
    }
}
