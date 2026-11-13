use runmat_test::event::TestEvent;
use runmat_test::result::RunResult;

use super::{RenderedReport, Reporter};
use crate::RunnerResult;

#[derive(Default)]
pub struct TapReporter {
    process_output: super::process_output::ProcessOutputReports,
}

impl Reporter for TapReporter {
    fn event(&mut self, event: &TestEvent) -> RunnerResult<()> {
        self.process_output.observe(event);
        Ok(())
    }

    fn finish(&mut self, result: &RunResult) -> RunnerResult<RenderedReport> {
        let mut tap = format!("TAP version 13\n1..{}\n", result.tests.len());
        for (index, test) in result.tests.iter().enumerate() {
            tap.push_str(&format!(
                "{} {} - {}\n",
                if test.state.is_success() {
                    "ok"
                } else {
                    "not ok"
                },
                index + 1,
                test.report_identity
                    .as_ref()
                    .map(runmat_test::result::TestReportIdentity::human_label)
                    .unwrap_or_else(|| test.test_id.as_str().to_owned())
                    .replace('\n', " ")
            ));
        }
        tap.push_str(&self.process_output.tap());
        Ok(RenderedReport {
            name: "test-results.tap".into(),
            media_type: "application/tap".into(),
            bytes: tap.into_bytes(),
        })
    }
}
