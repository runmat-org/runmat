use runmat_test::event::TestEvent;
use runmat_test::result::RunResult;

use super::{RenderedReport, Reporter};
use crate::RunnerResult;

#[derive(Default)]
pub struct TapReporter;

impl Reporter for TapReporter {
    fn event(&mut self, _event: &TestEvent) -> RunnerResult<()> {
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
                test.test_id.as_str().replace('\n', " ")
            ));
        }
        Ok(RenderedReport {
            name: "test-results.tap".into(),
            media_type: "application/tap".into(),
            bytes: tap.into_bytes(),
        })
    }
}
